//     vartivity.cpp: Implement variable activity interface for OpenBT.
//     Copyright (C) 2012-2018 Matthew T. Pratola.
//
//     This file is part of OpenBT.
//
//     OpenBT is free software: you can redistribute it and/or modify
//     it under the terms of the GNU Affero General Public License as published by
//     the Free Software Foundation, either version 3 of the License, or
//     (at your option) any later version.
//
//     OpenBT is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//     GNU Affero General Public License for more details.
//
//     You should have received a copy of the GNU Affero General Public License
//     along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
//     Author contact information
//     Matthew T. Pratola: mpratola@gmail.com


#include <iostream>
#include <string>
#include <ctime>
#include <sstream>
#include <chrono>

#include <fstream>
#include <vector>
#include <limits>

#include "crn.h"
#include "tree.h"
#include "brt.h"
#include "brtfuns.h"
#include "dinfo.h"
#include "mxbrt.h"
#include "amxbrt.h"
#include "psbrt.h"

using std::cout;
using std::endl;

int main(int argc, char* argv[])
{
    std::string folder("");

    if(argc>1)
    {
        //argument on the command line is path to config file.
        folder=std::string(argv[1]);
        folder=folder+"/";
    }

    //--------------------------------------------------
    // Banner
    cout << endl;
    cout << "-----------------------------------" << endl;
    cout << "OpenBT Random Path Variogram CLI" << endl;
    cout << "Loading config file at " << folder << endl;

    //--------------------------------------------------
    //process args
    std::ifstream conf(folder+"config.variogram");


    // model name, number of samples in the approximation and number of trees
    std::string modelname;
    size_t nd;
    size_t m;

    conf >> modelname;
    conf >> nd;
    conf >> m;
    
    // number of predictors
    size_t p;
    conf >> p;

    // Prior information
    double tau2;
    double base;
    double power;
    double shape1;
    double shape2;
    double q;
    double gamma0; // fixed value of gamma, used if const_gamma = True
    bool const_gamma = false;
    std::string const_gamma_str;

    conf >> tau2;
    conf >> base;
    conf >> power;
    conf >> shape1;
    conf >> shape2;
    conf >> q;
    conf >> gamma0;
    conf >> const_gamma_str;
    if(const_gamma_str == "TRUE" || const_gamma_str == "True"){const_gamma_str = true;}

    // xcut point and xgrid info
    std::string xicore;
    std::string xcore;
    std::string hcore;
    conf >> xicore;
    conf >> xcore;
    conf >> hcore;

    conf.close();

    // Read in xgrid
    std::vector<double> x, xh;
    double xtemp;
    std::stringstream xfss;
    std::string xfs;
    size_t n;
    
    xfss << folder << xcore;
    xfs=xfss.str();
    std::ifstream xf(xfs);
    while(xf >> xtemp)
        x.push_back(xtemp);
    n = x.size()/p;

    // Read in cuts and make xinfo
    xinfo xi;
    xi.resize(p);

    for(size_t i=0;i<p;i++) {
        std::vector<double> xivec;
        double xitemp;
        std::stringstream xifss;
        std::string xifs;

        xifss << folder << xicore << (i+1);
        xifs=xifss.str();
        std::ifstream xif(xifs);
        while(xif >> xitemp)
            xivec.push_back(xitemp);
        xi[i]=xivec;
    }


    // Read in h grid
    std::vector<double> h;
    double htemp;
    std::stringstream hfss;
    std::string hfs;
    size_t nh;
    
    hfss << folder << hcore;
    hfs=hfss.str();
    std::ifstream hf(hfs);
    while(hf >> htemp){
        h.push_back(htemp);
    }
    nh = h.size();

    // Random Number Generator
    crn gen, gen_gam1, gen_gam2;
    gen.set_seed(static_cast<long long>(std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()));

    gen_gam1.set_seed(static_cast<long long>(std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()));

    gen_gam2.set_seed(static_cast<long long>(std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()));

    gen_gam1.set_gam(shape1,1);
    gen_gam2.set_gam(shape2,1);

    // Numerically compute the variogram by drawing from the prior
    amxbrt axb(m);
    std::vector<double> gammavec(m,gamma0); 
    std::vector<mxd, Eigen::aligned_allocator<mxd>> phix_list, phixh_list;

    // Set the required items for the axb instance
    axb.setgamma(gammavec);
    axb.setxi(&xi);
    axb.settp(base, power);
    axb.setrpi(gamma0,q,shape1,shape2,n);

    dinfo digrid, dihgrid;
    digrid.x = &x[0]; digrid.n = n; digrid.p = p; digrid.tc = 1; digrid.y = NULL;
    

    // Containers for results
    std::vector<std::vector<double>> vg(nd,std::vector<double>(h));
    vxd phibar;

    // Sampling procedure for the variogram
    for(size_t i=0;i<nd;i++){        
        // Sample tree
        axb.sample_tree_prior(gen);

        // Sample gamma if desired
        if(!const_gamma){
            // Draw gammas from beta prior
            for(size_t j=0;j<m;j++){
                double a = gen_gam1.gamma();
                double b = gen_gam2.gamma();
                gammavec[j] = (a/(a+b));
                //cout << "gammavec[j] = " << gammavec[j] << endl;
            }
            // Set the gammas 
            axb.setgamma(gammavec);
        }

        //cout << "gamma is set" << endl;
        //cout << "gammavec[0] = " << gammavec[0] << endl;

        // Reset diter
        diterator diter(&digrid);
        
        // Use diter's to get the phi(x) functions
        phix_list.clear();
        axb.get_phix_list(diter, phix_list, n);

        // Get phix (might need to do this for the h group too....Loop over h starting here....)
        for(size_t k=0;k<h.size();k++){
            
            // Set the h diter
            xh.clear();
            for(size_t l=0;l<x.size();l++){
                xh.push_back(x[l]+h[k]);
                //cout << "x = " << x[l] << endl;
                //cout << "xh = " << xh[l] << endl;
            }

            // Set the h diter
            dihgrid.x = &xh[0]; dihgrid.n = n; dihgrid.p = p; dihgrid.tc = 1; dihgrid.y = NULL;
            diterator dhiter(&dihgrid);

            // Use diter's to get the phi(x) functions
            phixh_list.clear();
            axb.get_phix_list(dhiter, phixh_list, n);

            // Compute the phi(x) product for each pair in each tree
            phibar = vxd::Zero(n);
            for(size_t j=0;j<m;j++){
                for(size_t l=0;l<n;l++){
                    phibar.row(l) = phibar.row(l) + phix_list[j].row(l)*phixh_list[j].row(l).transpose()/m;
                }    
            }

            // Compute variogram and store for this h
            vg[i][k] = 2*tau2*m*(n - phibar.sum())/n; // average over the x's for numeric integral
        }
    }

    // Write the results
    std::ofstream ovf(folder + modelname + ".variogram");
    for(size_t i=0;i<nd;i++){
        for(size_t k=0;k<nh;k++){ovf << vg[i][k] << endl;}
    }
    ovf.close();
    return 0;
}