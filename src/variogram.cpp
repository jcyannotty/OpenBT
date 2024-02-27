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
    double fmean, rscale;
    double beta, sig2;
    std::string const_gamma_str, type;
    size_t maxd;

    conf >> tau2;
    conf >> base;
    conf >> power;
    conf >> maxd;
    conf >> shape1;
    conf >> shape2;
    conf >> q;
    conf >> gamma0;
    conf >> const_gamma_str;
    conf >> fmean;
    conf >> rscale;
    conf >> beta;
    conf >> sig2;
    conf >> type;
    if(const_gamma_str == "TRUE" || const_gamma_str == "True"){const_gamma = true;}

    // xcut point and xgrid info
    std::string xicore;
    std::string xbcore;
    std::string hcore;
    std::string xcore;
    std::string rcore;
    conf >> xicore;
    conf >> xbcore;
    conf >> hcore;
    conf >> xcore;
    conf >> rcore;
    conf.close();

    // Read in x bounds
    std::vector<double> xbnd, x, xh;
    double xtemp;
    std::stringstream xfss;
    std::string xfs;
    //size_t n;
    
    xfss << folder << xbcore;
    xfs=xfss.str();
    std::ifstream xf(xfs);
    while(xf >> xtemp)
        xbnd.push_back(xtemp);
    
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

    // If this a vg for the random process y, read in the x and y values  
    std::vector<double> xlist, xhlist, rlist;
    if(type == "y"){
        double xtemp, xhtemp, rtemp;
        std::stringstream xfss, xhfss, rfss;
        std::string xfs, xhfs, rfs;
        size_t n, nh;

        // x draws
        xfss << folder << xcore;
        xfs=xfss.str();
        std::ifstream xf(xfs);
        while(xf >> xtemp){
            xlist.push_back(xtemp);
        }
        n = xlist.size()/p;

        // R(h) values
        rfss << folder << rcore;
        rfs=rfss.str();
        std::ifstream rf(rfs);
        while(rf >> rtemp){
            rlist.push_back(rtemp);
        }
    }


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
    axb.setmaxd(maxd);
    axb.setrpi(gamma0,q,shape1,shape2,1);

    dinfo digrid, dihgrid;
    
    // Containers for results
    std::vector<std::vector<double>> vg(nd,std::vector<double>(h));
    vxd phibar;

    // Sampling procedure for the variogram
    for(size_t i=0;i<nd;i++){        
        // Print progress
        if(i%5000 == 0) cout << "Step: " << i << endl;
                
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

        // Sample x and set digrid
        x.clear();
        if(type == "w" || type == "b"){
            for(size_t k=0;k<p;k++){
                xtemp = xbnd[2*k] + gen.uniform()*(xbnd[2*k+1]-xbnd[2*k]);
                x.push_back(xtemp);
            }
        }else{
            for(size_t k=0;k<p;k++){
                xtemp = xlist[i*p + k];
                x.push_back(xtemp);
            }
        }
        digrid.x = &x[0]; digrid.n = 1; digrid.p = p; digrid.tc = 1; digrid.y = NULL;


        // Reset diter
        diterator diter(&digrid);
        
        // Use diter's to get the phi(x) functions
        phix_list.clear();
        axb.get_phix_list(diter, phix_list, 1);

        // Get phix (might need to do this for the h group too....Loop over h starting here....)
        for(size_t k=0;k<h.size();k++){
            
            // Set the h diter
            xh.clear();
            for(size_t l=0;l<x.size();l++){
                //xh.push_back(x[l]+h[k]/p);
                xh.push_back(x[l]+h[k]/pow(p,0.5));
            }

            // Set the h diter
            dihgrid.x = &xh[0]; dihgrid.n = 1; dihgrid.p = p; dihgrid.tc = 1; dihgrid.y = NULL;
            diterator dhiter(&dihgrid);

            // Use diter's to get the phi(x) functions
            phixh_list.clear();
            axb.get_phix_list(dhiter, phixh_list, 1);

            // Compute the phi(x) product for each pair in each tree
            phibar = vxd::Zero(1);
            for(size_t j=0;j<m;j++){
                // Used to be looping through n terms - now only one term so can eventually remove the inner loop  
                for(size_t l=0;l<1;l++){
                    phibar.row(l) = phibar.row(l) + phix_list[j].row(l)*phixh_list[j].row(l).transpose()/m;
                }    
            }

            // Compute variogram and store for this h
            //vg[i][k] = 2*tau2*m*(1 - phibar.sum()); // average over the x's for numeric integral
            if(type == "w" || type == "b"){
                vg[i][k] = 2*tau2*m*(1 - phibar(0)) + 2*sig2; // n = 1 now
            }else{
                vg[i][k] = 2*(sig2 + (tau2*m + m*m*beta*beta)*(rscale - rlist[k]) + (fmean*fmean + rlist[k])*tau2*m*(1 - phibar(0)));
            }
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