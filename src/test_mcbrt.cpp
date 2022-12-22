#include <iostream>
#include <fstream>
#include <map>

#include "crn.h"
#include "brt.h"
#include "brtfuns.h"
#include "dinfo.h"
#include "mcbrt.h"

//Include Eigen library
#include "Eigen/Dense"

int main(){
    //---Read in Data for mxbrt examples
    //-------------------------------------------------------
    crn gen;
    gen.set_seed(200);

    int tc=4; //thread count for OpenMP

    //--------------------------------------------------
    //read in y
    std::vector<double> y;
    double ytemp;

    std::ifstream yf("/home/johnyannotty/Documents/Calibration BART/BallDropData/y_balldrop.txt");
    while(yf >> ytemp)
        y.push_back(ytemp);
    size_t n = y.size();
    cout << "n from y.txt: " << n <<endl;
    cout << "y[50]: " << y[50] <<endl;

    //--------------------------------------------------
    //read in x
    std::vector<double> x;
    double xtemp;
    size_t p;
    p=2;

    std::ifstream xf("/home/johnyannotty/Documents/Calibration BART/BallDropData/x_balldrop.txt");
    while(xf >> xtemp){
        x.push_back(xtemp);
    }

    if(x.size() != n*p) {
        cout << "error: input x file has wrong number of values\n";
        return 1;
    }
    cout << "n,p: " << n << ", " << p << endl;

    //--------------------------------------------------
    //read in f
    std::vector<double> f;
    double ftemp;
    size_t k; //number of columns in f
    k=2;

    std::ifstream ff("/home/johnyannotty/Documents/Calibration BART/BallDropData/h_balldrop.txt");
    while(ff >> ftemp){
        f.push_back(ftemp);
    }
    
    if(f.size() != n*k) {
        cout << "error: input f file has wrong number of values\n";
        return 1;
    }
    cout << "n,k: " << n << ", " << k << endl;
    
    //--------------------------------------------------
    //Make dinfo and diterator
    dinfo di;
    di.n=n;di.p=p,di.x = &x[0];di.tc=tc;
    di.y = &y[0];

    diterator diter(&di);
    cout << "diter.gety = " << diter.gety() << endl;
    cout << "diter.getx = " << diter.getx() << endl;
    /*
    for(;diter<diter.until();diter++){
        cout << "diter.gety = " << diter.gety() << endl;
    }
    */

    //--------------------------------------------------
    //make xinfo
    xinfo xi;
    size_t nc=100; //100
    makexinfo(p,n,&x[0],xi,nc); //use the 1st column with x[0]
    //prxi(xi);

    //--------------------------------------------------
    //make finfo -- need to read in and store f formally, just using same x from above for now
    finfo fi;
    makefinfo(k, n, &f[0], fi);
    //cout << fi << endl;

    //--------------------------------------------------
    // read in the initial change of variable rank correlation matrix
    std::vector<std::vector<double> > chgv;
    std::vector<double> cvvtemp;
    double cvtemp;
    std::ifstream chgvf("/home/johnyannotty/Documents/Calibration BART/BallDropData/chgv_balldrop.txt");
    for(size_t i=0;i<di.p;i++) {
        cvvtemp.clear();
        for(size_t j=0;j<di.p;j++) {
            chgvf >> cvtemp;
            cvvtemp.push_back(cvtemp);
        }
        chgv.push_back(cvvtemp);
    }
    cout << "change of variable rank correlation matrix loaded:" << endl;
    for(size_t i=0;i<di.p;i++) {
        for(size_t j=0;j<di.p;j++)
            cout << "(" << i << "," << j << ")" << chgv[i][j] << "  ";
        cout << endl;
    }


    //--------------------------------------------------
    // Test mcinfo
    //--------------------------------------------------
    // (1) Constructors
    mcinfo mci1; // default constructor
    mcinfo mci2(true); // subtree node indicator
    mcinfo mci3(10.5,12.6,7.2,9.4,3); // pass in field and model runs suff stats 
    mci3.n = 12; // set value for n just so the print makes sense 
    mcinfo mci4(mci3); // using another instance of mcinfo

    mci1.print();
    mci2.print();
    mci3.print();
    mci4.print();
    
    //--------------------------------------------------
    // (2) Setting sibling - used for birth/death proposals
    //     & subtree information - used for all tree proposals
    // Left and right children
    mcinfo mcil(10.5,12.6,7.2,9.4,3); mcil.n = 12; mcil.subtree_node = true;
    mcinfo mcir(5.5,5.4,5.8,9.1,2); mcir.n = 10; mcir.subtree_node = true;
    
    // Set sibling info in the right node
    mcir.setsiblinginfo(mcil);
    
    // Now consider other nodes in the subtree and store in a vector
    mcinfo mciv1(1.0,2.0,3.0,4.0,2); mciv1.n = 2; mciv1.subtree_node = true;
    mcinfo mciv2(2.1,4.1,6.1,8.1,4); mciv2.n = 4; mciv2.subtree_node = true;
    std::vector<mcinfo*> mciv(2);
    mciv[0] = &mciv1; mciv[1] = &mciv2;

    // Now set subtree information for mcir
    mcir.setsubtreeinfo(mciv);

    //--------------------------------------------------
    // (3) Get subtree moments (and sibling)
    mcir.setsubtreemoments(2.0,1.0);
    cout << "subtree_means = " << mcir.subtree_mean[0] << ", " << mcir.subtree_mean[1] << endl;
    cout << "subtree_vars = " << mcir.subtree_var[0] << ", " << mcir.subtree_var[1] << endl;
    cout << "sibling_mean = " << mcir.sibling_mean << endl;
    cout << "sibling_var = " << mcir.sibling_var << endl;

    //--------------------------------------------------
    // (4) Operators
    mcinfo mcit;

    // Add mcir and mcil with mcit1, shows addition used in birth step where we need to populate sit using sir and sil
    cout << "\n----------- mcit = mcit + mcir -----------" << endl;
    mcit += mcir; // Add mcir's stats along with the subtree info to mcit using pushback!
    mcit.print();

    // Now add mcil with mcit, only the node specific suff stats should change, the subtree and sibiling stats should not! 
    cout << "\n----------- mcit = mcit + mcil -----------" << endl;
    mcit += mcil;
    mcit.print();

    // Now add mcir with mcir to test the addition operator when adding two nodes with subtree info
    // This feature may be useful in the mpi, particularly the birth and death one where we have
    // to pass subtree info and sibling info between cores
    cout << "\n----------- mcir = mcir + mcir -----------" << endl;
    mcir += mcir; 
    mcir.print();

    cout << "End of mcinfo test..." << endl;
    cout << "---------------------------------------------" << endl;
    cout << "---------------------------------------------\n" << endl;
    //--------------------------------------------------
    // Test mcbrt
    //--------------------------------------------------
    // Set hyperparameters
    double *sig = new double[di.n];
    double tau1 = 0.5, tau2 = 0.25, mu1= 100, mu2 = 5.0;
    std::vector<size_t> uvec(1);
    for(size_t i=0;i<di.n;i++) sig[i]=0.05;    
    cout << "---------------------------------------------" << endl;
    cout << "Testing mcbrt...." << endl;
    
    // Set the index of the calibration parameter and other data components
    mcbrt mc;
    uvec[0] = 1;
    mc.setuvars(uvec);
    mc.setfi(&fi,2);
    mc.setdata_vec(&di);
    mc.setxi(&xi);
    mc.setci(mu1,mu2,tau1,tau2,sig);

    // Create some tree splits
    //Assign left and right vectors for terminal nodes in birth step
    Eigen::VectorXd thetavecl, thetavecr, thetavecz;
    thetavecl = Eigen::VectorXd::Random(2);
    thetavecr = Eigen::VectorXd::Random(2);
    thetavecz.resize(2);
    thetavecz << 0.1, 0.1;

    mc.t.birth(1,0,50,thetavecl,thetavecr); // x tree
    mc.t.birth(2,0,25,thetavecl*2,thetavecr*1.3); // x tree
    mc.t.birth(4,1,30,thetavecz,thetavecz); // u tree
    mc.t.birth(9,1,80,thetavecz,thetavecz); // u tree

    mc.pr_vec();

    // Find the subtree roots
    tree::npv uroots;
    mc.t.getsubtreeroots(uroots, uvec);
    cout << "Subtree root = " << uroots[0] << endl; // should be pointer to nid 4

    // Check if different nodes are in a subtree
    tree::tree_p ni4, ni5, ni8, ni9; // true, false, true, true
    tree::tree_p subtree4,subtree5,subtree8,subtree9;
    ni4 = mc.t.l->l;
    ni5 = mc.t.l->r;
    ni8 = ni4->l;
    ni9 = ni4->r; 

    ni4->nodeinsubtree(uroots, subtree4);
    ni5->nodeinsubtree(uroots, subtree5);
    ni8->nodeinsubtree(uroots, subtree8);
    ni9->nodeinsubtree(uroots, subtree9);
    
    tree::tree_p ni3,ni18,ni19;
    ni3 = mc.t.r;
    ni19 = ni9->r;
    ni18 = ni9->l;

    cout << "Node 4 in subtree = " << subtree4 << endl;
    cout << "Node 5 in subtree = " << subtree5 << endl;
    cout << "Node 8 in subtree = " << subtree8 << endl;
    cout << "Node 9 in subtree = " << subtree9 << endl;

    // Get suff stats at each bottom node
    tree::npv bnv;
    std::vector<sinfo*> siv;
    mcinfo mil, mir,mil2, mir2, mil3, mir3, mit3;
    std::vector<mcinfo*> mcv;
    std::map<tree::tree_cp,size_t> bnmap;
    tree::tree_p nx;

    mc.t.getbots(bnv);
    mcv.resize(bnv.size());
    siv.resize(bnv.size());
    cout << "bnv size = " << bnv.size() << endl;
    for(size_t i=0;i<bnv.size();i++){
        bnmap[bnv[i]] = i;
        cout << "bnv[i] = " << bnv[i] << endl;
        mcv[i] = new mcinfo();
    }

    for(;diter<diter.until();diter++){
        nx = mc.t.bn(diter.getxp(),xi);
        mc.add_observation_to_suff(diter,(*mcv[bnmap[nx]]));
        /*
        if(bnmap[nx] == 1){
            cout << "sumzw = " << (*mcv[bnmap[nx]]).sumzw << endl;
            //cout << "gety = " << diter.gety() << endl;
        }
        */
    }
    // Get suff stats using local_getsuff for birth at node 4 with (1,30)
    // Uncomment and move the getsuff functions into public
    /*
    mc.getsuff(ni3,1,25,mil,mir);

    cout << "Split Node 3: ----" << endl;     
    mil.print();
    mir.print();

    cout << "Split Node 18: ----" << endl;
    mc.getsuff(ni18,0,10,mil2,mir2);
    mil2.print();
    mir2.print();

    cout << "Death at node 9: ----" << endl;
    mc.getsuff(ni18,ni19,mil3,mir3);
    mil3.print();
    mir3.print();

    cout << "--- Suff stats at node 9 after death: -----" << endl;
    mit3 += mir3;
    mit3 += mil3;
    mit3.print();
    
    // Testing lm...
    double lmstree, lmn;

    // Get subtre lm if node contains subtree info        
    lmstree = mc.lmsubtree(mir3);
    lmn = mc.lmsubtreenode(mir3);
    cout << "outlm = " << lmstree << endl;
    cout << "outlm = " << lmn << endl;
    */

    // test subsuff, used in rotate and perturb
    // Assume roation starts at node 4
    //tree::npv bnv_rot4;
    //std::vector<sinfo*> siv_rot4;
    //mc.subsuff(ni4,bnv_rot4,siv_rot4); // to see results, uncomment the print statements in the local_subsuff_subtree functions    

    cout << "Compare with original 5 bns ----" << endl;
    // Print each individual suff stat
    for(size_t i=0;i<bnv.size();i++){
        (*mcv[i]).print();
    }

    // Testing draw theta functions
    mc.drawthetavec(gen);
    mc.pr_vec();

}