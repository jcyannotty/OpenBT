#include <iostream>
#include <fstream>

#include "crn.h"
#include "brt.h"
#include "brtfuns.h"
#include "dinfo.h"
#include "mxbrt.h"

//Include Eigen library
#include "Eigen/Dense"

int main(){
    //-------------------------------------------------------
    //---Read in Data for mxbrt examples
    //-------------------------------------------------------
    crn gen;
    gen.set_seed(199);

    int tc=4; //thread count for OpenMP

    //--------------------------------------------------
    //read in y
    std::vector<double> y;
    double ytemp;

    std::ifstream yf("y.txt");
    while(yf >> ytemp)
        y.push_back(ytemp);
    size_t n = y.size();
    cout << "n from y.txt: " << n <<endl;

    //--------------------------------------------------
    //read in x
    std::vector<double> x;
    double xtemp;
    size_t p;
    p=1;

    std::ifstream xf("x.txt");
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

    std::ifstream ff("f.txt");
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
    di.n=n;di.p=p,di.x = &x[1];di.tc=tc;
    di.y = &y[0];

    diterator diter(&di);

    //--------------------------------------------------
    //make xinfo
    xinfo xi;
    size_t nc=50;
    makexinfo(p,n,&x[0],xi,nc); //use the 1st column with x[0]

    //prxi(xi);
    //--------------------------------------------------
    //make finfo -- need to read in and store f formally, just using same x from above for now
    finfo fi;
    makefinfo(k, n, &f[0], fi);
    cout << fi << endl;

    //--------------------------------------------------
    // read in the initial change of variable rank correlation matrix
    std::vector<std::vector<double> > chgv;
    std::vector<double> cvvtemp;
    double cvtemp;
    std::ifstream chgvf("chgv.txt");
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


    //-------------------------------------------------------
    //Example 1 -- Test mxsinfo
    //-------------------------------------------------------
    //Initialize matrix and vector
    Eigen::MatrixXd F;
    Eigen::VectorXd v;
    double yy = 10.2;

    F = Eigen::MatrixXd::Random(k,k);
    v = Eigen::VectorXd::Random(k);

    //Work with sinfo object
    sinfo s;
    std::cout << s.n << std::endl; //Prints out 0 as expected
    s.n = 10; //Change the sample size to 10

    //Try to create mxinfo objects
    mxsinfo mx1; //constructor 1
    mxsinfo mx2(s, k, F, v, yy); //Constructor 2
    mxsinfo mx3(mx2); //Constructor 3

    //See what they print
    mx1.print_mx();
    mx2.print_mx();
    mx3.print_mx();

    //Work with operators
    std::cout << "---Compound Addition Operator" << std::endl;
    mx3 += mx2;
    mx3.print_mx();

    std::cout << "---Addition and Equality Operator" << std::endl;    
    mxsinfo mx4 = mx2 + mx3; //Add two mxsinfo objects 
    mx4.print_mx();

    //-------------------------------------------------------
    //Example 2 -- Create an mxbrt object
    //-------------------------------------------------------
    cout << "\n\n-----------------------------------------" << endl;
    cout << "Example 2: Work with a mxbrt object \n" << endl;
    
    //Initialize prior parameters
    double *sig = new double[di.n];
    double tau=0.1;
    double beta0 = 0.8;
    for(size_t i=0;i<di.n;i++) sig[i]=0.1;

    //First mix bart object with basic constructor
    mxbrt mxb; 
    cout << "****Initial Object" << endl; 
    mxb.pr_vec();
    mxb.setxi(&xi);    //set the cutpoints for this model object
    //function output 
    mxb.setfi(&fi,k); //set function output for this model object
    //data objects
    mxb.setdata_mix(&di);  //set the data for model mixing
    //thread count
    mxb.settc(tc);      //set the number of threads when using OpenMP, etc.
    //tree prior
    mxb.settp(0.95, //the alpha parameter in the tree depth penalty prior
            1.0     //the beta parameter in the tree depth penalty prior
            );
    //MCMC info
    mxb.setmi(
            0.5,  //probability of birth/death
            0.5,  //probability of birth
            5,    //minimum number of observations in a bottom node
            true, //do perturb/change variable proposal?
            0.01,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
            0.01,  //probability of doing a change of variable proposal.  perturb prob=1-this.
            &chgv  //initialize the change of variable correlation matrix.
            );
    mxb.setci(tau,beta0,sig);

    cout << "\n*****After init:\n";
    mxb.pr_vec();

    cout << "-----------------------------------" << endl;
    cout << "Test Individual Functions involved in draw: \n\n" << endl; 
    cout << "mxbrt lm = " << mxb.lm(mx3) << endl;
    cout << "mxbrt drawnodethetavec: " << mxb.drawnodethetavec(mx3, gen) << endl;
    cout << "mxbrt birth/death: \n" << endl;
    mxb.bd_vec(gen);
    mxb.pr_vec();

    cout << "-----------------------------------" << endl;
    
       
    cout << "\n*****After 1 draw:\n";
    mxb.drawvec(gen);
    mxb.pr_vec();
    

    /*
    cout << "\n-----------------------------------" << endl;
    size_t nd = 5000;
    size_t nadapt=1000;
    size_t adaptevery=100;
    size_t nburn=200;
    std::vector<double> fitted(n);

    for(size_t i=0;i<nadapt;i++) { mxb.draw(gen); if((i+1)%adaptevery==0) mxb.adapt(); }
    for(size_t i=0;i<nburn;i++) mxb.draw(gen); 
    
    cout << "\n*****After "<< nd << " draws:\n";
    cout << "Collecting statistics" << endl;
    mxb.setstats(true);
    for(int i = 0; i<nd; i++){
        //cout << "*****Draw "<< i << endl;
        mxb.drawvec(gen);
        
        if((i % 2500) ==0){
            cout << "***Draw " << i << "\n" << endl;
            //mxb.pr_vec();
        } 
        
        for(size_t j=0;j<n;j++) fitted[j]+=mxb.f(j)/nd;
        //mxb.pr_vec();
    }    

    // summary statistics
    unsigned int varcount[p];
    unsigned int totvarcount=0;
    for(size_t i=0;i<p;i++) varcount[i]=0;
    unsigned int tmaxd=0;
    unsigned int tmind=0;
    double tavgd=0.0;

    mxb.getstats(&varcount[0],&tavgd,&tmaxd,&tmind);
    for(size_t i=0;i<p;i++) totvarcount+=varcount[i];
    tavgd/=(double)(nd);

    cout << "Average tree depth: " << tavgd << endl;
    cout << "Maximum tree depth: " << tmaxd << endl;
    cout << "Minimum tree depth: " << tmind << endl;
    cout << "Variable perctg:    ";
    for(size_t i=0;i<p;i++) cout << "  " << i+1 << "  ";
    cout << endl;
    cout << "                    ";
    for(size_t i=0;i<p;i++) cout << " " << ((double)varcount[i])/((double)totvarcount)*100.0 << " ";
    cout << endl;

    cout << "Print Fitted Values" << endl;
    for(int i = 0; i<n; i++){
        cout << "Y = " << y[i] <<" -- Fitted " << fitted[i] << endl;
    }
    */
    return 0;

}
