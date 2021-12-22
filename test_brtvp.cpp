//     test_brtvp.cpp: Base BT model class with vector parameters test/validation code.

#include <iostream>
#include <fstream>

#include "crn.h"
#include "brt.h"
#include "brtfuns.h"
#include "dinfo.h"

#ifdef _OPENMPI
#   include <mpi.h>
#endif


using std::cout;
using std::endl;

int main(){
    cout << "\n*****into test for brt\n";
    cout << "\n\n";

    crn gen;
    gen.set_seed(199);

    int tc=4; //thread count for OpenMP

    //--------------------------------------------------
    //read in x
    std::vector<double> x;
    double xtemp;
    size_t n,p;
    p=1;

    std::ifstream xf("x.txt");
    while(xf >> xtemp){
        x.push_back(xtemp);
    }
    
    n = x.size()/p;
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

    diterator diter(&di);
    /*
    cout << "Output of first diter: \n" << "X = " << diter.getx() << " i = " << diter.geti() <<  " *diter = "  << *diter << endl;

    for(;diter<diter.until();diter++){
        cout << diter.getx() << "-------" << *diter << endl;
    }
    */

    //--------------------------------------------------
    //make xinfo
    xinfo xi;
    size_t nc=100;
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

    //--------------------------------------------------
    //brt Example 1:
    cout << "\n******************************************" << endl;
    cout << "\n Make a brt object and print it out\n";
    brt bm;
    cout << "\nbefore init:\n";
    bm.pr_vec();
    //cutpoints
    bm.setxi(&xi);    //set the cutpoints for this model object
    //set fi
    bm.setfi(&fi,k); //set the function values 
    //data objects
    bm.setdata_mix(&di);  //set the data...not sure if we need to used setdata_mix() since this should only be used at the start of mcmc
    //thread count
    bm.settc(tc);      //set the number of threads when using OpenMP, etc.
    //tree prior
    bm.settp(0.95, //the alpha parameter in the tree depth penalty prior
            1.0     //the beta parameter in the tree depth penalty prior
            );

    //MCMC info
    bm.setmi(0.7,  //probability of birth/death
         0.5,  //probability of birth
         5,    //minimum number of observations in a bottom node
         true, //do perturb/change variable proposal?
         0.2,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
         0.2,  //probability of doing a change of variable proposal.  perturb prob=1-this.
         &chgv  //initialize the change of variable correlation matrix.
         );
    cout << "\nafter init:\n";
    bm.pr_vec();
    

    //--------------------------------------------------
    //brt Example 2:
    cout << "\n******************************************" << endl;
    cout << "\nTry some draws of brt and print it out\n";
    cout << "\n1 draw:\n";
    bm.drawvec(gen);
    bm.pr_vec();
    
    size_t nd=1000;
    cout << "\n" << nd << " draws:\n";
    for(size_t i=0;i<nd;i++){
        bm.drawvec(gen);
        //bm.pr_vec();
    } 
    bm.pr_vec();

    //--------------------------------------------------
    //Example 3: Test the setf_mix & setr_mix and compare to setf & setr
    cout << "\n******************************************" << endl;
    cout << "Before setf_mix ... " << bm.f(2) << endl;
    bm.setf_mix();
    cout << "After setf_mix ... " << bm.f(2) << endl;

    cout << "Before setr_mix ... " << bm.r(2) << endl;
    bm.setr_mix();
    cout << "After setr_mix ... " << bm.r(2) << endl;
    
    //--------------------------------------------------
    //Example 4: Work with sinfo
    cout << "\n******************************************" << endl;
    std::vector<sinfo> siv(2);
    std::cout << "testing vectors of sinfos\n";
    std::cout << siv[0].n << ", " << siv[1].n << std::endl;

    siv.clear();
    siv.resize(2);
    std::cout << siv[0].n << ", " << siv[1].n << std::endl;

    //--------------------------------------------------
    //Example 5: Work towards constructing a full mcmc
    cout << "\n******************************************" << endl;
    size_t tuneevery=250;
    size_t tune=5000;
    size_t burn=5000;
    size_t draws=5000;
    brt b;

    b.setxi(&xi);
    b.setfi(&fi, k);
    b.setdata_mix(&di);
    b.settc(tc);
    //Setmi -- pbd,pb,minperbot,dopert,pertalpha,pchgv,chgv
    b.setmi(0.8,0.5,5,true,0.1,0.2,&chgv);

    // tune the sampler   
    for(size_t i=0;i<tune;i++){
        b.drawvec(gen);
        if((i+1)%tuneevery==0){
            b.adapt();
        }
    }

    b.t.pr_vec();
    // run some burn-in, tuning is fixed now
    for(size_t i=0;i<burn;i++){
        b.drawvec(gen);
    }

    // draw from the posterior
    // After burn-in, turn on statistics if we want them:
    cout << "Collecting statistics" << endl;
    b.setstats(true);
    // then do the draws
    for(size_t i=0;i<draws;i++){
        b.drawvec(gen);
    }

    // summary statistics
    unsigned int varcount[p];
    unsigned int totvarcount=0;
    for(size_t i=0;i<p;i++) varcount[i]=0;
    unsigned int tmaxd=0;
    unsigned int tmind=0;
    double tavgd=0.0;

    b.getstats(&varcount[0],&tavgd,&tmaxd,&tmind);
    for(size_t i=0;i<p;i++) totvarcount+=varcount[i];
    tavgd/=(double)(draws);

    cout << "Average tree depth: " << tavgd << endl;
    cout << "Maximum tree depth: " << tmaxd << endl;
    cout << "Minimum tree depth: " << tmind << endl;
    cout << "Variable perctg:    ";
    for(size_t i=0;i<p;i++) cout << "  " << i+1 << "  ";
    cout << endl;
    cout << "                    ";
    for(size_t i=0;i<p;i++) cout << " " << ((double)varcount[i])/((double)totvarcount)*100.0 << " ";
    cout << endl;
    
    //--------------------------------------------------
    //--------------------------------------------------
    //Extra 
    //cout << x[44] << endl;

    /*
    //Important Eigen storage example. Shows ways to work with eigen pointers/references
    Eigen::MatrixXd A(2,2);
    Eigen::MatrixXd *B;
    A << 2,4,1,3;
    B = &A;
    cout << "print A:" << endl;
    cout << A << endl;
    cout << &A << endl; //Address of the matrix
    cout << A.data() << endl; //Address of the first element
    cout << *(A.data() + 1) << endl; //Dereference the 2nd row 1st column element -- prints 1
    cout << *(A.data() + 2) << endl; //Dereference the 1st row 2nd column element -- prints 4
    cout << *(A.data() + 3) << endl; //Dereference the 2nd row 2nd column element -- prints 3

    cout << "print B:" << endl;
    cout << *B << endl; //Prints a dereferenced B -- same output as A
    cout << ((*B).data()) << endl; //Prints the same as A.data()
    cout << (*B).row(1) << endl; //Print the 2nd row of dereferenced B.
    */

   /*
   //Two ways to convert a vector to a matrix -- used to make finfo
   Eigen::VectorXd v(6);
   v << 10,20,30,40, 50, 60;
   Eigen::Map<mxd, Eigen::RowMajor> M(v.data(), 2,3);
   cout << M.transpose() << endl;
   
   mxd M2 = Eigen::Map<Eigen::Matrix<double, 3,2, Eigen::RowMajor>>(v.data());
   cout << M2 << endl;
   */


}