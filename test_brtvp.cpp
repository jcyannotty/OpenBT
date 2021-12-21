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
    p=2;

    std::ifstream xf("f.txt");
    while(xf >> xtemp){
        x.push_back(xtemp);
    }
    cout << x.size() << endl;
    n = x.size()/p;
    if(x.size() != n*p) {
        cout << "error: input x file has wrong number of values\n";
        return 1;
    }
    cout << "n,p: " << n << ", " << p << endl;
    
    //--------------------------------------------------
    //Make dinfo -- (want to test vector parameter functions here. Test model mixing later with dinfo_mx)
    dinfo di;
    di.n=n;di.p=p,di.x = &x[1];di.tc=tc;

    diterator diter(&di);
    cout << "Output of first diter: \n" << "X = " << diter.getx() << " i = " << diter.geti() <<  " *diter = "  << *diter << endl;

    for(;diter<diter.until();diter++){
        cout << diter.getx() << "-------" << *diter << endl;
    }

    //--------------------------------------------------
    //make xinfo
    xinfo xi;
    size_t nc=100;
    makexinfo(p,n,&x[0],xi,nc); //use the 1st column with x[0]

    //prxi(xi);
    //--------------------------------------------------
    //make finfo -- need to read in and store f formally, just using same x from above for now
    finfo fi;
    size_t k;
    k = 2;
    //n = x.size()/k;
    makefinfo(k, n, &x[0], fi);

    cout << fi << endl;

    //--------------------------------------------------
    //make a tree and print out results from bottom nodes function
    tree t;
    t.birth(1, 0, 10, 1.0, 2.0);
    //t.pr();

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