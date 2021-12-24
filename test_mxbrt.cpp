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
    //Initialize matrix and vector
    Eigen::MatrixXd F;
    Eigen::VectorXd v;
    int k = 4;
    double yy = 10.2;

    F = Eigen::MatrixXd::Random(k,k);
    v = Eigen::VectorXd::Random(k);

    //Work with sinfo object
    sinfo s;
    std::cout << s.n << std::endl; //Prints out 0 as expected

    //Try to create mxinfo objects
    mxsinfo mx1; //constructor 1
    mxsinfo mx2(s, k, F, v, yy); //Constructor 2
    mxsinfo mx3(mx2); //Constructor 3

    //See what they print
    mx2.print_mx();
    mx3.print_mx();

    //Work with operators
    std::cout << "Compound Addition Operator" << std::endl;
    mx2 += mx3;
    mx2.print_mx();

    std::cout << "Addition Operator" << std::endl;    
    //Try the other addition operator  
    mxsinfo mx4 = mx2.operator+(mx3); //this can be used to create a new instance of mxinfo 
    mx3.print_mx();

    return 0;

}
