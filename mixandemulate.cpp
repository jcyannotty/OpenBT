#include <chrono>
#include <iostream>
#include <string>
#include <ctime>
#include <sstream>

#include <fstream>
#include <vector>
#include <limits>

#include "Eigen/Dense"
#include <Eigen/StdVector>

#include "crn.h"
#include "tree.h"
#include "brt.h"
#include "brtfuns.h"
#include "dinfo.h"
#include "mbrt.h"
#include "ambrt.h"
#include "psbrt.h"
#include "tnorm.h"
#include "mxbrt.h"
#include "amxbrt.h"

using std::cout;
using std::endl;

#define MODEL_BT 1
#define MODEL_BINOMIAL 2
#define MODEL_POISSON 3
#define MODEL_BART 4
#define MODEL_HBART 5
#define MODEL_PROBIT 6
#define MODEL_MODIFIEDPROBIT 7
#define MODEL_MERCK_TRUNCATED 8
#define MODEL_MIXBART 9


int main(int argc, char* argv[])
{
   std::string folder("");

   if(argc>1)
   {
      std::string confopt("--conf");
      if(confopt.compare(argv[1])==0) {
#ifdef _OPENMPI
         return 101;
#else
         return 100;
#endif
      }

      //otherwise argument on the command line is path to conifg file.
      folder=std::string(argv[1]);
      folder=folder+"/";
   }


   //-----------------------------------------------------------
   //random number generation
   crn gen;
   gen.set_seed(static_cast<long long>(std::chrono::high_resolution_clock::now()
                                   .time_since_epoch()
                                   .count()));

   //--------------------------------------------------
   //process args
   std::ifstream conf(folder+"config");

   // model type
   int modeltype;
   conf >> modeltype;

   // core filenames for x,y,f input --- f input is for model mixing
   std::string xcore,ycore;
   conf >> xcore;
   conf >> ycore;
   
   //offset -- used in probit, but not in bart for instance.
   double off;
   conf >> off;


return 0;
}