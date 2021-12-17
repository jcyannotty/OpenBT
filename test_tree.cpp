//     test.cpp: BT tree class testing/validation code.
//     Copyright (C) 2012-2016 Matthew T. Pratola, Robert E. McCulloch and Hugh A. Chipman
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
//     Robert E. McCulloch: robert.e.mculloch@gmail.com
//     Hugh A. Chipman: hughchipman@gmail.com


#include <iostream>

#include "crn.h"
#include "tree.h"
#include "brtfuns.h"

#ifdef _OPENMPI
#   include <mpi.h>
#endif


using std::cout;
using std::endl;

int main()
{
   /*
   cout << "*****into test for tree\n";

   crn gen;

   //--------------------------------------------------
   //make a simple tree

   tree t;
   cout << "** print out a null tree\n";
   t.pr();
      
   t.birth(1,0,50,-1.0,1.0);
   cout << "** print out a tree with one split\n";
   t.pr();
   
   t.birth(2,0,25,-1.5,-0.5);
   cout << "** print out a tree with two splits\n";
   t.pr();


   tree t2;
   t2.birth(1,0,33,-3,3);
   cout << "** tree 2\n";
   t2.pr();

   tree::npv bots;
   tree st;

   st=t; //copy
   cout << "** supertree:\n";
   st.pr();

   cout << "** get bots:\n";
   st.getbots(bots);
   cout << "bots.size=" << bots.size() << endl;
   //collapse each tree j=1..m into the supertree
   for(size_t i=0;i<bots.size();i++) {
      cout << "iteration i=" << i << endl;
      collapsetree(st,bots[i],&t2); //mb[j]->t);
      st.pr();
   }
   bots.clear();

   cout << "** collapsed supertree:\n";
   st.pr();
   */

   //**************************************
   //Test out the model mixing tree functions
   //**************************************
   
   //Make simple null tree and print 
   tree tv1;
   cout << "Simple Tree with Vector Parameters:" << endl;
   tv1.pr_vec();

   //Assign left and right vectors for terminal nodes in birth step
   Eigen::VectorXd thetavecl, thetavecr;
   thetavecl = Eigen::VectorXd::Random(tv1.k);
   thetavecr = Eigen::VectorXd::Random(tv1.k);

   tv1.birth(1,0,50,thetavecl,thetavecr);
   cout << "** print out a tree with one split\n";
   tv1.pr_vec();

   return 0;
}
