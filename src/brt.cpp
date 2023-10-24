//     brt.cpp: Base BT model class methods.
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


#include "brt.h"
#include "brtfuns.h"
#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <vector>

using std::cout;
using std::endl;

//--------------------------------------------------
//a single iteration of the MCMC for brt model
void brt::draw(rn& gen)
{
   // Structural/topological proposal(s)
   if(gen.uniform()<mi.pbd)
//   if(mi.pbd>0.0)
      bd(gen);
   else
   {
      tree::tree_p tnew;
      tnew=new tree(t); //copy of current to make life easier upon rejection
      rot(tnew,t,gen);
      delete tnew;
   }
   
   // Perturbation Proposal
   if(mi.dopert)
      pertcv(gen);

   // Gibbs Step
    drawtheta(gen);

   //update statistics
   if(mi.dostats) {
      tree::npv bnv; //all the bottom nodes
      for(size_t k=0;k< xi->size();k++) mi.varcount[k]+=t.nuse(k);
      t.getbots(bnv);
      unsigned int tempdepth[bnv.size()];
      unsigned int tempavgdepth=0;
      for(size_t i=0;i!=bnv.size();i++)
         tempdepth[i]=(unsigned int)bnv[i]->depth();
      for(size_t i=0;i!=bnv.size();i++) {
         tempavgdepth+=tempdepth[i];
         mi.tmaxd=std::max(mi.tmaxd,tempdepth[i]);
         mi.tmind=std::min(mi.tmind,tempdepth[i]);
      }
      mi.tavgd+=((double)tempavgdepth)/((double)bnv.size());
   }
}
//--------------------------------------------------
//slave controller for draw when using MPI
void brt::draw_mpislave(rn& gen)
{
   #ifdef _OPENMPI
   char buffer[SIZE_UINT3];
   int position=0;
   MPI_Status status;
   typedef tree::npv::size_type bvsz;

   // Structural/topological proposal(s)
   // MPI receive the topological proposal type and nlid and nrid if applicable.
   MPI_Recv(buffer,SIZE_UINT3,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
   sinfo& tsil = *newsinfo();
   sinfo& tsir = *newsinfo();
   if(status.MPI_TAG==MPI_TAG_BD_BIRTH_VC) {
      unsigned int nxid,v,c;
      tree::tree_p nx;
      MPI_Unpack(buffer,SIZE_UINT3,&position,&nxid,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      MPI_Unpack(buffer,SIZE_UINT3,&position,&v,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      MPI_Unpack(buffer,SIZE_UINT3,&position,&c,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      nx=t.getptr((size_t)nxid);
      getsuff(nx,(size_t)v,(size_t)c,tsil,tsir);
      MPI_Status status2;
      MPI_Recv(buffer,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status2);
      if(status2.MPI_TAG==MPI_TAG_BD_BIRTH_VC_ACCEPT) t.birthp(nx,(size_t)v,(size_t)c,0.0,0.0); //accept birth
      //else reject, for which we do nothing.
   }
   else if(status.MPI_TAG==MPI_TAG_BD_DEATH_LR) {
      unsigned int nlid,nrid;
      tree::tree_p nl,nr;
      MPI_Unpack(buffer,SIZE_UINT3,&position,&nlid,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      MPI_Unpack(buffer,SIZE_UINT3,&position,&nrid,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      nl=t.getptr((size_t)nlid);
      nr=t.getptr((size_t)nrid);
      getsuff(nl,nr,tsil,tsir);
      MPI_Status status2;
      MPI_Recv(buffer,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status2);
      if(status2.MPI_TAG==MPI_TAG_BD_DEATH_LR_ACCEPT) t.deathp(nl->getp(),0.0); //accept death
      //else reject, for which we do nothing.
   }
   else if(status.MPI_TAG==MPI_TAG_ROTATE) {
      mpi_resetrn(gen);
      tree::tree_p tnew;
      tnew=new tree(t); //copy of current to make life easier upon rejection
      rot(tnew,t,gen);
      delete tnew;
   }
   delete &tsil;
   delete &tsir;

   // Perturbation Proposal
   // nothing to perturb if tree is a single terminal node, so we would just skip.
   if(mi.dopert && t.treesize()>1)
   {
      tree::npv intnodes;
      tree::tree_p pertnode;
      t.getintnodes(intnodes);
      for(size_t pertdx=0;pertdx<intnodes.size();pertdx++)
      {
         std::vector<sinfo*>& sivold = newsinfovec();
         std::vector<sinfo*>& sivnew = newsinfovec();
         pertnode = intnodes[pertdx];
         MPI_Recv(buffer,SIZE_UINT3,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
         if(status.MPI_TAG==MPI_TAG_PERTCV)
         {
            size_t oldc = pertnode->getc();
            unsigned int propcint;
            position=0;
            MPI_Unpack(buffer,SIZE_UINT1,&position,&propcint,1,MPI_UNSIGNED,MPI_COMM_WORLD);
            size_t propc=(size_t)propcint;
            pertnode->setc(propc);
            tree::npv bnv;
            getpertsuff(pertnode,bnv,oldc,sivold,sivnew);
            MPI_Status status2;
            MPI_Recv(buffer,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status2);
            if(status2.MPI_TAG==MPI_TAG_PERTCV_ACCEPT) pertnode->setc(propc); //accept new cutpoint
            //else reject, for which we do nothing.
         }
         else if(status.MPI_TAG==MPI_TAG_PERTCHGV)
         {
            size_t oldc = pertnode->getc();
            size_t oldv = pertnode->getv();
            bool didswap=false;
            unsigned int propcint;
            unsigned int propvint;
            position=0;
            mpi_update_norm_cormat(rank,tc,pertnode,*xi,(*mi.corv)[oldv],chv_lwr,chv_upr);
            MPI_Recv(buffer,SIZE_UINT3,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            MPI_Unpack(buffer,SIZE_UINT3,&position,&propcint,1,MPI_UNSIGNED,MPI_COMM_WORLD);
            MPI_Unpack(buffer,SIZE_UINT3,&position,&propvint,1,MPI_UNSIGNED,MPI_COMM_WORLD);
            MPI_Unpack(buffer,SIZE_UINT3,&position,&didswap,1,MPI_CXX_BOOL,MPI_COMM_WORLD);
            size_t propc=(size_t)propcint;
            size_t propv=(size_t)propvint;
            pertnode->setc(propc);
            pertnode->setv(propv);
            if(didswap)
               pertnode->swaplr();
            mpi_update_norm_cormat(rank,tc,pertnode,*xi,(*mi.corv)[propv],chv_lwr,chv_upr);
            tree::npv bnv;
            getchgvsuff(pertnode,bnv,oldc,oldv,didswap,sivold,sivnew);
            MPI_Status status2;
            MPI_Recv(buffer,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status2);
            if(status2.MPI_TAG==MPI_TAG_PERTCHGV_ACCEPT) { //accept change var and pert
               pertnode->setc(propc);
               pertnode->setv(propv);
               if(didswap)
                  pertnode->swaplr();
            }
            // else reject, for which we do nothing.
         }
         // no other possibilities.
         for(bvsz j=0;j<sivold.size();j++) delete sivold[j];
         for(bvsz j=0;j<sivnew.size();j++) delete sivnew[j];
         delete &sivold;
         delete &sivnew;
      }
   }

   // Gibbs Step
   drawtheta(gen);

   #endif
}
//--------------------------------------------------
//adapt the proposal widths for perturb proposals,
//bd or rot proposals and b or d proposals.
void brt::adapt()
{
//   double pert_rate,b_rate,d_rate,bd_rate,rot_rate,m_rate,chgv_rate;
   double pert_rate,m_rate,chgv_rate;

   pert_rate=((double)mi.pertaccept)/((double)mi.pertproposal);
   chgv_rate=((double)mi.chgvaccept)/((double)mi.chgvproposal);
//   pert_rate=((double)(mi.pertaccept+mi.baccept+mi.daccept+mi.rotaccept))/((double)(mi.pertproposal+mi.dproposal+mi.bproposal+mi.rotproposal));
//   b_rate=((double)mi.baccept)/((double)mi.bproposal);
//   d_rate=((double)mi.daccept)/((double)mi.dproposal);
//   bd_rate=((double)(mi.baccept+mi.daccept))/((double)(mi.dproposal+mi.bproposal));
//   rot_rate=((double)mi.rotaccept)/((double)mi.rotproposal);
   m_rate=((double)(mi.baccept+mi.daccept+mi.rotaccept))/((double)(mi.dproposal+mi.bproposal+mi.rotproposal));

   //update pbd
   // a mixture between calibrating to m_rate (25%) and not moving too quickly away from
   // the existing probability of birth/death (75%):
//   mi.pbd=0.25*mi.pbd*m_rate/0.24+0.75*mi.pbd;
   // avoid too small or large by truncating to 0.1,0.9 range:
//   mi.pbd=std::max(std::min(0.9,mi.pbd),0.1);

   //update pb
//old   mi.pb=mi.pb*bd_rate/0.24;
//old   mi.pb=mi.pb*(b_rate+d_rate)/2.0/bd_rate;
   // a mixture between calibrating to the (bd_rate and m_rate) and existing probability of birth
   // in other words, don't move too quickly away from existing probability of birth
   // and when we do move, generally we favor targetting bd_rate (90%) but also target m_rate to
   // a small degree (10%):
//   mi.pb=0.25*(0.9*mi.pb*(b_rate+d_rate)/2.0/bd_rate + 0.1*mi.pb*m_rate/0.24)+0.75*mi.pb;
   // avoid too small or large by truncating to 0.1,0.9 range:
//   mi.pb=std::max(std::min(0.9,mi.pb),0.1);

   //update pertalpha
   mi.pertalpha=mi.pertalpha*pert_rate/0.44;
//   if(mi.pertalpha>2.0) mi.pertalpha=2.0;
//   if(mi.pertalpha>(1.0-1.0/ncp1)) mi.pertalpha=(1.0-1.0/ncp1);
   if(mi.pertalpha>2.0) mi.pertalpha=2.0;
   if(mi.pertalpha<(1.0/ncp1)) mi.pertalpha=(1.0/ncp1);

   mi.pertaccept=0; mi.baccept=0; mi.rotaccept=0; mi.daccept=0;
   mi.pertproposal=1; mi.bproposal=1; mi.rotproposal=1; mi.dproposal=1;
   //if(mi.dostats) {

#ifdef SILENT
   //Ugly hack to get rid of silly compiler warning
   if(m_rate) ;
   if(chgv_rate) ;
#endif

#ifndef SILENT
   cout << "pert_rate=" << pert_rate << " pertalpha=" << mi.pertalpha << " chgv_rate=" << chgv_rate;
   // cout << "   b_rate=" << b_rate << endl;
   // cout << "   d_rate=" << d_rate << endl;
   // cout << "   bd_rate=" << bd_rate << endl;
   // cout << " rot_rate=" << rot_rate << endl;
   cout << "   m_rate=" << m_rate;
   //   cout << "mi.pbd=" << mi.pbd << "  mi.pb=" << mi.pb<< "  mi.pertalpha=" << mi.pertalpha << endl;
   //   cout << endl;
#endif
   //}
}
//--------------------------------------------------
//draw all the bottom node theta's for the brt model
void brt::drawtheta(rn& gen)
{
   tree::npv bnv;
//   std::vector<sinfo> siv;
   std::vector<sinfo*>& siv = newsinfovec();

   allsuff(bnv,siv);
#ifdef _OPENMPI
   mpi_resetrn(gen);
#endif
   
   for(size_t i=0;i<bnv.size();i++) {
      bnv[i]->settheta(drawnodetheta(*(siv[i]),gen));
      delete siv[i]; //set it, then forget it!
   }
   delete &siv;  //and then delete the vector of pointers.
}
//--------------------------------------------------
//draw theta for a single bottom node for the brt model
double brt::drawnodetheta(sinfo& si, rn& gen)
{
//   return 1.0;
   return si.n;
}
//--------------------------------------------------
//pr for brt
void brt::pr()
{
   std::cout << "***** brt object:\n";
#ifdef _OPENMPI
   std::cout << "mpirank=" << rank << endl;
#endif
   if(xi) {
      size_t p = xi->size();
      cout  << "**xi cutpoints set:\n";
      cout << "\tnum x vars: " << p << endl;
      cout << "\tfirst x cuts, first and last " << (*xi)[0][0] << ", ... ," << 
              (*xi)[0][(*xi)[0].size()-1] << endl;
      cout << "\tlast x cuts, first and last " << (*xi)[p-1][0] << ", ... ," << 
              (*xi)[p-1][(*xi)[p-1].size()-1] << endl;
   } else {
      cout << "**xi cutpoints not set\n";
   }
   if(di) {
      cout << "**data set, n,p: " << di->n << ", " << di->p << endl;
   } else {
      cout << "**data not set\n";
   }
   std::cout << "**the tree:\n";
   t.pr();
}
//--------------------------------------------------
//lm: log of integrated likelihood, depends on prior and suff stats
double brt::lm(sinfo& si)
{
   return 0.0;  //just drawing from prior for now.
}
//--------------------------------------------------
//getsuff used for birth.
void brt::local_getsuff(diterator& diter, tree::tree_p nx, size_t v, size_t c, sinfo& sil, sinfo& sir)    
{
   double *xx;//current x
   sil.n=0; sir.n=0;
   // Soft BART Version ----
   //cout << "nx = " << nx << " --- rank -- " << rank << endl;
   //cout << "nx nid = " << nx->nid() << " --- rank -- " << rank << endl;
   if(randpath){
      // Random Path BART update
      for(;diter<diter.until();diter++){
         //cout << "randz[*diter] nid = " << randz[*diter]->nid() << " --- rank " << rank << endl;
         if(nx==randz[*diter]){ //does the bottom node = xx's bottom node
            if(randz_bdp[*diter] == 1) {
               // Left move
               add_observation_to_suff(diter,sil);
            }else if(randz_bdp[*diter] == 2){
               // Right move (using else if rather than else as a precaution)
               add_observation_to_suff(diter,sir);
            }
         }
      }
   }else{
   // Deterministic BART Version ----
      for(;diter<diter.until();diter++)
      {
         xx = diter.getxp();
         if(nx==t.bn(diter.getxp(),*xi)) { //does the bottom node = xx's bottom node
            if(xx[v] < (*xi)[v][c]) {
                  //sil.n +=1;
                  add_observation_to_suff(diter,sil);
            } else {
                  //sir.n +=1;
                  add_observation_to_suff(diter,sir);
            }
         }
      }
   }

}

//--------------------------------------------------
//getsuff used for death
void brt::local_getsuff(diterator& diter, tree::tree_p l, tree::tree_p r, sinfo& sil, sinfo& sir)
{
   tree::tree_cp bn;
   sil.n=0; sir.n=0;

   for(;diter<diter.until();diter++)
   {
      // Soft vs deterministic BART
      if(randpath){
         bn = randz[*diter];
      }else{
         // Regular Deterministic step
         bn = t.bn(diter.getxp(),*xi);   
      }
      
      // Suff stat step
      if(bn==l) {
         //sil.n +=1;
         add_observation_to_suff(diter,sil);
      }
      if(bn==r) {
         //sir.n +=1;
         add_observation_to_suff(diter,sir);
      }
   }
}
//--------------------------------------------------
//Add in an observation, this has to be changed for every model.
//Note that this may well depend on information in brt with our leading example
//being double *sigma in cinfo for the case of e~N(0,sigma_i^2).
// Note that we are using the training data and the brt object knows the training data
//     so all we need to specify is the row of the data (argument size_t i).
void brt::add_observation_to_suff(diterator& diter, sinfo& si)
{
   si.n+=1; //in add_observation_to_suff
}
//--------------------------------------------------
//getsuff wrapper used for birth.  Calls serial or parallel code depending on how
//the code is compiled.
void brt::getsuff(tree::tree_p nx, size_t v, size_t c, sinfo& sil, sinfo& sir)
{
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_ompgetsuff(nx,v,c,*di,sil,sir); //faster if pass dinfo by value.
   #elif _OPENMPI 
      local_mpigetsuff(nx,v,c,*di,sil,sir);
   #else
      diterator diter(di);
      local_getsuff(diter,nx,v,c,sil,sir);
   #endif
}
//--------------------------------------------------
//allsuff (1)
void brt::allsuff(tree::npv& bnv,std::vector<sinfo*>& siv)
{
   //get bots once and pass them around
   bnv.clear();
   t.getbots(bnv);

   #ifdef _OPENMP
      typedef tree::npv::size_type bvsz;
      siv.clear(); //need to setup space threads will add into
      siv.resize(bnv.size());
      for(bvsz i=0;i!=bnv.size();i++) siv[i]=newsinfo();
#     pragma omp parallel num_threads(tc)
      local_ompallsuff(*di,bnv,siv); //faster if pass di and bnv by value.
   #elif _OPENMPI
      diterator diter(di);
      local_mpiallsuff(diter,bnv,siv);
   #else
      diterator diter(di);
      local_allsuff(diter,bnv,siv); //will resize siv
   #endif
}
//--------------------------------------------------
//local_subsuff
void brt::local_subsuff(diterator& diter, tree::tree_p nx, tree::npv& path, tree::npv& bnv, std::vector<sinfo*>& siv)
{
   tree::tree_cp tbn; //the pointer to the bottom node for the current observation
   size_t ni;         //the  index into vector of the current bottom node
   size_t index;      //the index into the path vector.
   double *x;
   tree::tree_p root=path[path.size()-1];

   typedef tree::npv::size_type bvsz;
   bvsz nb = bnv.size();
   siv.clear();
   siv.resize(nb);

   std::map<tree::tree_cp,size_t> bnmap;
   for(bvsz i=0;i!=bnv.size();i++) { bnmap[bnv[i]]=i; siv[i]=newsinfo(); }

   if(!randpath){
      // Determinisitc version
      for(;diter<diter.until();diter++) {
         index=path.size()-1;
         x=diter.getxp();
         if(root->xonpath(path,index,x,*xi)) { //x is on the subtree, 
            tbn = nx->bn(x,*xi);              //so get the right bn below interior node n.
            ni = bnmap[tbn];
            //siv[ni].n +=1;
            add_observation_to_suff(diter, *(siv[ni]));
         }
         //else this x doesn't map to the subtree so it's not added into suff stats.
      }
   }else{
      // Random path version
      for(;diter<diter.until();diter++) {
         // If randz pointer is in the bnmap, then add some suff stats
         if(bnmap.find(randz[*diter]) != bnmap.end()) {
            tbn = randz[*diter];          
            ni = bnmap[tbn];
            add_observation_to_suff(diter, *(siv[ni]));
         }
         //else this x doesn't map to the subtree so it's not added into suff stats.
      }
   }
}
//-------------------------------------------------- 
//local_ompsubsuff
void brt::local_ompsubsuff(dinfo di, tree::tree_p nx, tree::npv& path, tree::npv bnv,std::vector<sinfo*>& siv)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = di.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   std::vector<sinfo*>& tsiv = newsinfovec(); //will be sized in local_subsuff
   diterator diter(&di,beg,end);
   local_subsuff(diter,nx,path,bnv,tsiv);

#  pragma omp critical
   {
      for(size_t i=0;i<siv.size();i++) *(siv[i]) += *(tsiv[i]);
   }

   for(size_t i=0;i<tsiv.size();i++) delete tsiv[i];
   delete &tsiv;
#endif
}
//--------------------------------------------------
//local_mpisubsuff
void brt::local_mpisubsuff(diterator& diter, tree::tree_p nx, tree::npv& path, tree::npv& bnv, std::vector<sinfo*>& siv)
{
#ifdef _OPENMPI
   if(rank==0) {
      siv.clear(); //need to setup space threads will add into
      siv.resize(bnv.size());
      typedef tree::npv::size_type bvsz;
      for(bvsz i=0;i!=bnv.size();i++) siv[i]=newsinfo();
 
      // reduce all the sinfo's across the nodes, which is model-specific.
      local_mpi_reduce_allsuff(siv);
   }
   else
   {
      local_subsuff(diter,nx,path,bnv,siv);

      // reduce all the sinfo's across the nodes, which is model-specific.
      local_mpi_reduce_allsuff(siv);
   }
#endif
}

//--------------------------------------------------
//get suff stats for bots that are only below node n.
//NOTE!  subsuff is the only method for computing suff stats that does not
//       assume the root of the tree you're interested is brt.t.  Instead,
//       it takes the root of the tree to be the last entry in the path
//       vector.  In other words, for MCMC proposals that physically
//       construct a new proposed tree, t', suff stats must be computed
//       on t' using subsuff.  Using getsuff or allsuff is WRONG and will
//       result in undefined behaviour since getsuff/allsuff *assume* the 
//       the root of the tree is brt.t.
void brt::subsuff(tree::tree_p nx, tree::npv& bnv, std::vector<sinfo*>& siv)
{
   tree::npv path;

   bnv.clear();
   nx->getpathtoroot(path);  //path from n back to root
   nx->getbots(bnv);  //all bots ONLY BELOW node n!!

   #ifdef _OPENMP
      typedef tree::npv::size_type bvsz;
      siv.clear(); //need to setup space threads will add into
      siv.resize(bnv.size());
      for(bvsz i=0;i!=bnv.size();i++) siv[i]=newsinfo();
#     pragma omp parallel num_threads(tc)
      local_ompsubsuff(*di,nx,path,bnv,siv); //faster if pass di and bnv by value.
   #elif _OPENMPI
      diterator diter(di);
      local_mpisubsuff(diter,nx,path,bnv,siv);
   #else
      diterator diter(di);
      local_subsuff(diter,nx,path,bnv,siv);
   #endif
}

//--------------------------------------------------
//allsuff (2)
void brt::local_ompallsuff(dinfo di, tree::npv bnv,std::vector<sinfo*>& siv)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = di.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   std::vector<sinfo*>& tsiv = newsinfovec(); //will be sized in local_allsuff

   diterator diter(&di,beg,end);
   local_allsuff(diter,bnv,tsiv);

#  pragma omp critical
   {
      for(size_t i=0;i<siv.size();i++) *(siv[i]) += *(tsiv[i]);
   }
   
   for(size_t i=0;i<tsiv.size();i++) delete tsiv[i];
   delete &tsiv;
#endif
}

//--------------------------------------------------
// reset random number generator in MPI so it's the same on all nodes.
void brt::mpi_resetrn(rn& gen)
{
#ifdef _OPENMPI
   if(rank==0) {
      // reset the rn generator so they are the same on all nodes
      // so that we can draw random numbers in parallel on each node w/o communication.
      std::stringstream state;
      crn& tempgen=static_cast<crn&>(gen);

      state << tempgen.get_engine_state();
      unsigned long ulstate = std::stoul(state.str(),nullptr,0);

//      cout << "state is " << state.str() << " and ul is " << ulstate << endl;

      MPI_Request *request=new MPI_Request[tc];
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(&ulstate,1,MPI_UNSIGNED_LONG,i,MPI_TAG_RESET_RNG,MPI_COMM_WORLD,&request[i-1]);
      }
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
   }
   else
   {
      unsigned long ulstate;
      MPI_Status status;

      MPI_Recv(&ulstate,1,MPI_UNSIGNED_LONG,0,MPI_TAG_RESET_RNG,MPI_COMM_WORLD,&status); 

      std::string strstate=std::to_string(ulstate);
      std::stringstream state;
      state << strstate;

//      cout << "(slave) state is " << state.str() << " and ul is " << ulstate << endl;

      crn& tempgen=static_cast<crn&>(gen);
      tempgen.set_engine_state(state);
     }
/*   if(rank==0) {
      MPI_Request *request=new MPI_Request[tc];
      // reset the rn generator so they are the same on all nodes
      // so that we can draw random numbers in parallel on each node w/o communication.
      time_t timer;
      struct tm y2k = {0};
      int seconds;
      y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
      y2k.tm_year = 118; y2k.tm_mon = 0; y2k.tm_mday = 7;

      time(&timer);  // get current time
      seconds=(int)difftime(timer,mktime(&y2k));

      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(&seconds,1,MPI_INT,i,MPI_TAG_RESET_RNG,MPI_COMM_WORLD,&request[i-1]);
      }

      crn& tempgen=static_cast<crn&>(gen);
      tempgen.set_seed(seconds);
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
// cout << "0) Reset seconds: " << seconds << " gen.unif:" << gen.uniform() << " gen.unif:" << gen.uniform() << endl;
   }
   else
   {
      int seconds;
      MPI_Status status;
      MPI_Recv(&seconds,1,MPI_INT,0,MPI_TAG_RESET_RNG,MPI_COMM_WORLD,&status);
      crn& tempgen=static_cast<crn&>(gen);
      tempgen.set_seed(seconds);
// cout << "1) Reset seconds: " << seconds << " gen.unif:" << gen.uniform() << " gen.unif:" << gen.uniform() << endl;
   }*/
#endif
}
//--------------------------------------------------
//allsuff (2) -- MPI version
void brt::local_mpiallsuff(diterator& diter, tree::npv& bnv,std::vector<sinfo*>& siv)
{
#ifdef _OPENMPI
   if(rank==0) {
      siv.clear(); //need to setup space threads will add into
      siv.resize(bnv.size());
      typedef tree::npv::size_type bvsz;
      for(bvsz i=0;i!=bnv.size();i++) siv[i]=newsinfo();
 
      // reduce all the sinfo's across the nodes, which is model-specific.
      local_mpi_reduce_allsuff(siv);
   }
   else
   {
      local_allsuff(diter,bnv,siv);

      // reduce all the sinfo's across the nodes, which is model-specific.
      local_mpi_reduce_allsuff(siv);
   }
#endif
}
//--------------------------------------------------
//allsuff(2) -- the MPI communication part of local_mpiallsuff.  This is model-specific.
void brt::local_mpi_reduce_allsuff(std::vector<sinfo*>& siv)
{
#ifdef _OPENMPI
   unsigned int nvec[siv.size()];

   // cast to int
   for(size_t i=0;i<siv.size();i++)
      nvec[i]=(unsigned int)siv[i]->n;  // on root node, this should be 0 because of newsinfo().
// cout << "pre:" << siv[0]->n << " " << siv[1]->n << endl;
   // MPI sum
//   MPI_Allreduce(MPI_IN_PLACE,&nvec,siv.size(),MPI_UNSIGNED,MPI_SUM,MPI_COMM_WORLD);
   if(rank==0) {
      MPI_Status status;
      unsigned int tempvec[siv.size()];
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Recv(&tempvec,siv.size(),MPI_UNSIGNED,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
         for(size_t j=0;j<siv.size();j++)
            nvec[j]+=tempvec[j];
      }

      MPI_Request *request=new MPI_Request[tc];
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(&nvec,siv.size(),MPI_UNSIGNED,i,0,MPI_COMM_WORLD,&request[i-1]);
      }

      // cast back to size_t
      for(size_t i=0;i<siv.size();i++)
         siv[i]->n=(size_t)nvec[i];

      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
   }
   else {
      MPI_Request *request=new MPI_Request;
      MPI_Isend(&nvec,siv.size(),MPI_UNSIGNED,0,0,MPI_COMM_WORLD,request);
      MPI_Status status;
      MPI_Wait(request,MPI_STATUSES_IGNORE);
      delete request;

      MPI_Recv(&nvec,siv.size(),MPI_UNSIGNED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
      // cast back to size_t
      for(size_t i=0;i<siv.size();i++)
         siv[i]->n=(size_t)nvec[i];
   }

   // cast back to size_t
   // for(size_t i=0;i<siv.size();i++)
   //    siv[i]->n=(size_t)nvec[i];
// cout << "reduced:" << siv[0]->n << " " << siv[1]->n << endl;
#endif
}
//--------------------------------------------------
//allsuff (3)
void brt::local_allsuff(diterator& diter, tree::npv& bnv,std::vector<sinfo*>& siv)
{
   tree::tree_cp tbn; //the pointer to the bottom node for the current observations
   size_t ni;         //the  index into vector of the current bottom node

   typedef tree::npv::size_type bvsz;
   bvsz nb = bnv.size();
   siv.clear();
   siv.resize(nb);

   std::map<tree::tree_cp,size_t> bnmap;
   for(bvsz i=0;i!=bnv.size();i++) { bnmap[bnv[i]]=i; siv[i]=newsinfo(); }

   for(;diter<diter.until();diter++) {
      if(!randpath){tbn = t.bn(diter.getxp(),*xi);}else{tbn = randz[*diter];}      
      ni = bnmap[tbn];
      //siv[ni].n +=1; 
      add_observation_to_suff(diter, *(siv[ni]));
   }
}
/*
//--------------------------------------------------
//get suff stats for nodes related to change of variable proposal
//this is simply the allsuff for all nodes under the perturb node, not the entire tree.
void brt::getchgvsuff(tree::tree_p pertnode, tree::npv& bnv, size_t oldc, size_t oldv, bool didswap, 
                  std::vector<sinfo*>& sivold, std::vector<sinfo*>& sivnew)
{
   subsuff(pertnode,bnv,sivnew);
   if(didswap) pertnode->swaplr();  //undo the swap so we can calculate the suff stats for the original variable, cutpoint.
   pertnode->setv(oldv);
   pertnode->setc(oldc);
   subsuff(pertnode,bnv,sivold);
}

//--------------------------------------------------
//get suff stats for nodes related to perturb proposal
//this is simply the allsuff for all nodes under the perturb node, not the entire tree.
void brt::getpertsuff(tree::tree_p pertnode, tree::npv& bnv, size_t oldc, 
                  std::vector<sinfo*>& sivold, std::vector<sinfo*>& sivnew)
{
   subsuff(pertnode,bnv,sivnew);
   pertnode->setc(oldc);
   subsuff(pertnode,bnv,sivold);
}
*/
//--------------------------------------------------
//getsuff wrapper used for death.  Calls serial or parallel code depending on how
//the code is compiled.
void brt::getsuff(tree::tree_p l, tree::tree_p r, sinfo& sil, sinfo& sir)
{
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_ompgetsuff(l,r,*di,sil,sir); //faster if pass dinfo by value.
   #elif _OPENMPI
      local_mpigetsuff(l,r,*di,sil,sir);
   #else
         diterator diter(di);
         local_getsuff(diter,l,r,sil,sir);
   #endif
}

//--------------------------------------------------
//--------------------------------------------------
//#ifdef _OPENMP
//--------------------------------------------------
//openmp version of getsuff for birth
void brt::local_ompgetsuff(tree::tree_p nx, size_t v, size_t c, dinfo di, sinfo& sil, sinfo& sir)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = di.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   sinfo& tsil = *newsinfo();
   sinfo& tsir = *newsinfo();

   diterator diter(&di,beg,end);
   local_getsuff(diter,nx,v,c,tsil,tsir);

#  pragma omp critical
   {
      sil+=tsil; sir+=tsir;
   }

   delete &tsil;
   delete &tsir;
#endif
}
//--------------------------------------------------
//opemmp version of getsuff for death
void brt::local_ompgetsuff(tree::tree_p l, tree::tree_p r, dinfo di, sinfo& sil, sinfo& sir)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = di.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

//   sinfo tsil, tsir;
   sinfo& tsil = *newsinfo();
   sinfo& tsir = *newsinfo();

   diterator diter(&di,beg,end);
   local_getsuff(diter,l,r,tsil,tsir);

#  pragma omp critical
   {
      sil+=tsil; sir+=tsir;
   }

   delete &tsil;
   delete &tsir;
#endif
}
//#endif

//--------------------------------------------------
//--------------------------------------------------
//#ifdef _OPENMPI
//--------------------------------------------------
// MPI version of getsuff for birth
void brt::local_mpigetsuff(tree::tree_p nx, size_t v, size_t c, dinfo di, sinfo& sil, sinfo& sir)
{
#ifdef _OPENMPI
   if(rank==0) {
      char buffer[SIZE_UINT3];
      int position=0;
      MPI_Request *request=new MPI_Request[tc];
      const int tag=MPI_TAG_BD_BIRTH_VC;
      unsigned int vv,cc,nxid;

      vv=(unsigned int)v;
      cc=(unsigned int)c;
      nxid=(unsigned int)nx->nid();

      // Pack and send info to the slaves
      MPI_Pack(&nxid,1,MPI_UNSIGNED,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      MPI_Pack(&vv,1,MPI_UNSIGNED,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      MPI_Pack(&cc,1,MPI_UNSIGNED,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(buffer,SIZE_UINT3,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
      }
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);

      // MPI receive all the answers from the slaves
      local_mpi_sr_suffs(sil,sir);
      delete[] request;
   }
   else
   {
      diterator diter(&di);
      local_getsuff(diter,nx,v,c,sil,sir);

      // MPI send all the answers to root
      local_mpi_sr_suffs(sil,sir);
   }
#endif
}
//--------------------------------------------------
// MPI version of getsuff for death
void brt::local_mpigetsuff(tree::tree_p l, tree::tree_p r, dinfo di, sinfo& sil, sinfo& sir)
{
#ifdef _OPENMPI
   if(rank==0) {
      char buffer[SIZE_UINT3];
      int position=0;  
      MPI_Request *request=new MPI_Request[tc];
      const int tag=MPI_TAG_BD_DEATH_LR;
      unsigned int nlid,nrid;

      nlid=(unsigned int)l->nid();
      nrid=(unsigned int)r->nid();

      // Pack and send info to the slaves
      MPI_Pack(&nlid,1,MPI_UNSIGNED,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      MPI_Pack(&nrid,1,MPI_UNSIGNED,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      for(size_t i=1; i<=(size_t)tc; i++) {   
         MPI_Isend(buffer,SIZE_UINT3,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
      }
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);

      // MPI receive all the answers from the slaves
      local_mpi_sr_suffs(sil,sir);

      delete[] request;
   }
   else {
      diterator diter(&di);
      local_getsuff(diter,l,r,sil,sir);

      // MPI send all the answers to root
      local_mpi_sr_suffs(sil,sir);
   }
#endif
}
//--------------------------------------------------
// MPI virtualized part for sending/receiving left,right suffs
// This is model-dependent.
void brt::local_mpi_sr_suffs(sinfo& sil, sinfo& sir)
{
#ifdef _OPENMPI
   if(rank==0) { // MPI receive all the answers from the slaves
      MPI_Status status;
      sinfo& tsil = *newsinfo();
      sinfo& tsir = *newsinfo();
      char buffer[SIZE_UINT2];
      int position=0;
      unsigned int ln,rn;      
      for(size_t i=1; i<=(size_t)tc; i++) {
         position=0;
         MPI_Recv(buffer,SIZE_UINT2,MPI_PACKED,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
         MPI_Unpack(buffer,SIZE_UINT2,&position,&ln,1,MPI_UNSIGNED,MPI_COMM_WORLD);
         MPI_Unpack(buffer,SIZE_UINT2,&position,&rn,1,MPI_UNSIGNED,MPI_COMM_WORLD);
         tsil.n=(size_t)ln;
         tsir.n=(size_t)rn;
         sil+=tsil;
         sir+=tsir;
      }
      delete &tsil;
      delete &tsir;
   }
   else // MPI send all the answers to root
   {
      char buffer[SIZE_UINT2];
      int position=0;  
      unsigned int ln,rn;
      ln=(unsigned int)sil.n;
      rn=(unsigned int)sir.n;
      MPI_Pack(&ln,1,MPI_UNSIGNED,buffer,SIZE_UINT2,&position,MPI_COMM_WORLD);
      MPI_Pack(&rn,1,MPI_UNSIGNED,buffer,SIZE_UINT2,&position,MPI_COMM_WORLD);
      MPI_Send(buffer,SIZE_UINT2,MPI_PACKED,0,0,MPI_COMM_WORLD);
   }
#endif   
}

//--------------------------------------------------
//--------------------------------------------------
//set the vector of predicted values
void brt::setf() {
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_ompsetf(*di); //faster if pass dinfo by value.
   #else
         diterator diter(di);
         local_setf(diter);
   #endif
}
void brt::local_ompsetf(dinfo di)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = di.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   diterator diter(&di,beg,end);
   local_setf(diter);
#endif
}
void brt::local_setf(diterator& diter)
{
   tree::tree_p bn;

   for(;diter<diter.until();diter++) {
      bn = t.bn(diter.getxp(),*xi);
      yhat[*diter] = bn->gettheta();
   }
}
//--------------------------------------------------
//set the vector of residual values
void brt::setr() {
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_ompsetr(*di); //faster if pass dinfo by value.
   #else
         diterator diter(di);
         local_setr(diter);
   #endif
}
void brt::local_ompsetr(dinfo di)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = di.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   diterator diter(&di,beg,end);
   local_setr(diter);
#endif
}
void brt::local_setr(diterator& diter)
{
   tree::tree_p bn;

   for(;diter<diter.until();diter++) {
      bn = t.bn(diter.getxp(),*xi);
      resid[*diter] = 0.0 - bn->gettheta();
//      resid[*diter] = di->y[*diter] - bn->gettheta();
   }
}
//--------------------------------------------------
//predict the response at the (npred x p) input matrix *x
//Note: the result appears in *dipred.y.
void brt::predict(dinfo* dipred) {
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_omppredict(*dipred); //faster if pass dinfo by value.
   #else
         diterator diter(dipred);
         local_predict(diter);
   #endif
}
void brt::local_omppredict(dinfo dipred)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = dipred.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   diterator diter(&dipred,beg,end);
   local_predict(diter);
#endif
}
void brt::local_predict(diterator& diter)
{
   tree::tree_p bn;

   for(;diter<diter.until();diter++) {
      bn = t.bn(diter.getxp(),*xi);
      diter.sety(bn->gettheta());
   }
}
//--------------------------------------------------
//save/load tree to/from vector format
//Note: for single tree models the parallelization just directs
//      back to the serial path (ie no parallel execution occurs).
//      For multi-tree models, the parallelization occurs in the
//      definition of that models class.
//void brt::savetree(int* id, int* v, int* c, double* theta)
void brt::savetree(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
   #ifdef _OPENMP
#    pragma omp parallel num_threads(tc)
     local_ompsavetree(iter,m,nn,id,v,c,theta);
   #else
     int beg=0;
     int end=(int)m;
     local_savetree(iter,beg,end,nn,id,v,c,theta);
   #endif
}
//void brt::local_ompsavetree(int* id, int* v, int* c, double* theta)
void brt::local_ompsavetree(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = (int)m; //1 tree in brt version of save/load tree(s)
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);
   if(end>my_rank)
      local_savetree(iter,beg,end,nn,id,v,c,theta);
#endif
}
void brt::local_savetree(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, 
     std::vector<std::vector<int> >& v, std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
   //beg,end are not used in the single-tree models.
   nn[iter]=t.treesize();
   id[iter].resize(nn[iter]);
   v[iter].resize(nn[iter]);
   c[iter].resize(nn[iter]);
   theta[iter].resize(nn[iter]);
   t.treetovec(&id[iter][0],&v[iter][0],&c[iter][0],&theta[iter][0]);
}
void brt::loadtree(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
   #ifdef _OPENMP
#    pragma omp parallel num_threads(tc)
     local_omploadtree(iter,m,nn,id,v,c,theta);
   #else
     int beg=0;
     int end=(int)m;
     local_loadtree(iter,beg,end,nn,id,v,c,theta);
   #endif
}
//void brt::local_omploadtree(size_t nn, int* id, int* v, int* c, double* theta)
void brt::local_omploadtree(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = (int)m; //1 tree in brt version of save/load tree(s)
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);
   if(end>my_rank)
      local_loadtree(iter,beg,end,nn,id,v,c,theta);
#endif
}
void brt::local_loadtree(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
   //beg,end are not used in the single-tree models.
   t.vectotree(nn[iter],&id[iter][0],&v[iter][0],&c[iter][0],&theta[iter][0]);
}

//--------------------------------------------------
//--------------------------------------------------
//bd: birth/death
void brt::bd(rn& gen)
{
//   cout << "--------------->>into bd" << endl;
   tree::npv goodbots;  //nodes we could birth at (split on)
   double PBx = getpb(t,*xi,mi.pb,goodbots,tp); //prob of a birth at x

   if(gen.uniform() < PBx) { //do birth or death
      mi.bproposal++;
      //--------------------------------------------------
      //draw proposal
      tree::tree_p nx; //bottom node
      size_t v,c; //variable and cutpoint
      double pr; //part of metropolis ratio from proposal and prior
      bprop(t,*xi,tp,mi.pb,goodbots,PBx,nx,v,c,pr,gen);

      //--------------------------------------------------
      //compute sufficient statistics
      sinfo& sil = *newsinfo();
      sinfo& sir = *newsinfo();
      sinfo& sit = *newsinfo();

      getsuff(nx,v,c,sil,sir);
      // sit = sil + sir; NO! The + operator cannot be overloaded, so instead we do this:
      sit += sil;
      sit += sir;

      //--------------------------------------------------
      //compute alpha
      bool hardreject=true;
      double lalpha=0.0;
      double lml, lmr, lmt;  // lm is the log marginal left,right,total
      if((sil.n>=mi.minperbot) && (sir.n>=mi.minperbot)) { 
         lml=lm(sil); lmr=lm(sir); lmt=lm(sit);
         hardreject=false;
         lalpha = log(pr) + (lml+lmr-lmt);
         lalpha = std::min(0.0,lalpha);
      }
      //--------------------------------------------------
      //try metrop
      double thetal,thetar; //parameters for new bottom nodes, left and right
      double uu = gen.uniform();
#ifdef _OPENMPI
      MPI_Request *request = new MPI_Request[tc];
#endif
      if( !hardreject && (log(uu) < lalpha) ) {
         thetal = 0.0;//drawnodetheta(sil,gen);
         thetar = 0.0;//drawnodetheta(sir,gen);
         t.birthp(nx,v,c,thetal,thetar);
         mi.baccept++;
#ifdef _OPENMPI
//        cout << "accept birth " << lalpha << endl;
         const int tag=MPI_TAG_BD_BIRTH_VC_ACCEPT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
      else { //transmit reject over MPI
//        cout << "reject birth " << lalpha << endl;
         const int tag=MPI_TAG_BD_BIRTH_VC_REJECT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
#else
      }
#endif
      delete &sil;
      delete &sir;
      delete &sit;
#ifdef _OPENMPI
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
#endif
   } else {
      mi.dproposal++;
      //--------------------------------------------------
      //draw proposal
      double pr;  //part of metropolis ratio from proposal and prior
      tree::tree_p nx; //nog node to death at
      dprop(t,*xi,tp,mi.pb,goodbots,PBx,nx,pr,gen);

      //--------------------------------------------------
      //compute sufficient statistics
      //sinfo sil,sir,sit;
      sinfo& sil = *newsinfo();
      sinfo& sir = *newsinfo();
      sinfo& sit = *newsinfo();
      getsuff(nx->getl(),nx->getr(),sil,sir);
      // sit = sil + sir; NO! The + operator cannot be overloaded, so instead we do this:
      sit += sil;
      sit += sir;

      //--------------------------------------------------
      //compute alpha
      double lml, lmr, lmt;  // lm is the log marginal left,right,total
      lml=lm(sil); lmr=lm(sir); lmt=lm(sit);
      double lalpha = log(pr) + (lmt - lml - lmr);
      lalpha = std::min(0.0,lalpha);

      //--------------------------------------------------
      //try metrop
      double theta;
#ifdef _OPENMPI
      MPI_Request *request = new MPI_Request[tc];
#endif
      if(log(gen.uniform()) < lalpha) {
         theta = 0.0;//drawnodetheta(sit,gen);
         t.deathp(nx,theta);
         mi.daccept++;
#ifdef _OPENMPI
//        cout << "accept death " << lalpha << endl;
         const int tag=MPI_TAG_BD_DEATH_LR_ACCEPT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
      else { //transmit reject over MPI
//        cout << "reject death " << lalpha << endl;
         const int tag=MPI_TAG_BD_DEATH_LR_REJECT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
#else
      }
#endif
      delete &sil;
      delete &sir;
      delete &sit;
#ifdef _OPENMPI
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
#endif
   }
}
//--------------------------------------------------
//mpislave_bd: birth/death code on the slave side
void mpislave_bd(rn& gen)
{


}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//Model Mixing functions for brt.cpp
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//Draw function for vector parameters -- calls drawthetavec
void brt::drawvec(rn& gen)
{
   // Structural/topological proposal(s)
   if(gen.uniform()<mi.pbd){
//   if(mi.pbd>0.0)
      //std::cout << "bd" << std::endl;
      bd_vec(gen);
   }
   else
   {
      //std::cout << "Rotate" << std::endl; 
      tree::tree_p tnew;
      tnew=new tree(t); //copy of current to make life easier upon rejection
      //t.pr_vec();
      rot(tnew,t,gen);
      //t.pr_vec();
      delete tnew;
   }

   // Perturbation Proposal
   if(mi.dopert)
      pertcv(gen);


   // Shuffle step
   if(randpath){
      if(t.treesize()>1){
         shuffle_randz(gen);
      }
   }

   // Gibbs Step
    drawthetavec(gen);

   //update statistics
   if(mi.dostats) {
      tree::npv bnv; //all the bottom nodes
      for(size_t k=0;k< xi->size();k++) mi.varcount[k]+=t.nuse(k);
      t.getbots(bnv);
      unsigned int tempdepth[bnv.size()];
      unsigned int tempavgdepth=0;
      for(size_t i=0;i!=bnv.size();i++)
         tempdepth[i]=(unsigned int)bnv[i]->depth();
      for(size_t i=0;i!=bnv.size();i++) {
         tempavgdepth+=tempdepth[i];
         mi.tmaxd=std::max(mi.tmaxd,tempdepth[i]);
         mi.tmind=std::min(mi.tmind,tempdepth[i]);
      }
      mi.tavgd+=((double)tempavgdepth)/((double)bnv.size());
   }
}


//Draw theta vector -- samples the theta vector and assigns to tree 
void brt::drawthetavec(rn& gen)
{
   tree::npv bnv;
//   std::vector<sinfo> siv;
   std::vector<sinfo*>& siv = newsinfovec();

  allsuff(bnv,siv);
#ifdef _OPENMPI
  mpi_resetrn(gen);
#endif
   for(size_t i=0;i<bnv.size();i++) {
      // Update random hyperparameters if randhp is true
      if(randhp){
         bnv[i]->setthetahypervec(drawnodehypervec(*(siv[i]),gen));   
      }
      // Update thetavec
      bnv[i]->setthetavec(drawnodethetavec(*(siv[i]),gen));
      delete siv[i]; //set it, then forget it!
   }
   delete &siv;  //and then delete the vector of pointers.
}

//--------------------------------------------------
//draw theta for a single bottom node for the brt model
Eigen::VectorXd brt::drawnodethetavec(sinfo& si, rn& gen)
{
//   return 1.0;
   Eigen::VectorXd sin_vec(k); //cast si.n to a vector of dimension 1.
   for(size_t i = 0; i<k; i++){
      sin_vec(i) = si.n; //Input si.n into each vector component
   }
   return sin_vec;
}

//--------------------------------------------------
// Draw node hyperparameter vector for random hyper parameters 
std::vector<double> brt::drawnodehypervec(sinfo& si, rn& gen){
   // Placeholder, edit in model specific classes
   std::vector<double> temp(1,0);
   return temp;
}

//--------------------------------------------------
//slave controller for draw when using MPI
void brt::drawvec_mpislave(rn& gen)
{
   #ifdef _OPENMPI
   char buffer[SIZE_UINT3], bufferrp[SIZE_UINT3];
   int position=0;
   MPI_Status status;
   MPI_Status statusrp;
   typedef tree::npv::size_type bvsz;

   // Check random path proposal if using rpath
   if(randpath){ 
      unsigned int nxidrp,vrp,crp;      
      tree::tree_p nxrp;
      MPI_Recv(bufferrp,SIZE_UINT3,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&statusrp);
      MPI_Unpack(bufferrp,SIZE_UINT3,&position,&nxidrp,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      MPI_Unpack(bufferrp,SIZE_UINT3,&position,&vrp,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      MPI_Unpack(bufferrp,SIZE_UINT3,&position,&crp,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      nxrp=t.getptr((size_t)nxidrp);
      //cout << "nxrp = " << nxrp << "-- rank " << rank << endl;
      if(statusrp.MPI_TAG==MPI_TAG_RPATH_BIRTH_PROPOSAL)
      {
         randz_proposal(nxrp,vrp,crp,gen,true);
      }else if(statusrp.MPI_TAG==MPI_TAG_RPATH_DEATH_PROPOSAL)
      {
         randz_proposal(nxrp,vrp,crp,gen,false);
      }
      // reset position
      position = 0;
   }
   
   // Structural/topological proposal(s)
   // MPI receive the topological proposal type and nlid and nrid if applicable.
   MPI_Recv(buffer,SIZE_UINT3,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
   sinfo& tsil = *newsinfo();
   sinfo& tsir = *newsinfo();
   vxd theta0(k);
   theta0 = vxd::Zero(k);

   if(status.MPI_TAG==MPI_TAG_BD_BIRTH_VC) {
      unsigned int nxid,v,c;
      tree::tree_p nx;
      MPI_Unpack(buffer,SIZE_UINT3,&position,&nxid,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      MPI_Unpack(buffer,SIZE_UINT3,&position,&v,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      MPI_Unpack(buffer,SIZE_UINT3,&position,&c,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      nx=t.getptr((size_t)nxid);
      getsuff(nx,(size_t)v,(size_t)c,tsil,tsir);
      MPI_Status status2;
      MPI_Recv(buffer,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status2);
      if(status2.MPI_TAG==MPI_TAG_BD_BIRTH_VC_ACCEPT) t.birthp(nx,(size_t)v,(size_t)c,theta0,theta0); //accept birth
      if((status2.MPI_TAG==MPI_TAG_BD_BIRTH_VC_ACCEPT) & randpath){
         //cout << "ACCEPT AND UPDATE -- " << rank << endl;
         update_randz_bd(nx, true);
      }
      //else reject, for which we do nothing.
   }
   else if(status.MPI_TAG==MPI_TAG_BD_DEATH_LR) {
      unsigned int nlid,nrid;
      tree::tree_p nl,nr;
      MPI_Unpack(buffer,SIZE_UINT3,&position,&nlid,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      MPI_Unpack(buffer,SIZE_UINT3,&position,&nrid,1,MPI_UNSIGNED,MPI_COMM_WORLD);
      nl=t.getptr((size_t)nlid);
      nr=t.getptr((size_t)nrid);
      //if(randpath){randz_proposal(nl->p,(nl->p)->v,(nl->p)->c,gen,false);}
      getsuff(nl,nr,tsil,tsir);
      MPI_Status status2;
      MPI_Recv(buffer,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status2);
      if((status2.MPI_TAG==MPI_TAG_BD_DEATH_LR_ACCEPT) & randpath){update_randz_bd(nl->p, false);} // random path move for death
      if(status2.MPI_TAG==MPI_TAG_BD_DEATH_LR_ACCEPT) t.deathp(nl->getp(),theta0); //accept death
      
      //else reject, for which we do nothing.
   }
   else if(status.MPI_TAG==MPI_TAG_ROTATE) {
      mpi_resetrn(gen);
      tree::tree_p tnew;
      tnew=new tree(t); //copy of current to make life easier upon rejection
      rot(tnew,t,gen);
      delete tnew;
   }
   
   delete &tsil;
   delete &tsir;

   // Perturbation Proposal
   // nothing to perturb if tree is a single terminal node, so we would just skip.
   if(mi.dopert && t.treesize()>1)
   {
      tree::npv intnodes;
      tree::tree_p pertnode;
      t.getintnodes(intnodes);
      for(size_t pertdx=0;pertdx<intnodes.size();pertdx++)
      {
         std::vector<sinfo*>& sivold = newsinfovec();
         std::vector<sinfo*>& sivnew = newsinfovec();
         double oldsumlog, newsumlog;
         pertnode = intnodes[pertdx];
         MPI_Recv(buffer,SIZE_UINT3,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
         if(status.MPI_TAG==MPI_TAG_PERTCV)
         {
            size_t oldc = pertnode->getc();
            unsigned int propcint;
            position=0;
            MPI_Unpack(buffer,SIZE_UINT1,&position,&propcint,1,MPI_UNSIGNED,MPI_COMM_WORLD);
            size_t propc=(size_t)propcint;
            pertnode->setc(propc);
            tree::npv bnv;
            if(!randpath){getpertsuff(pertnode,bnv,oldc,sivold,sivnew);}else{getpertsuff_rpath(pertnode,bnv,oldc,oldsumlog,newsumlog);}
            MPI_Status status2;
            MPI_Recv(buffer,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status2);
            if(status2.MPI_TAG==MPI_TAG_PERTCV_ACCEPT) pertnode->setc(propc); //accept new cutpoint
            //else reject, for which we do nothing.
         }
         else if(status.MPI_TAG==MPI_TAG_PERTCHGV)
         {
            size_t oldc = pertnode->getc();
            size_t oldv = pertnode->getv();
            bool didswap=false;
            unsigned int propcint;
            unsigned int propvint;
            position=0;
            mpi_update_norm_cormat(rank,tc,pertnode,*xi,(*mi.corv)[oldv],chv_lwr,chv_upr);
            MPI_Recv(buffer,SIZE_UINT3,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
            MPI_Unpack(buffer,SIZE_UINT3,&position,&propcint,1,MPI_UNSIGNED,MPI_COMM_WORLD);
            MPI_Unpack(buffer,SIZE_UINT3,&position,&propvint,1,MPI_UNSIGNED,MPI_COMM_WORLD);
            MPI_Unpack(buffer,SIZE_UINT3,&position,&didswap,1,MPI_CXX_BOOL,MPI_COMM_WORLD);
            size_t propc=(size_t)propcint;
            size_t propv=(size_t)propvint;
            pertnode->setc(propc);
            pertnode->setv(propv);
            if(didswap)
               pertnode->swaplr();
            mpi_update_norm_cormat(rank,tc,pertnode,*xi,(*mi.corv)[propv],chv_lwr,chv_upr);
            tree::npv bnv;
            if(!randpath){
               getchgvsuff(pertnode,bnv,oldc,oldv,didswap,sivold,sivnew);
            }else{
               getchgvsuff_rpath(pertnode,bnv,oldc,oldv,didswap,oldsumlog,newsumlog);
            }
            MPI_Status status2;
            MPI_Recv(buffer,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status2);
            if(status2.MPI_TAG==MPI_TAG_PERTCHGV_ACCEPT) { //accept change var and pert
               pertnode->setc(propc);
               pertnode->setv(propv);
               if(didswap)
                  pertnode->swaplr();
            }
            // else reject, for which we do nothing.
         }
         // no other possibilities.
         for(bvsz j=0;j<sivold.size();j++) delete sivold[j];
         for(bvsz j=0;j<sivnew.size();j++) delete sivnew[j];
         delete &sivold;
         delete &sivnew;
      }
   }

   // Shuffle proposal for rpath
   if(randpath & t.treesize()>1){
      tree::npv rbnv;
      tree::tree_p root;
      vxd phix;
      vxd thetavec_temp(k); 
      std::map<tree::tree_p,double> lbmap;
      std::map<tree::tree_p,double> ubmap;
      std::map<tree::tree_p,tree::npv> pathmap;
      std::vector<sinfo*>& sivold = newsinfovec();
      std::vector<sinfo*>& sivnew = newsinfovec();

      diterator diter(di);
      root = t.getptr(t.nid());

      // Get bounds
      t.getbots(rbnv);
      get_phix_bounds(rbnv, lbmap, ubmap, pathmap);

      // Apply subsuff to the current assignments
      subsuff(root,rbnv,sivold);

      // Store current randz in the randz_shuffle, then clear randz
      randz_shuffle.clear();
      for(size_t i=0;i<randz.size();i++){randz_shuffle.push_back(randz[i]);}
      randz.clear();

      // Get proposed randz
      for(;diter<diter.until();diter++) {
         thetavec_temp = vxd::Zero(k);
         phix = vxd::Ones(rbnv.size());
         double *xx = diter.getxp(); 
         double u0 = gen.uniform();
         double prob = 0.0;
         get_phix(xx,phix,rbnv,lbmap,ubmap,pathmap);
         //if(phix.sum() != 1){cout << "sum to 1 error..." << phix.sum()<< endl;}
         for(size_t i=0;i<rbnv.size();i++){
            prob += phix(i);
            if((u0<prob)){
               randz.push_back(rbnv[i]); 
               break;  
            }   
         }
      }
      // Checker
      if(randz.size() != randz_shuffle.size()){
         cout << "ERROR: randz.size() != randz_shuffle.size()..." << randz_shuffle.size() << "---" << randz.size() << endl;
      }

      // Apply subsuff to the updated assignments
      subsuff(root,rbnv,sivnew);

      // Accept/reject proposal
      MPI_Status status3;
      MPI_Recv(buffer,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status3);
      if(status3.MPI_TAG==MPI_TAG_SHUFFLE_REJECT) { 
         // Reject the move, so sub in the original assingments
         randz.clear();
         for(size_t i=0;i<randz_shuffle.size();i++){randz.push_back(randz_shuffle[i]);}
         randz_shuffle.clear();
      }else{
         randz_shuffle.clear();
      }

   }

   // Gibbs Step
   drawthetavec(gen);

   #endif
}


//--------------------------------------------------
//Model Mixing Birth and Death
//--------------------------------------------------
//bd_vec: birth/death for vector parameters
void brt::bd_vec(rn& gen)
{
//   cout << "--------------->>into bd" << endl;
   tree::npv goodbots;  //nodes we could birth at (split on)
   double PBx = getpb(t,*xi,mi.pb,goodbots,tp); //prob of a birth at x

   if(gen.uniform() < PBx) { //do birth or death
      mi.bproposal++;
      //std::cout << "Birth" << std::endl;
      //--------------------------------------------------
      //draw proposal
      tree::tree_p nx; //bottom node
      size_t v,c; //variable and cutpoint
      double pr; //part of metropolis ratio from proposal and prior
      bprop(t,*xi,tp,mi.pb,goodbots,PBx,nx,v,c,pr,gen);
      
      if(randpath){randz_proposal(nx,v,c,gen,true);}

      //--------------------------------------------------
      //compute sufficient statistics
      sinfo& sil = *newsinfo();
      sinfo& sir = *newsinfo();
      sinfo& sit = *newsinfo();
      
      getsuff(nx,v,c,sil,sir);
      
      // sit = sil + sir; NO! The + operator cannot be overloaded, so instead we do this:
      sit += sil;
      sit += sir;

      //--------------------------------------------------
      //compute alpha
      bool hardreject=true;
      double lalpha=0.0;
      double lml, lmr, lmt;  // lm is the log marginal left,right,total
      if((sil.n>=mi.minperbot) && (sir.n>=mi.minperbot)) { 
         lml=lm(sil); lmr=lm(sir); lmt=lm(sit);
         hardreject=false;
         lalpha = log(pr) + (lml+lmr-lmt);
         if(randpath){lalpha = lalpha-rpi.logproppr;}
         //std::cout << "lml" << lml << std::endl;
         //std::cout << "lmr" << lmr << std::endl;
         //std::cout << "lmt" << lmt << std::endl;
         //std::cout << "lalpha = " << lalpha << std::endl;
         lalpha = std::min(0.0,lalpha);
      }

      //--------------------------------------------------
      //try metrop
      Eigen::VectorXd thetavecl,thetavecr; //parameters for new bottom nodes, left and right
      double uu = gen.uniform();
      //std::cout << "lu" << log(uu) << std::endl;
#ifdef _OPENMPI
      MPI_Request *request = new MPI_Request[tc];
#endif
      if( !hardreject && (log(uu) < lalpha) ) {
         thetavecl = Eigen::VectorXd:: Zero(k); 
         thetavecr = Eigen::VectorXd:: Zero(k); 
         t.birthp(nx,v,c,thetavecl,thetavecr);
         mi.baccept++;
         //cout << "ACCEPT & Update" << endl;
         if(randpath){update_randz_bd(nx, true);}
#ifdef _OPENMPI
//        cout << "accept birth " << lalpha << endl;
         const int tag=MPI_TAG_BD_BIRTH_VC_ACCEPT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
      else { //transmit reject over MPI
//        cout << "reject birth " << lalpha << endl;
         const int tag=MPI_TAG_BD_BIRTH_VC_REJECT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
#else
      }
#endif
      delete &sil;
      delete &sir;
      delete &sit;
#ifdef _OPENMPI
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
#endif
   } else {
      mi.dproposal++;
      //std::cout << "Death" << std::endl;
      //--------------------------------------------------
      //draw proposal
      double pr;  //part of metropolis ratio from proposal and prior
      tree::tree_p nx; //nog node to death at
      dprop(t,*xi,tp,mi.pb,goodbots,PBx,nx,pr,gen);
      
      if(randpath){randz_proposal(nx,nx->v,nx->c,gen,false);} // random path death step
      
      //--------------------------------------------------
      //compute sufficient statistics
      //sinfo sil,sir,sit;
      sinfo& sil = *newsinfo();
      sinfo& sir = *newsinfo();
      sinfo& sit = *newsinfo();
      getsuff(nx->getl(),nx->getr(),sil,sir);
      // sit = sil + sir; NO! The + operator cannot be overloaded, so instead we do this:
      sit += sil;
      sit += sir;

      //--------------------------------------------------
      //compute alpha
      double lml, lmr, lmt;  // lm is the log marginal left,right,total
      lml=lm(sil); lmr=lm(sir); lmt=lm(sit);
      double lalpha = log(pr) + (lmt - lml - lmr);
      if(randpath){lalpha = lalpha+rpi.logproppr;}
      lalpha = std::min(0.0,lalpha);

      //--------------------------------------------------
      //try metrop
      Eigen::VectorXd thetavec(k);
#ifdef _OPENMPI
      MPI_Request *request = new MPI_Request[tc];
#endif
      if(log(gen.uniform()) < lalpha) {
         thetavec = Eigen::VectorXd::Zero(k); 
         t.deathp(nx,thetavec);
         mi.daccept++;
         if(randpath){update_randz_bd(nx, false);}
#ifdef _OPENMPI
//        cout << "accept death " << lalpha << endl;
         const int tag=MPI_TAG_BD_DEATH_LR_ACCEPT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
      else { //transmit reject over MPI
//        cout << "reject death " << lalpha << endl;
         const int tag=MPI_TAG_BD_DEATH_LR_REJECT;
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
         }
      }
#else
      }
#endif
      delete &sil;
      delete &sir;
      delete &sit;
#ifdef _OPENMPI
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
#endif
   }
}

//--------------------------------------------------
//Model Mixing - set residuals and fitted values
//--------------------------------------------------
//set the vector of predicted values
void brt::setf_vec() {
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_ompsetf_vec(*di); //faster if pass dinfo by value.
   #else
         diterator diter(di);
         local_setf_vec(diter);
   #endif
}

void brt::local_ompsetf_vec(dinfo di)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = di.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   diterator diter(&di,beg,end);
   local_setf_vec(diter);
#endif
}

void brt::local_setf_vec(diterator& diter)
{
   tree::tree_p bn;
   vxd thetavec_temp(k); //Initialize a temp vector to facilitate the fitting
   for(;diter<diter.until();diter++) {
      if(!randpath){bn = t.bn(diter.getxp(),*xi);}else{bn = randz[*diter];}
      thetavec_temp = bn->getthetavec(); 
      yhat[*diter] = (*fi).row(*diter)*thetavec_temp;
   }
}

//--------------------------------------------------
//set the vector of residual values
void brt::setr_vec() {
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_ompsetr_vec(*di); //faster if pass dinfo by value.
   #else
         diterator diter(di);
         local_setr_vec(diter);
   #endif
}
void brt::local_ompsetr_vec(dinfo di)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = di.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   diterator diter(&di,beg,end);
   local_setr_vec(diter);
#endif
}
void brt::local_setr_vec(diterator& diter)
{
   tree::tree_p bn;
   vxd thetavec_temp(k); //Initialize a temp vector to facilitate the fitting

   // Set residual for random or deterministic path models
   if(randpath){
      for(;diter<diter.until();diter++) {
         bn = randz[*diter];
         thetavec_temp = bn->getthetavec();
         resid[*diter] = di->y[*diter] - (*fi).row(*diter)*thetavec_temp;
      }
   }else{
      for(;diter<diter.until();diter++) {
         bn = t.bn(diter.getxp(),*xi);
         thetavec_temp = bn->getthetavec();
         resid[*diter] = di->y[*diter] - (*fi).row(*diter)*thetavec_temp;
      }
   }
}

//--------------------------------------------------
//predict the response at the (npred x p) input matrix *x
//Note: the result appears in *dipred.y.
void brt::predict_vec(dinfo* dipred, finfo* fipred) {
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_omppredict_vec(*dipred, *fipred); //faster if pass dinfo by value.
   #else
         diterator diter(dipred);
         local_predict_vec(diter, *fipred);
   #endif
}

//Local predictions for model mixing over omp
void brt::local_omppredict_vec(dinfo dipred, finfo fipred)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = dipred.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   diterator diter(&dipred,beg,end);
   local_predict_vec(diter, fipred);
#endif
}

//Local preditions for model mixing
void brt::local_predict_vec(diterator& diter, finfo& fipred){
   tree::tree_p bn;
   vxd thetavec_temp(k); 
   for(;diter<diter.until();diter++) {
      bn = t.bn(diter.getxp(),*xi);
      thetavec_temp = bn->getthetavec();
      diter.sety(fipred.row(*diter)*thetavec_temp);
   }
}

//Mix using the discrepancy --- REMOVE -------
void brt::predict_mix_fd(dinfo* dipred, finfo* fipred, finfo* fpdmean, finfo* fpdsd, rn& gen) {
   size_t np = (*fpdmean).rows();
   finfo fdpred(np,k);
   double z;
   //Update the fdpred matrix to sum the point estimates + random discrepancy: fipred + fidelta 
   for(size_t i = 0; i<np; i++){
        for(size_t j=0; j<k;j++){
           z = gen.normal();
           fdpred(i,j) = (*fipred)(i,j) + (*fpdmean)(i,j) + (*fpdsd)(i,j)*z; 
        }
    }
   //cout << fdpred << endl;
   //Run the same functions -- just now using the updated prediction matrix
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_omppredict_vec(*dipred, fdpred); //faster if pass dinfo by value.
   #else
         diterator diter(dipred);
         local_predict_vec(diter, fdpred);
   #endif
}

//--------------------------------------------------
//Get modeling mixing weights
void brt::predict_thetavec(dinfo* dipred, mxd *wts){
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_ompget_predict_thetavec(*dipred, *wts); //faster if pass dinfo by value.
   #else
         diterator diter(dipred);
         local_predict_thetavec(diter, *wts);
   #endif   
}

void brt::local_predict_thetavec(diterator &diter, mxd &wts){
   tree::tree_p bn;
   vxd thetavec_temp(k); 
   for(;diter<diter.until();diter++) {
      bn = t.bn(diter.getxp(),*xi);
      thetavec_temp = bn->getthetavec();
      wts.col(*diter) = thetavec_temp; //sets the thetavec to be the ith column of the wts eigen matrix. 
   }
}

void brt::local_omppredict_thetavec(dinfo dipred, mxd wts){
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = dipred.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   diterator diter(&dipred,beg,end);
   local_predict_thetavec(diter, wts);
#endif
}

//--------------------------------------------------
//Get modeling mixing weights per tree
void brt::get_mix_theta(dinfo* dipred, mxd *wts){
   #ifdef _OPENMP
#     pragma omp parallel num_threads(tc)
      local_ompget_mix_theta(*dipred, *wts); //faster if pass dinfo by value.
   #else
         diterator diter(dipred);
         local_get_mix_theta(diter, *wts);
   #endif   
}

void brt::local_get_mix_theta(diterator &diter, mxd &wts){
   tree::tree_p bn;
   vxd thetavec_temp(k);
   bool enter = true; 
   for(;diter<diter.until();diter++) {
      bn = t.bn(diter.getxp(),*xi);
      thetavec_temp = bn->getthetavec();
      if(enter){
         wts.col(0) = thetavec_temp; //sets the thetavec to be the 1st column of the wts eigen matrix.
         enter = false;
      }
   }
}

void brt::local_ompget_mix_theta(dinfo dipred, mxd wts){
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = dipred.n;
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);

   diterator diter(&dipred,beg,end);
   local_get_mix_theta(diter, wts);
#endif
}


//--------------------------------------------------
//Print for brt with vector parameters
void brt::pr_vec(){
   std::cout << "***** brt object:\n";
#ifdef _OPENMPI
   std::cout << "mpirank=" << rank << endl;
#endif
   if(xi) {
      size_t p = xi->size();
      cout  << "**xi cutpoints set:\n";
      cout << "\tnum x vars: " << p << endl;
      cout << "\tfirst x cuts, first and last " << (*xi)[0][0] << ", ... ," << 
              (*xi)[0][(*xi)[0].size()-1] << endl;
      cout << "\tlast x cuts, first and last " << (*xi)[p-1][0] << ", ... ," << 
              (*xi)[p-1][(*xi)[p-1].size()-1] << endl;
   } else {
      cout << "**xi cutpoints not set\n";
   }
   if(di) {
      cout << "**data set, n,p: " << di->n << ", " << di->p << endl;
   } else {
      cout << "**data not set\n";
   }
   std::cout << "**the tree:\n";
   t.pr_vec();   
}


//--------------------------------------------------
//save/load tree to/from vector format -- for these functions, each double vector is of length k*nn. 
//Save tree with vector parameters
void brt::savetree_vec(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
   #ifdef _OPENMP
#    pragma omp parallel num_threads(tc)
     local_ompsavetree_vec(iter,m,nn,id,v,c,theta);
   #else
     int beg=0;
     int end=(int)m;
     local_savetree_vec(iter,beg,end,nn,id,v,c,theta);
   #endif
}

//--------------------------------------------------
//void brt::local_ompsavetree(int* id, int* v, int* c, double* theta)
void brt::local_ompsavetree_vec(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = (int)m; //1 tree in brt version of save/load tree(s)
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);
   if(end>my_rank)
      local_savetree_vec(iter,beg,end,nn,id,v,c,theta);
#endif
}

//--------------------------------------------------
void brt::local_savetree_vec(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, 
     std::vector<std::vector<int> >& v, std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
   //beg,end are not used in the single-tree models.
   nn[iter]=t.treesize();
   id[iter].resize(nn[iter]);
   v[iter].resize(nn[iter]);
   c[iter].resize(nn[iter]);
   theta[iter].resize(k*nn[iter]);

   //t.treetovec(&id[iter][0],&v[iter][0],&c[iter][0],&theta[iter][0]);
   t.treetovec(&id[iter][0],&v[iter][0],&c[iter][0],&theta[iter][0], k);
}

//--------------------------------------------------
void brt::loadtree_vec(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
   #ifdef _OPENMP
#    pragma omp parallel num_threads(tc)
     local_omploadtree_vec(iter,m,nn,id,v,c,theta);
   #else
     int beg=0;
     int end=(int)m;
     local_loadtree_vec(iter,beg,end,nn,id,v,c,theta);
   #endif
}
//--------------------------------------------------
//void brt::local_omploadtree(size_t nn, int* id, int* v, int* c, double* theta)
void brt::local_omploadtree_vec(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = (int)m; //1 tree in brt version of save/load tree(s)
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);
   if(end>my_rank)
      local_loadtree_vec(iter,beg,end,nn,id,v,c,theta);
#endif
}

//--------------------------------------------------
void brt::local_loadtree_vec(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta)
{
   //beg,end are not used in the single-tree models.
   t.vectotree(nn[iter],&id[iter][0],&v[iter][0],&c[iter][0],&theta[iter][0],k);
}


//--------------------------------------------------
// Save tree with hyperparameters
//--------------------------------------------------
// Save function
void brt::savetree_vec(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta, std::vector<std::vector<double> >& hyper)
{
   #ifdef _OPENMP
#    pragma omp parallel num_threads(tc)
     local_ompsavetree_vec(iter,m,nn,id,v,c,theta,hyper);
   #else
     int beg=0;
     int end=(int)m;
     local_savetree_vec(iter,beg,end,nn,id,v,c,theta,hyper);
   #endif
}

//--------------------------------------------------
//void brt::local_ompsavetree(int* id, int* v, int* c, double* theta)
void brt::local_ompsavetree_vec(size_t iter, size_t m, std::vector<int>& nn, std::vector<std::vector<int> >& id, std::vector<std::vector<int> >& v,
                  std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta, std::vector<std::vector<double> >& hyper)
{
#ifdef _OPENMP
   int my_rank = omp_get_thread_num();
   int thread_count = omp_get_num_threads();
   int n = (int)m; //1 tree in brt version of save/load tree(s)
   int beg=0;
   int end=0;
   calcbegend(n,my_rank,thread_count,&beg,&end);
   if(end>my_rank)
      local_savetree_vec(iter,beg,end,nn,id,v,c,theta,hyper);
#endif
}

//--------------------------------------------------
void brt::local_savetree_vec(size_t iter, int beg, int end, std::vector<int>& nn, std::vector<std::vector<int> >& id, 
     std::vector<std::vector<int> >& v, std::vector<std::vector<int> >& c, std::vector<std::vector<double> >& theta,
     std::vector<std::vector<double> >& hyper)
{
   //beg,end are not used in the single-tree models.
   nn[iter]=t.treesize();
   id[iter].resize(nn[iter]);
   v[iter].resize(nn[iter]);
   c[iter].resize(nn[iter]);
   theta[iter].resize(k*nn[iter]);
   hyper[iter].resize(kp*nn[iter]);

   //t.treetovec(&id[iter][0],&v[iter][0],&c[iter][0],&theta[iter][0]);
   t.treetovec(&id[iter][0],&v[iter][0],&c[iter][0],&theta[iter][0],&hyper[iter][0],k,kp);
}




//--------------------------------------------------
// Random Path Functions -- Move Elsewhere eventually
//--------------------------------------------------
// predict_vec_rpath
// ----- need path to root and loop through the root per bnv
void brt::get_phix_matrix(diterator &diter, mxd &phix, tree::npv bnv, size_t np){
   tree::npv path; 
   tree::tree_p p0, n0;
   mxd logphix;
   std::map<tree::tree_p,double> lbmap;
   std::map<tree::tree_p,double> ubmap;
   std::map<tree::tree_p,tree::npv> pathmap;
   int L,U, v0, c0;
   double lb, ub, psi0;

   phix = mxd::Ones(np, bnv.size());
   phix = (1/bnv.size())*phix; // init as a random draw, should always be changed unless node size == 1
   logphix = mxd::Zero(np, bnv.size());
   if(bnv.size()>1){
      // Get the upper and lower bounds for each path
      get_phix_bounds(bnv,lbmap,ubmap,pathmap);

      for(;diter<diter.until();diter++){
         double *xx = diter.getxp();
         for(size_t j = 0;j<bnv.size();j++){
            for(size_t l=0;l<(pathmap[bnv[j]].size()-1);l++){
               n0 = pathmap[bnv[j]][l];
               p0 = n0->p;
               v0 = p0->v;
               c0 = (*xi)[v0][p0->c];
               ub = ubmap[p0];
               lb = lbmap[p0];
               psi0 = psix(rpi.gamma,xx[v0],c0,lb,ub);
               if((n0->nid())%2 == 0){
                  // Left move prob
                  logphix(*diter,j)=logphix(*diter,j)+log(1-psi0);
               }else{
                  // Right move prob
                  logphix(*diter,j)=logphix(*diter,j)+log(psi0);
               }
               if(std::isnan(logphix(*diter,j))){
                  /*
                  cout << "logphi(x) nan ... " << endl;
                  cout << "psi0 = " << psi0 << endl;
                  cout << "rpi.gamma = " << rpi.gamma << endl;
                  cout << "c0 = " << c0 << endl;
                  cout << "v0 = " << v0 << endl;
                  cout << "ub = " << ub << endl;
                  cout << "lb = " << lb << endl;
                  */
               }
            }
            // Convert back to phix scale
            phix(*diter,j)=exp(logphix(*diter,j));
         }
      }
   }   
}


//--------------------------------------------------
// Get phi(x) bounds
void brt::get_phix_bounds(tree::npv bnv, std::map<tree::tree_p,double> &lbmap, std::map<tree::tree_p,double> &ubmap,
                        std::map<tree::tree_p,tree::npv> &pathmap) 
{ 
   tree::npv path; 
   tree::tree_p p0, n0;
   std::vector<double> ubvectemp, lbvectemp;
   int L,U, v0;
   double lb, ub;
   // Get the upper and lower bounds for each path
   for(size_t j=0;j<bnv.size();j++){
      path.clear();
      bnv[j]->getpathtoroot(path);
      pathmap[bnv[j]] = path;
      for(size_t l=0;l<(path.size()-1);l++){
         // Reset L and U to min and max & then update
         L=std::numeric_limits<int>::min(); U=std::numeric_limits<int>::max();
         n0 = path[l];
         p0 = path[l]->p; // get the parent of the current node on the path
         v0 = p0->v; // get its split var
         if(lbmap.find(p0) == lbmap.end()){
            L=std::numeric_limits<int>::min(); U=std::numeric_limits<int>::max();
            p0->rgi(v0,&L,&U);
            
            // Now we have the interval endpoints, put corresponding values in a,b matrices.
            if(L!=std::numeric_limits<int>::min()){ 
                  lb=(*xi)[v0][L];
            }else{
                  lb=(*xi)[v0][0];
            }
            if(U!=std::numeric_limits<int>::max()) {
                  ub=(*xi)[v0][U];
            }else{
                  ub=(*xi)[v0][(*xi)[v0].size()-1];
            }
            // Store in the maps
            lbmap[p0] = lb;
            ubmap[p0] = ub;
         }            
      }
   }
}


void brt::get_phix(double *xx, vxd &phixvec, tree::npv bnv, std::map<tree::tree_p,double> &lbmap, std::map<tree::tree_p,double> &ubmap,
                        std::map<tree::tree_p,tree::npv> &pathmap){
   vxd logphix;
   tree::tree_p n0,p0;
   size_t v0;
   double c0, ub, lb, psi0;
   logphix = vxd::Zero(bnv.size());

   if(bnv.size()>1){
      for(size_t j = 0;j<bnv.size();j++){
         for(size_t l=0;l<(pathmap[bnv[j]].size()-1);l++){
            n0 = pathmap[bnv[j]][l];
            p0 = n0->p;
            v0 = p0->v;
            c0 = (*xi)[v0][p0->c];
            ub = ubmap[p0];
            lb = lbmap[p0];
            psi0 = psix(rpi.gamma,xx[v0],c0,lb,ub);
            if((n0->nid())%2 == 0){
               // Left move prob
               logphix(j)=logphix(j)+log(1-psi0);
            }else{
               // Right move prob
               logphix(j)=logphix(j)+log(psi0);
            }
            if(std::isnan(logphix(j))){
               cout << "logphi(x) nan ... " << endl;
               cout << "psi0 = " << psi0 << endl;
               cout << "rpi.gamma = " << rpi.gamma << endl;
               cout << "c0 = " << c0 << endl;
               cout << "v0 = " << v0 << endl;
               cout << "ub = " << ub << endl;
               cout << "lb = " << lb << endl;
            }
         }
         // Convert back to phix scale
         phixvec(j)=exp(logphix(j));
      }
   }else{
      phixvec = vxd::Ones(1);
   }
}


//--------------------------------------------------
void brt::predict_vec_rpath(dinfo* dipred, finfo* fipred){
   diterator diter(dipred);
   //diterator diterphix(dipred);
   local_predict_vec_rpath(diter,*fipred);        
}


void brt::local_predict_vec_rpath(diterator& diter, finfo& fipred){
   tree::npv bnv;
   vxd phix;
   vxd thetavec_temp(k); 
   std::map<tree::tree_p,double> lbmap;
   std::map<tree::tree_p,double> ubmap;
   std::map<tree::tree_p,tree::npv> pathmap;

   // Get bots and then get the path and bounds
   t.getbots(bnv);
   get_phix_bounds(bnv, lbmap, ubmap, pathmap);

   for(;diter<diter.until();diter++) {
      thetavec_temp = vxd::Zero(k);
      phix = vxd::Ones(bnv.size());
      double *xx = diter.getxp(); 
      get_phix(xx,phix,bnv,lbmap,ubmap,pathmap);
      for(size_t i=0;i<bnv.size();i++){
         thetavec_temp = thetavec_temp + (bnv[i]->getthetavec())*phix(i);
      }
      diter.sety(fipred.row(*diter)*thetavec_temp);
   }   
}

/*
void brt::local_predict_vec_rpath(diterator& diterphix, diterator& diter, finfo& fipred){
   tree::npv bnv;
   mxd phix;
   vxd thetavec_temp(k); 
   t.getbots(bnv);
   get_phix_matrix(diter,phix,bnv,fipred.rows());
   // Fix by removing diter below...
   for(;diter<diter.until();diter++) {
      thetavec_temp = vxd::Zero(k);
      for(size_t i=0;i<bnv.size();i++){
         thetavec_temp = thetavec_temp + (bnv[i]->getthetavec())*phix(*diter,i);
      }
      diter.sety(fipred.row(*diter)*thetavec_temp);
   }   
}
*/


void brt::predict_thetavec_rpath(dinfo* dipred, mxd* wts){
   diterator diter(dipred);
   //diterator diterphix(dipred);
   local_predict_thetavec_rpath(diter,*wts);          
}


void brt::local_predict_thetavec_rpath(diterator& diter, mxd& wts){
   tree::npv bnv;
   vxd phix;
   vxd thetavec_temp(k); 
   std::vector<std::vector<double>> lbvec;
   std::vector<std::vector<double>> ubvec;
   std::map<tree::tree_p,double> lbmap;
   std::map<tree::tree_p,double> ubmap;
   std::map<tree::tree_p,tree::npv> pathmap;

   // Get bots and then get the path and bounds
   t.getbots(bnv);
   get_phix_bounds(bnv, lbmap, ubmap, pathmap);

   // Fix by removing diter below...
   for(;diter<diter.until();diter++){
      thetavec_temp = vxd::Zero(k);
      phix = vxd::Ones(bnv.size());
      double *xx = diter.getxp(); 
      get_phix(xx,phix,bnv,lbmap,ubmap,pathmap);
      for(size_t i=0;i<bnv.size();i++){
         thetavec_temp = thetavec_temp + (bnv[i]->getthetavec())*phix(i);
      }
      wts.col(*diter) = thetavec_temp; //sets the thetavec to be the ith column of the wts eigen matrix. 
   }   
}

/*
void brt::local_predict_thetavec_rpath(diterator& diterphix, diterator& diter, mxd& wts){
   tree::npv bnv;
   vxd thetavec_temp(k); 
   mxd phix;
   t.getbots(bnv);
   get_phix_matrix(diterphix,phix,bnv,wts.cols());
   for(;diter<diter.until();diter++) {
      thetavec_temp = vxd::Zero(k);
      for(size_t i=0;i<bnv.size();i++){
         thetavec_temp = thetavec_temp + (bnv[i]->getthetavec())*phix(*diter,i);
      }
      wts.col(*diter) = thetavec_temp; //sets the thetavec to be the ith column of the wts eigen matrix. 
   }
}
*/

//--------------------------------------------------
// Random Path Class Methods - Move elsewhere eventually
//--------------------------------------------------
// Draw gamma -- base
void brt::drawgamma(rn &gen){
   // Get new proposal for gamma
   double newgam = 0;
   double old_sumlogphix = 0.0, new_sumlogphix = 0.0;
   newgam = rpi.gamma + (gen.uniform()-0.5)*rpi.propwidth;
   // Pass the new gamma using mpi  
#ifdef _OPENMPI
   if(rank==0){ //should always be true when using mpi
      char buffer[SIZE_UINT3];
      int position=0;
      MPI_Request *request=new MPI_Request[tc];
      const int tag=MPI_TAG_RPATHGAMMA;
   
      // Pack and send info to the slaves
      MPI_Pack(&newgam,1,MPI_DOUBLE,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(buffer,SIZE_UINT3,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
      }
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);

      delete[] request;
   }
#else
   // With the old gammma, get the sumlogphix
   old_sumlogphix = sumlogphix(rpi.gamma,t.getptr(t.nid()));

   // With the new gammma, get the sumlogphix 
   if(newgam < 0 || newgam > 1){
      new_sumlogphix = -std::numeric_limits<double>::infinity();
   }else{
      new_sumlogphix = sumlogphix(newgam,t.getptr(t.nid()));
   }
   
#endif
   
   // Send the sum log term back (mpi is handled in this function)
   rpath_mhstep(old_sumlogphix,new_sumlogphix,newgam,gen);
}


// Draw gamma -- mpi version
void brt::drawgamma_mpi(rn &gen){
   double newgam;
   double old_sumlogphix = 0, new_sumlogphix = 0;
#ifdef _OPENMPI
   int buffer_size = SIZE_UINT3;
   char buffer[buffer_size];
   int position=0;
   MPI_Status status;

   // MPI receive the new gamma and unpack it
   MPI_Recv(buffer,buffer_size,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
   MPI_Unpack(buffer,buffer_size,&position,&newgam,1,MPI_DOUBLE,MPI_COMM_WORLD);

   // Compute the log sum terms
   old_sumlogphix = sumlogphix(rpi.gamma,t.getptr(t.nid()));
   if(newgam < 0 || newgam > 1){
      new_sumlogphix = -std::numeric_limits<double>::infinity();
   }else{
      new_sumlogphix = sumlogphix(newgam, t.getptr(t.nid()));
   }
#endif

   // Send the sum log term back (mpi is handled in this function)
   rpath_mhstep(old_sumlogphix,new_sumlogphix,newgam,gen);

}


// Adapt for rpath
void brt::rpath_adapt(){
   double accrate;
   accrate=((double)rpi.accept)/((double)(rpi.accept+rpi.reject));
   if(accrate>0.29 || accrate<0.19) rpi.propwidth*=accrate/0.24;
   rpi.accept = 0;
   rpi.reject = 0;
}


// Get sum log(phix) for each obs on the processor
// This can be used for the entire tree or a subset of the tree starting at node nx
double brt::sumlogphix(double gam, tree::tree_p nx){
   double sumlogphix = 0;
   double psi0 = 0;
   int v0, U, L;
   double lb,ub, c0;
   tree::tree_p p0, n0;
   diterator diter(di);
   tree::npv bnv;
   tree::npv path;
   std::map<tree::tree_p,double> lbmap;
   std::map<tree::tree_p,double> ubmap;
   std::map<tree::tree_p,tree::npv> pathmap;

   //t.getbots(bnv);
   nx->getbots(bnv);
   // Get the rectangles at each internal node
   for(size_t i=0;i<bnv.size();i++){   
      path.clear();
      bnv[i]->getpathtoroot(path);
      pathmap[bnv[i]] = path;
      for(size_t j=0;j<(path.size()-1);j++){
         p0 = path[j]->p;
         v0 = p0->v;
         // Add new p0 to map if not already included (CHECK SECOND CONDITION)
         if(lbmap.find(p0) == lbmap.end() && p0 != 0){
            L=std::numeric_limits<int>::min(); U=std::numeric_limits<int>::max();
            p0->rgi(v0,&L,&U);
            
            // Now we have the interval endpoints, put corresponding values in a,b matrices.
            if(L!=std::numeric_limits<int>::min()){ 
                  lb=(*xi)[v0][L];
            }else{
                  lb=(*xi)[v0][0];
            }
            if(U!=std::numeric_limits<int>::max()) {
                  ub=(*xi)[v0][U];
            }else{
                  ub=(*xi)[v0][(*xi)[v0].size()-1];
            }
            // Store in the maps
            lbmap[p0] = lb;
            ubmap[p0] = ub;
         }
      }
   }

   //if(rank == 1) cout << "randz.size() = " << randz.size() << " ---- " << rank << endl;
   for(;diter<diter.until();diter++){
      double *xx = diter.getxp();
      if(pathmap.find(randz[*diter]) != pathmap.end()){
         for(size_t j=0;j<(pathmap[randz[*diter]].size()-1);j++){
            // Get cutpoint
            //if(randz[*diter][j].p != 0){
            n0 = pathmap[randz[*diter]][j];
            p0 = n0->p; // randz[*diter][j].p;
            v0 = p0->v; // (randz[*diter][j].p)->v;
            c0 = (*xi)[v0][p0->c]; // (*xi)[v0][(randz[*diter][j].p)->c];
            lb = lbmap[p0];
            ub = ubmap[p0];

            // Need to account for left vs right move here
            psi0 = psix(gam,xx[v0],c0,lb,ub);
            if((n0->nid()%2) == 0){
               // node derived from left move
               /*
               cout << "v = " << v0 << endl;
               cout << "c = " << c0 << endl;
               cout << "ub = " << ub << endl;
               cout << "lb = " << lb << endl;
               cout << "x = " << xx[v0] << endl;
               cout << "n0->nid() = " << n0->nid() << endl;
               cout << "psi0 = " << psi0 << endl;
               */
               sumlogphix=sumlogphix+log(1-psi0); 
            }else{
               // node derived from right move
               sumlogphix=sumlogphix+log(psi0); 
            }
            //if(rank == 1) cout << "contributed to sum...." << endl; 
            //}else{
               // psi0 = 1, log(psi0)  = 0
               // No update to sumlogphix
            //}
         }
      }
   }
   return(sumlogphix);
}



double brt::psix(double gam, double x, double c, double L, double U){
   double psi = 0.0;
   double a, b, d1, d2;   
   a = c - (c-L)*gam;
   b = c + (U-c)*gam;
   d1 = (c-x)/(c-a);
   d2 = (x-c)/(b-c);
   
   if(c<L || c>U){
      cout << "ERROR: c is out of [L,U]..." << endl;
   }

   if(x<c){
      psi = 0.5*std::pow(std::max(1-d1,0.0),rpi.q);
      if(psi>1){   
         cout << "L = " << L << endl;
         cout << "U = " << U << endl;
         cout << "a = " << a << endl;
         cout << "c = " << c << endl;
         cout << "gamma = " << gam << endl;
         cout << "x = " << x << endl;
         cout << "d1 = " << d1 << endl;
      }
   }else{
      psi = 1-0.5*std::pow(std::max(1-d2,0.0),rpi.q);
      if(psi>1){   
         cout << "L = " << L << endl;
         cout << "U = " << U << endl;
         cout << "b = " << b << endl;
         cout << "c = " << c << endl;
         cout << "d2 = " << d2 << endl;
      }
   }
   return(psi);
}


// MH step for updating gamma
void brt::rpath_mhstep(double osum, double nsum, double newgam,rn &gen){
#ifdef _OPENMPI
      if(rank>0){
         // Sum the suff stats then get accept/reject status
         MPI_Status status;
         MPI_Reduce(&osum,NULL,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
         MPI_Reduce(&nsum,NULL,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
         MPI_Recv(NULL,0,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
         // Carry out the accept/reject step
         if(status.MPI_TAG==MPI_TAG_RPATHGAMMA_ACCEPT) {
               rpi.accept+=1;
               rpi.gamma = newgam;
         }else{
               rpi.reject+=1;
         }
      }           

      if(rank==0){
         // Reduce the suff stats
         MPI_Request *request = new MPI_Request[tc];
         MPI_Reduce(MPI_IN_PLACE,&osum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
         MPI_Reduce(MPI_IN_PLACE,&nsum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

         // Do mhstep
         double lnprior, loprior;
         if(newgam > 1 || newgam < 0){
            lnprior = -std::numeric_limits<double>::infinity();
         }else{
            lnprior = (rpi.shp1-1)*log(newgam) + (rpi.shp2-1)*log(1-newgam);
         }
         
         loprior = (rpi.shp1-1)*log(rpi.gamma) + (rpi.shp2-1)*log(1-rpi.gamma);
         
         //cout << "lnprior = " << lnprior << endl;
         //cout << "loprior = " << loprior << endl;
         //cout << "nsum = " << nsum << endl;
         //cout << "osum = " << osum << endl;

         double lmout = nsum+lnprior-osum-loprior;
         double alpha = gen.uniform();
         //cout << "log alpha = " << log(alpha) << endl;
         //cout << "lmout = " << lmout << endl;
         if((log(alpha)<lmout) & (newgam<1 || newgam>0)){
            // accept
            rpi.gamma = newgam;
            rpi.accept+=1;
            const int tag=MPI_TAG_RPATHGAMMA_ACCEPT;
            for(size_t l=1; l<=(size_t)tc; l++){
               MPI_Isend(NULL,0,MPI_PACKED,l,tag,MPI_COMM_WORLD,&request[l-1]);
            }
         }else{ 
            // reject
            const int tag=MPI_TAG_RPATHGAMMA_REJECT;
            //cout << "reject here" << endl;
            for(size_t l=1; l<=(size_t)tc; l++) {
               MPI_Isend(NULL,0,MPI_PACKED,l,tag,MPI_COMM_WORLD,&request[l-1]);
            }
            rpi.reject+=1;
         }
         //if(newgam<0){cout << "gamma = " << rpi.gamma << "----" << rank << endl;}
         MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
         delete[] request;
      }
#else
    // Do mhstep
   double lnprior, loprior;
   if(newgam > 1 || newgam < 0){
      lnprior = -std::numeric_limits<double>::infinity();
   }else{
      lnprior = (rpi.shp1-1)*log(newgam) + (rpi.shp2-1)*log(1-newgam);
   }
   loprior = (rpi.shp1-1)*log(rpi.gamma) + (rpi.shp2-1)*log(1-rpi.gamma);

   double lmout = nsum+lnprior-osum-loprior;
   double alpha=gen.uniform();
   if(log(alpha)<lmout){
      // accept
      rpi.gamma = newgam;
      rpi.accept+=1;
   }else{ 
      // reject
      rpi.reject+=1;
   }
#endif

}

// Proposal distribution for random path assignments (birth)
void brt::randz_proposal(tree::tree_p nx, size_t v, size_t c, rn& gen, bool birth){
   //cout << "Enter randz proposal..." << endl;
   diterator diter(di);
   double *xx; 
   double psi0;
   double ub, lb;
   //drandouble logproppr = 0.0;
   int L, U;
   double cx;

   // Get upper and lower bounds...make this into a function
   L=std::numeric_limits<int>::min(); U=std::numeric_limits<int>::max();
   nx->rgi(v,&L,&U);
   cx = (*xi)[v][c];
   // Get interval endpoints
   if(L!=std::numeric_limits<int>::min()){ 
         lb=(*xi)[v][L];
   }else{
         lb=(*xi)[v][0];
   }
   if(U!=std::numeric_limits<int>::max()) {
         ub=(*xi)[v][U];
   }else{
         ub=(*xi)[v][(*xi)[v].size()-1];
   }
   // Reset rpi proposal probability
   rpi.logproppr = 0; 
   // Assign the z's for the right and left node
   randz_bdp.clear();
   if(birth){
      for(;diter<diter.until();diter++){
         if(nx==randz[*diter]){ //does the bottom node = xx's bottom node
            // compute psi(x), the prob of moving right
            //cout << "nx in randz = " << nx << endl;
            xx = diter.getxp();
            //if(rank == 1) cout << "xx[v} = " << xx[v] << endl;
            psi0 = psix(rpi.gamma,xx[v],cx,lb,ub);
            //if(rank == 1) cout << "psi0 = " << psi0 << endl;   
            if(gen.uniform()<psi0){
               // Rightrandz_p
               randz_bdp.push_back(2);
               rpi.logproppr+=log(psi0);      
            } else {
               // Left
               randz_bdp.push_back(1);
               rpi.logproppr+=log((1-psi0));
            }
         }else{
            // Not involved
            randz_bdp.push_back(0);
         }
      }
   }else{
      // No need to modify the randz vector, just compute the log proposal probs
      for(;diter<diter.until();diter++){
         if((nx->r)==randz[*diter]){
            // compute psi(x), the prob of moving right
            xx = diter.getxp();      
            psi0 = psix(rpi.gamma,xx[v],cx,lb,ub);   
            rpi.logproppr+=log(psi0);      
            randz_bdp.push_back(2);
         }else if((nx->l)==randz[*diter]){
            // compute 1-psi(x), the prob of moving left
            xx = diter.getxp();      
            psi0 = psix(rpi.gamma,xx[v],cx,lb,ub);   
            rpi.logproppr+=log((1-psi0));
            randz_bdp.push_back(1);
         }else{
            randz_bdp.push_back(0);
         }
      }
   }
   
   // Pass the proposal probability
   mpi_randz_proposal(rpi.logproppr,nx,v,c,birth);
}


void brt::mpi_randz_proposal(double &logproppr, tree::tree_p nx, size_t v, size_t c, bool birth){
#ifdef _OPENMPI
   if(rank==0) {
      MPI_Status status;
      char buffer[SIZE_UINT3], buffer2[SIZE_UINT2];
      int position=0;
      MPI_Request *request=new MPI_Request[tc];
      double tlpp;
      unsigned int vv,cc,nxid;
      int tag;
      if(birth){
         tag=MPI_TAG_RPATH_BIRTH_PROPOSAL;
      }else{
         tag=MPI_TAG_RPATH_DEATH_PROPOSAL;
      }

      vv=(unsigned int)v;
      cc=(unsigned int)c;
      nxid=(unsigned int)nx->nid();

      //**** This is terribly inefficient *****

      // Pack and send info to the slaves
      MPI_Pack(&nxid,1,MPI_UNSIGNED,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      MPI_Pack(&vv,1,MPI_UNSIGNED,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      MPI_Pack(&cc,1,MPI_UNSIGNED,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(buffer,SIZE_UINT3,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
      }
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      
      // Now receive the values from each processor
      for(size_t i=1; i<=(size_t)tc; i++) {
         position=0;
         MPI_Recv(buffer2,SIZE_UINT2,MPI_PACKED,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
         MPI_Unpack(buffer2,SIZE_UINT2,&position,&tlpp,1,MPI_DOUBLE,MPI_COMM_WORLD);
         logproppr+=tlpp;
      }
      delete[] request;
   }
   else
   {
      // Send the log prop probability from tall nodes to root
      char buffer[SIZE_UINT2];
      int position=0;  
      MPI_Pack(&logproppr,1,MPI_DOUBLE,buffer,SIZE_UINT2,&position,MPI_COMM_WORLD);
      MPI_Send(buffer,SIZE_UINT2,MPI_PACKED,0,0,MPI_COMM_WORLD);
   }   
#endif

}


void brt::update_randz_bd(tree::tree_p nx, bool birth){
   diterator diter(di);
   if(birth){
      for(;diter<diter.until();diter++){
         if(randz_bdp[*diter]==2){
            // Right move accepted
            randz[*diter] = randz[*diter]->r;
         }else if(randz_bdp[*diter]==1){
            // Left move accepted
            randz[*diter] = randz[*diter]->l;
         }
      }
   }else{
      for(;diter<diter.until();diter++){
         // Was a left or right child
         if(randz_bdp[*diter] > 0){
            randz[*diter] = nx;
         }
      }
   }
   randz_bdp.clear();
}


//--------------------------------------------------
// For peturb and change of variables
void brt::sumlogphix_mpi(double &osum, double &nsum){
#ifdef _OPENMPI
   char buffer[SIZE_UINT3];
   int position=0;
   const int tag = MPI_TAG_SHUFFLE;
   MPI_Status status;

   if(rank>0){    
      // Sum the suff stats then get accept/reject status
      MPI_Reduce(&osum,NULL,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Reduce(&nsum,NULL,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

      // Receive the updated sums across all nodes
      MPI_Recv(buffer,SIZE_UINT3,MPI_PACKED,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
      MPI_Unpack(buffer,SIZE_UINT3,&position,&osum,1,MPI_DOUBLE,MPI_COMM_WORLD);
      MPI_Unpack(buffer,SIZE_UINT3,&position,&nsum,1,MPI_DOUBLE,MPI_COMM_WORLD);

   }           

   if(rank==0){
      // Reduce the suff stats
      MPI_Request *request = new MPI_Request[tc];
      //cout << "reduce 0...." << endl;
      MPI_Reduce(MPI_IN_PLACE,&osum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE,&nsum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

      //cout << "pack 0...." << endl;
      // Pack and send info to the slaves
      MPI_Pack(&osum,1,MPI_DOUBLE,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      MPI_Pack(&nsum,1,MPI_DOUBLE,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      //cout << "send 0...." << endl;
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(buffer,SIZE_UINT3,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
      }
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
   }
#else
   // Add as needed
#endif
   
}


void brt::getchgvsuff_rpath(tree::tree_p pertnode, tree::npv& bnv, size_t oldc, size_t oldv, bool didswap, double &osumlog, double &nsumlog){
   // Compute values
   nsumlog = sumlogphix(rpi.gamma,pertnode);
   if(didswap) pertnode->swaplr();  //undo the swap so we can calculate the suff stats for the original variable, cutpoint.
   pertnode->setv(oldv);
   pertnode->setc(oldc);
   osumlog = sumlogphix(rpi.gamma,pertnode);

   // Pass the sums across the mpi
   sumlogphix_mpi(osumlog,nsumlog);
}


void brt::getpertsuff_rpath(tree::tree_p pertnode, tree::npv& bnv, size_t oldc, double &osumlog, double &nsumlog){
   // Compute values
   nsumlog = sumlogphix(rpi.gamma,pertnode);
   pertnode->setc(oldc);
   osumlog = sumlogphix(rpi.gamma,pertnode);

   // Pass the sums across the mpi
   sumlogphix_mpi(osumlog,nsumlog);
}


//--------------------------------------------------
// For joint randz proposal
void brt::shuffle_randz(rn &gen){
   std::vector<sinfo*>& sold = newsinfovec();
   std::vector<sinfo*>& snew = newsinfovec();
   tree::npv rbnv;
   tree::tree_p root;
   double lmold, lmnew;
   bool hardreject = false;

   root = t.getptr(t.nid());

   // Get suff stats
   subsuff(root,rbnv,sold);
   subsuff(root,rbnv,snew);

   // Get lm
   for(size_t i=0;i<rbnv.size();i++)
      lmold += lm(*(sold[i]));

   for(size_t i=0;i<rbnv.size();i++) {
      if((snew[i]->n) >= mi.minperbot)
         lmnew += lm(*(snew[i]));
      else 
         hardreject=true;
   }

   // Delete suff stats
   for(size_t i=0;i<sold.size();i++) delete sold[i];
   for(size_t i=0;i<snew.size();i++) delete snew[i];
   delete &sold;
   delete &snew;

   // MH step
   double alpha1 = exp(lmnew-lmold);
   double alpha = std::min(1.0,alpha1);

   if(hardreject)
      alpha=0.0;

   // Send accept/reject
#ifdef _OPENMPI
   MPI_Request *request = new MPI_Request[tc];
#endif

   if(gen.uniform()<alpha) {
#ifdef _OPENMPI
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(NULL,0,MPI_PACKED,i,MPI_TAG_SHUFFLE_ACCEPT,MPI_COMM_WORLD,&request[i-1]);
      }
   }
   else { //transmit reject over MPI
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(NULL,0,MPI_PACKED,i,MPI_TAG_SHUFFLE_REJECT,MPI_COMM_WORLD,&request[i-1]);
      }
   }
#else
   }
#endif
}



//--------------------------------------------------
// Sample the tree prior - for rpath variogram
void brt::sample_tree_prior(rn& gen){
   std::vector<int> nidqueue;
   tree::npv goodbots; 
   double PBx = 0.5;
   double u0;
   int nid;
   size_t dnx;
   tree::tree_p nx0;

   // Init tree and splitting queue
   t.tonull();
   nidqueue.push_back(1);
 
   while(nidqueue.size()>0){
      // Remove leading element in nidqueue
      nid = nidqueue[0];
      nx0 = t.getptr(nid);
      nidqueue.erase(nidqueue.begin());
      
      //cout << "queue size = " << nidqueue.size() << endl;
      dnx = nx0->depth();

      // Sampling procedure at current node
      u0 = gen.uniform();
      if(tp.alpha/pow(1.0+dnx,tp.beta) >= u0){
         // Try to split using birth proposal (these probs don't really matter - only used here to ensure we have valid splits)
         goodbots.clear();
         tree::tree_p nx; //bottom node
         size_t v,c; //variable and cutpoint
         int L,U;
         double pr; //part of metropolis ratio from proposal and prior
         
         //cout << "nid = " << nid << endl;
         
         if(cansplit(nx0,*xi)&(nx0->depth() < (tp.maxd+1))) goodbots.push_back(nx0);
         if(goodbots.size()>0){
            L=std::numeric_limits<int>::min(); U=std::numeric_limits<int>::max();
            bprop(t,*xi,tp,mi.pb,goodbots,PBx,nx,v,c,pr,gen);            
            nx->rgi(v,&L,&U);
            if((L!=c) && (U!=c)){
               t.birthp(nx,v,c,0.0,0.0);
               // bprop gives us the new tree - need to add to queue
               nidqueue.push_back(nid*2); // left child id
               nidqueue.push_back(nid*2 + 1); // right child id
            }            
         }
      }
   }
   
   //cout << "treesize = " << t.treesize() << endl;
}



/*
//--------------------------------------------------
//peturb proposal for internal node cut points.
void brt::pertcv(rn& gen)
{
//   cout << "--------------->>into pertcv" << endl;
   tree::tree_p pertnode;
   if(t.treesize()==1) // nothing to perturb if the tree is a single terminal node
      return;

   // Get interior nodes and propose new split value
   tree::npv intnodes;
   t.getintnodes(intnodes);
   for(size_t pertdx=0;pertdx<intnodes.size();pertdx++)
   if(gen.uniform()<mi.pchgv) {
      mi.chgvproposal++;
      pertnode = intnodes[pertdx];

      //get L,U for the old variable and save it as well as oldc
      int Lo,Uo;
      getLU(pertnode,*xi,&Lo,&Uo);
      size_t oldc = pertnode->getc();

      //update correlation matrix
      bool didswap=false;
      size_t oldv=pertnode->getv();
      size_t newv;
#ifdef _OPENMPI 
      MPI_Request *request = new MPI_Request[tc];
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(NULL,0,MPI_PACKED,i,MPI_TAG_PERTCHGV,MPI_COMM_WORLD,&request[i-1]);
      }
      std::vector<double> chgvrow;
      chgvrow=(*mi.corv)[oldv]; 
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;

      mpi_update_norm_cormat(rank,tc,pertnode,*xi,chgvrow,chv_lwr,chv_upr);
      newv=getchgvfromrow(oldv,chgvrow,gen);
#else
      std::vector<std::vector<double> > chgv;
      chgv= *mi.corv; //initialize it
      updatecormat(pertnode,*xi,chgv);
      normchgvrow(oldv,chgv);
      newv=getchgv(oldv,chgv,gen);
#endif

      //choose new variable randomly
      pertnode->setv(newv);
      if((*mi.corv)[oldv][newv]<0.0) {
         pertnode->swaplr();
         didswap=true;
      }

      //get L,U for the new variable and save it and set newc
      int Ln,Un;
      getLU(pertnode,*xi,&Ln,&Un);
      size_t newc = Ln + (size_t)(floor(gen.uniform()*(Un-Ln+1.0)));
      pertnode->setc(newc);

#ifdef _OPENMPI
      unsigned int propcint=(unsigned int)newc;
      unsigned int propvint=(unsigned int)newv;
      request = new MPI_Request[tc];
      char buffer[SIZE_UINT3];
      int position=0;
      MPI_Pack(&propcint,1,MPI_UNSIGNED,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      MPI_Pack(&propvint,1,MPI_UNSIGNED,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      MPI_Pack(&didswap,1,MPI_CXX_BOOL,buffer,SIZE_UINT3,&position,MPI_COMM_WORLD);
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(buffer,SIZE_UINT3,MPI_PACKED,i,MPI_TAG_PERTCHGV,MPI_COMM_WORLD,&request[i-1]);
      }
      std::vector<double> chgvrownew;
      chgvrownew=(*mi.corv)[newv];
 
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;

      mpi_update_norm_cormat(rank,tc,pertnode,*xi,chgvrownew,chv_lwr,chv_upr);
      if(chgvrownew[oldv]==0.0)
         cout << "Proposal newv cannot return to oldv!  This is not possible!" << endl;

      double alpha0=chgvrownew[oldv]/chgvrow[newv];  //proposal ratio for newv->oldv and oldv->newv
#else
      //now we also need to update the row of chgv for newv->oldv to calc MH correctly
      updatecormat(pertnode,*xi,chgv);
      normchgvrow(newv,chgv);
      //sanity check:
      if(chgv[newv][oldv]==0.0)
         cout << "Proposal newv cannot return to oldv!  This is not possible!" << endl;
      double alpha0=chgv[newv][oldv]/chgv[oldv][newv];  //proposal ratio for newv->oldv and oldv->newv
#endif

      //get sufficient statistics and calculate lm
      std::vector<sinfo*>& sivold = newsinfovec();
      std::vector<sinfo*>& sivnew = newsinfovec();
      tree::npv bnv;
      getchgvsuff(pertnode,bnv,oldc,oldv,didswap,sivold,sivnew);

      typedef tree::npv::size_type bvsz;
      double lmold,lmnew;
      bool hardreject=false;
      lmold=0.0;
      for(bvsz j=0;j!=sivold.size();j++) {
         if(sivold[j]->n < mi.minperbot)
            cout << "Error: old tree has some bottom nodes with <minperbot observations!" << endl;
         lmold += lm(*(sivold[j]));
      }

      lmnew=0.0;
      for(bvsz j=0;j!=sivnew.size();j++) {
         if(sivnew[j]->n < mi.minperbot)
            hardreject=true;
         lmnew += lm(*(sivnew[j]));
      }
      double alpha1 = ((double)(Uo-Lo+1.0))/((double)(Un-Ln+1.0)); //from prior for cutpoints
      double alpha2=alpha0*alpha1*exp(lmnew-lmold);
      double alpha = std::min(1.0,alpha2);
      if(hardreject) alpha=0.0;  //change of variable led to an bottom node with <minperbot observations in it, we reject this.
#ifdef _OPENMPI
      request = new MPI_Request[tc];
#endif

      if(gen.uniform()<alpha) {
         mi.chgvaccept++;
         if(didswap) pertnode->swaplr();  //because the call to getchgvsuff unswaped if they were swapped
         pertnode->setv(newv); //because the call to getchgvsuff changes it back to oldv to calc the old lil
         pertnode->setc(newc); //because the call to getchgvsuff changes it back to oldc to calc the old lil
#ifdef _OPENMPI
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,MPI_TAG_PERTCHGV_ACCEPT,MPI_COMM_WORLD,&request[i-1]);
         }
      }
      else { //transmit reject over MPI
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,MPI_TAG_PERTCHGV_REJECT,MPI_COMM_WORLD,&request[i-1]);
         }
      }
#else
      }
#endif
      //else nothing, pertnode->c and pertnode->v is already reset to the old values and if a swap was done in the 
      //proposal it was already undone by getchgvsuff.
      for(bvsz j=0;j<sivold.size();j++) delete sivold[j];
      for(bvsz j=0;j<sivnew.size();j++) delete sivnew[j];
      delete &sivold;
      delete &sivnew;
#ifdef _OPENMPI
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
#endif
   }
   else {
      mi.pertproposal++;
      pertnode = intnodes[pertdx];

      // Get allowable range for perturbing cv at pertnode
      int L,U;
      bool hardreject=false;
      getLU(pertnode,*xi,&L,&U);
      size_t oldc = pertnode->getc();
      int ai,bi,oldai,oldbi;
      ai=(int)(floor(oldc-mi.pertalpha*(U-L+1)/2.0));
      bi=(int)(floor(oldc+mi.pertalpha*(U-L+1)/2.0));
      ai=std::max(ai,L);
      bi=std::min(bi,U);
      size_t propc = ai + (size_t)(floor(gen.uniform()*(bi-ai+1.0)));
      pertnode->setc(propc);
#ifdef _OPENMPI
         unsigned int propcint=(unsigned int)propc;
         MPI_Request *request = new MPI_Request[tc];
         char buffer[SIZE_UINT1];
         int position=0;
         MPI_Pack(&propcint,1,MPI_UNSIGNED,buffer,SIZE_UINT1,&position,MPI_COMM_WORLD);
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(buffer,SIZE_UINT1,MPI_PACKED,i,MPI_TAG_PERTCV,MPI_COMM_WORLD,&request[i-1]);
         }
#endif
      oldai=(int)(floor(propc-mi.pertalpha*(U-L+1)/2.0));
      oldbi=(int)(floor(propc+mi.pertalpha*(U-L+1)/2.0));
      oldai=std::max(oldai,L);
      oldbi=std::min(oldbi,U);

      std::vector<sinfo*>& sivold = newsinfovec();
      std::vector<sinfo*>& sivnew = newsinfovec();

      tree::npv bnv;
#ifdef _OPENMPI
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
#endif
      getpertsuff(pertnode,bnv,oldc,sivold,sivnew);

      typedef tree::npv::size_type bvsz;
      double lmold,lmnew;
      lmold=0.0;
      for(bvsz j=0;j!=sivold.size();j++) {
         if(sivold[j]->n < mi.minperbot)
            cout << "Error: old tree has some bottom nodes with <minperbot observations!" << endl;
         lmold += lm(*(sivold[j]));
      }

      lmnew=0.0;
      for(bvsz j=0;j!=sivnew.size();j++) {
         if(sivnew[j]->n < mi.minperbot)
            hardreject=true;
         lmnew += lm(*(sivnew[j]));
      }
      double alpha1 = ((double)(bi-ai+1.0))/((double)(oldbi-oldai+1.0)); //anything from the prior?
      double alpha2=alpha1*exp(lmnew-lmold);
      double alpha = std::min(1.0,alpha2);
#ifdef _OPENMPI
      request = new MPI_Request[tc];
#endif
      if(hardreject) alpha=0.0;  //perturb led to an bottom node with <minperbot observations in it, we reject this.

      if(gen.uniform()<alpha) {
         mi.pertaccept++;
         pertnode->setc(propc); //because the call to getpertsuff changes it back to oldc to calc the old lil.
#ifdef _OPENMPI
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,MPI_TAG_PERTCV_ACCEPT,MPI_COMM_WORLD,&request[i-1]);
         }
      }
      else { //transmit reject over MPI
         for(size_t i=1; i<=(size_t)tc; i++) {
            MPI_Isend(NULL,0,MPI_PACKED,i,MPI_TAG_PERTCV_REJECT,MPI_COMM_WORLD,&request[i-1]);
         }
      }
#else
      }
#endif
      //else nothing, pertnode->c is already reset to the old value.
      for(bvsz j=0;j<sivold.size();j++) delete sivold[j];
      for(bvsz j=0;j<sivnew.size();j++) delete sivnew[j];
      delete &sivold;
      delete &sivnew;
#ifdef _OPENMPI
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      delete[] request;
#endif
   }
}

//--------------------------------------------------
//do a rotation proposal at a randomly selected internal node.
bool brt::rot(tree::tree_p tnew, tree& x, rn& gen)
{
//   cout << "--------------->>into rot" << endl;
   #ifdef _OPENMPI
   MPI_Request *request = new MPI_Request[tc];
   if(rank==0) {
      const int tag=MPI_TAG_ROTATE;
      for(size_t i=1; i<=(size_t)tc; i++) {
         MPI_Isend(NULL,0,MPI_PACKED,i,tag,MPI_COMM_WORLD,&request[i-1]);
      }
   }
   #endif

   tree::tree_p rotp,temp;
   tree::tree_cp xp;
   tree::npv subtold, subtnew, nbold, nbnew;
   double Qold_to_new, Qnew_to_old;
   unsigned int rdx=0;
   bool twowaystoinvert=false;
   double prinew=1.0,priold=1.0;
   size_t rotid;
   bool hardreject=false;
   std::vector<size_t> goodvars; //variables an internal node can split on

   mi.rotproposal++;

   // Get rot nodes
   tree::npv rnodes;
   tnew->getrotnodes(rnodes);
   #ifdef _OPENMPI
   if(rank==0) {
      MPI_Waitall(tc,request,MPI_STATUSES_IGNORE);
      mpi_resetrn(gen);
   }
   delete[] request;
   #endif
   if(rnodes.size()==0)  return false;  //no rot nodes so that's a reject.

   rdx = (unsigned int)floor(gen.uniform()*rnodes.size()); //which rotatable node will we rotate at?
   rotp = rnodes[rdx];
   rotid=rotp->nid();
   xp=x.getptr(rotid);

//   Can check the funcitonality of getpathtoroot:  
//   tree::npv path;
//   rotp->getpathtoroot(path);
//   cout << "rot id=" << rotid << endl;
//   tnew->pr();
//   for(size_t i=0;i<path.size();i++)
//      cout << "i=" << i << ", node id=" << path[i]->nid() << endl;

   int nwaysm1=0,nwaysm2=0,nwayss1=0,nwayss2=0;
   double pm1=1.0,pm2=1.0,ps1=1.0,ps2=1.0;
   if(rotp->isleft()) {
      if(rotp->v==rotp->p->v) //special case, faster to handle it direclty
      {
         rotright(rotp);
         rotp=tnew->getptr(rotid);
         delete rotp->r;
         temp=rotp->l;
         rotp->p->l=temp;
         temp->p=rotp->p;
         rotp->r=0;
         rotp->l=0;
         rotp->p=0;
         delete rotp;
         rotp=tnew->getptr(rotid);
         //pm1=pm2=ps1=ps2=1.0 in this case
      }
      else
      {
         rotright(rotp);
         rotp=tnew->getptr(rotid); //just in case the above changed the pointer.
         reduceleft(rotp,rotp->p->v,rotp->p->c);
         rotp=tnew->getptr(rotid); //just in case the above changed the pointer.
         reduceright(rotp->p->r,rotp->p->v,rotp->p->c);
         rotp=tnew->getptr(rotid); //just in case the above changed the pointer.
         splitleft(rotp->r,rotp->p->v,rotp->p->c);
         splitright(rotp->p->r->r,rotp->p->v,rotp->p->c);

         mergecount(rotp->r,rotp->p->r->r,rotp->p->v,rotp->p->c,&nwayss1);
         ps1=1.0/nwayss1;

         mergecount(rotp->l,rotp->p->r->l,rotp->p->v,rotp->p->c,&nwayss2);
         ps2=1.0/nwayss2;

         tree::tree_p tmerged=new tree;
         tmerged->p=rotp->p;

         mergecount(rotp->p->r->l,rotp->p->r->r,rotp->p->r->v,rotp->p->r->c,&nwaysm1);
         pm1=1.0/nwaysm1;
         merge(rotp->p->r->l,rotp->p->r->r,tmerged,rotp->p->r->v,rotp->p->r->c,gen);
         rotp->p->r->p=0;
         delete rotp->p->r;
         rotp->p->r=tmerged;

         tmerged=new tree;
         tmerged->p=rotp->p;

         mergecount(rotp->l,rotp->r,rotp->v,rotp->c,&nwaysm2);
         pm2=1.0/nwaysm2;
         size_t v,c;
         v=rotp->v;
         c=rotp->c;
         merge(rotp->l,rotp->r,tmerged,rotp->v,rotp->c,gen);
         rotp->p->l=tmerged;
         rotp->p=0;
         delete rotp;
         rotp=tnew->getptr(rotid);

      //end of merge code if rotp isleft.
      //there are some "extra" isleaf's here because we don't explicitly reset v,c if node becomes leaf so we need to check.
         if( !isleaf(rotp) && !isleaf(rotp->p->r) && (rotp->v!=v && rotp->c!=c) && (rotp->p->r->v != v && rotp->p->r->c != c))
            hardreject=true;
         if( isleaf(rotp) && isleaf(rotp->p->r))
            hardreject=true;
         if(rotp->p->r->v==rotp->v && rotp->p->r->c==rotp->c && !isleaf(rotp->p->r) && !isleaf(rotp))
            twowaystoinvert=true;

      }
   }
   else { //isright
      if(rotp->v==rotp->p->v) //special case, faster to handle it directly
      {
         rotleft(rotp);
         rotp=tnew->getptr(rotid);
         delete rotp->l;
         temp=rotp->r;
         rotp->p->r=temp;
         temp->p=rotp->p;
         rotp->r=0;
         rotp->l=0;
         rotp->p=0;
         delete rotp;
         rotp=tnew->getptr(rotid);
         //pm1=pm2=ps1=ps2=1.0 in this case
      }
      else
      {
         rotleft(rotp);
         rotp=tnew->getptr(rotid); //just in case the above changed the pointer.
         reduceleft(rotp->p->l,rotp->p->v,rotp->p->c);
         rotp=tnew->getptr(rotid); //just in case the above changed the pointer.
         reduceright(rotp,rotp->p->v,rotp->p->c);
         rotp=tnew->getptr(rotid); //just in case the above changed the pointer.
         splitleft(rotp->p->l->l,rotp->p->v,rotp->p->c);
         splitright(rotp->l,rotp->p->v,rotp->p->c);

         mergecount(rotp->p->l->l,rotp->l,rotp->p->v,rotp->p->c,&nwayss1);
         ps1=1.0/nwayss1;

         mergecount(rotp->p->l->r,rotp->r,rotp->p->v,rotp->p->c,&nwayss2);
         ps2=1.0/nwayss2;

         tree::tree_p tmerged=new tree;
         tmerged->p=rotp->p;

         mergecount(rotp->p->l->l,rotp->p->l->r,rotp->p->l->v,rotp->p->l->c,&nwaysm1);
         pm1=1.0/nwaysm1;
         merge(rotp->p->l->l,rotp->p->l->r,tmerged,rotp->p->l->v,rotp->p->l->c,gen);
         rotp->p->l->p=0;
         delete rotp->p->l;
         rotp->p->l=tmerged;

         tmerged=new tree;
         tmerged->p=rotp->p;

         mergecount(rotp->l,rotp->r,rotp->v,rotp->c,&nwaysm2);
         pm2=1.0/nwaysm2;
         size_t v,c;
         v=rotp->v;
         c=rotp->c;
         merge(rotp->l,rotp->r,tmerged,rotp->v,rotp->c,gen);
         rotp->p->r=tmerged;
         rotp->p=0;
         delete rotp;
         rotp=tnew->getptr(rotid);

      //end of merge code if rotp isright
      //there are some "extra" isleaf's here because we don't explicitly reset v,c if node becomes leaf so we need to check.
         if( !isleaf(rotp) && !isleaf(rotp->p->l) && (rotp->v!=v && rotp->c!=c) && (rotp->p->l->v != v && rotp->p->l->c != c))
            hardreject=true;
         if( isleaf(rotp) && isleaf(rotp->p->l))
            hardreject=true;
         if(rotp->p->l->v==rotp->v && rotp->p->l->c==rotp->c && !isleaf(rotp->p->l) && !isleaf(rotp))
            twowaystoinvert=true;
      }
   }

   // Calculate prior probabilities, we just need to use the subtree where the rotation occured of tnew and x.
   subtold.clear();
   subtnew.clear();
   xp->p->getnodes(subtold);
   rotp->p->getnodes(subtnew);

   for(size_t i=0;i<subtold.size();i++) {
      if(subtold[i]->l) { //interior node
         priold*=tp.alpha/pow(1.0 + subtold[i]->depth(),tp.beta);
         goodvars.clear();
         getinternalvars(subtold[i],*xi,goodvars);
         priold*=1.0/((double)goodvars.size()); //prob split on v 
         priold*=1.0/((double)getnumcuts(subtold[i],*xi,subtold[i]->v)); //prob split on v at c is 1/numcutpoints
      }
      else //terminal node
         priold*=(1.0-tp.alpha/pow(1.0 + subtold[i]->depth(),tp.beta)); 
   }
   for(size_t i=0;i<subtnew.size();i++) {
      if(subtnew[i]->l) { //interior node
         prinew*=tp.alpha/pow(1.0 + subtnew[i]->depth(),tp.beta);
         goodvars.clear();
         getinternalvars(subtnew[i],*xi,goodvars);
         prinew*=1.0/((double)goodvars.size()); //prob split on v
         prinew*=1.0/((double)getnumcuts(subtnew[i],*xi,subtnew[i]->v)); //prob split on v at c is 1/numcutpoints
         if(getnumcuts(subtnew[i],*xi,subtnew[i]->v)<1)
         {
            x.pr(true);
            tnew->pr(true);
         }
      }
      else //terminal node
         prinew*=(1.0-tp.alpha/pow(1.0 + subtnew[i]->depth(),tp.beta)); 
   }

   Qold_to_new=1.0/((double)rnodes.size()); //proposal probability of rotating from x to tnew
   
   rnodes.clear();
   tnew->getrotnodes(rnodes);  //this is very inefficient, could make it much nicer later on.
//   if(rnodes.size()==0) hardreject=true; //if we're back down to just a root node we can't transition back, so this is a hard reject.

   if(!twowaystoinvert)
      Qnew_to_old=1.0/((double)rnodes.size()); //proposal probability of rotating from tnew back to x
   else
      Qnew_to_old=2.0/((double)rnodes.size());

   // Calculate log integrated likelihoods for the subtree where the rotation occured of tnew and x.
   double lmold=0.0,lmnew=0.0;
//   std::vector<sinfo> sold,snew;
   std::vector<sinfo*>& sold = newsinfovec();
   std::vector<sinfo*>& snew = newsinfovec();
   nbold.clear();
   nbnew.clear();
   sold.clear();
   snew.clear();
   x.getbots(nbold);
   tnew->getbots(nbnew);

   //get sufficient statistics for subtree involved in rotation
   //which is just everything below rotp->p.
   //Use subsuff here, which will get the suff stats for both the
   //orignal tree and the proposed tree without needed to explicitly
   //know the root node of either tree as this is recovered when
   //finding the path to rotp->p within the subsuff method.
   rotp=x.getptr(rotid);
   subsuff(rotp->p,nbold,sold);
   rotp=tnew->getptr(rotid);
   subsuff(rotp->p,nbnew,snew);

   for(size_t i=0;i<nbold.size();i++)
         lmold += lm(*(sold[i]));

   for(size_t i=0;i<nbnew.size();i++) {
      if( (snew[i]->n) >= mi.minperbot )
         lmnew += lm(*(snew[i]));
      else 
         hardreject=true;
   }

   for(size_t i=0;i<sold.size();i++) delete sold[i];
   for(size_t i=0;i<snew.size();i++) delete snew[i];
   delete &sold;
   delete &snew;

   double alpha1;
   alpha1=prinew*Qnew_to_old/priold/Qold_to_new/pm1/pm2*ps1*ps2;
   double alpha2 = alpha1*exp(lmnew-lmold);
   double alpha = std::min(1.0,alpha2);

   if(hardreject)
      alpha=0.0;

   if(gen.uniform()<alpha) {
      mi.rotaccept++;
      x = *tnew;
      return true;
   }
   else {
      return false;
   }

   return false;  // we never actually get here.
}

*/