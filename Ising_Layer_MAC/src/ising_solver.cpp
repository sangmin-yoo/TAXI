#include "mylib.h"
#include "ising_solver.h"
#include "cost_function.h"
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <bits/stdc++.h>

#include <algorithm> 
#include <iostream>

using namespace std;

Edge::Edge(int to, Weight weight) : to(to), weight(weight) {}

//cf is costfunction
IsingSolver::IsingSolver(const CostFunction& cf, const int size_opt, const std::vector<double> Wd, const std::vector<double> Wd_Paras, const std::vector<double> WdG, const std::vector<double> WdG_Paras, std::vector<double> KeepH)
  : steps(0), total_step(0),
    random_selector(size_opt), active_ratio(0), cf(cf),
    Wd(Wd), Wd_Paras(Wd_Paras), WdG(WdG), WdG_Paras(WdG_Paras), H_Optim(INFINITY), H_curr(0), KeepH(KeepH) {}
    
int IsingSolver::calcTotalStep(double initial_active_ratio, double init_Irand) const {
  //double n = - (log(initial_active_ratio) + log(size_opt())) / log(cool_coe);
  double n = (init_Irand-Istop)/cool_coe;
  assert(n >= 0);
  return int(ceil(n));
}
void IsingSolver::init(const IsingSolver::InitMode mode, const int seed, const double cool_coe, const double initial_active_ratio, const int seq_clust,
  const double VDD, const double RonTr, const double RoffTr, const double RonArr, const double RoffArr, const double Rw, const int BitPrec, const double Factor, const double Threshold, const int Patience, double init_Irand) {
  assert(0 <= cool_coe && cool_coe < 1);
  rnd.seed(seed);
  this->cool_coe = cool_coe;
  this->steps = 0;
  this->active_ratio = initial_active_ratio;
  this->seq_clust = seq_clust;
  
  this->VDD = VDD;
  this->RonTr = RonTr;
  this->RoffTr = RoffTr;
  this->BitPrec = BitPrec;
  
  this->dRtr = RoffTr - RonTr;
  
  this->RonArr = RonArr;
  this->RoffArr = RoffArr;
  this->Rw = Rw;

  this->Factor = Factor;
  this->Threshold = Threshold;
  this->Patience = Patience;

  nMAC = 0;
  nRAND = 0;
  PlateauCNT = 0;
  OptimCNT = 0;
  Imid = init_Irand;
  Icool = cool_coe;
  //Stop Ising Solver when stochastic probability reaches 1%
  Istop = -log(100/1-1)*21+450;
  //Stop Ising Solver when stochastic probability reaches 2%
  //Istop = -log(100/2-1)*21+450;

  this->init_Irand = init_Irand;
  this->total_step = calcTotalStep(initial_active_ratio, init_Irand);

  switch (mode) {
    case Negative:
      current_spin.assign(size(), -1);
      break;
    case Positive:
      current_spin.assign(size(), 1);
      break;
    case Random:
      current_spin.assign(size(), -1);
      break;
  }
  random_device rd;
  mt19937 g(rd());
  std::vector<int> shuffled;
  switch (seq_clust) {
    case 0: // The topmost cluster
      rep(i, map_size()) {
        OptSpins.push_back(i);
      }
      break;
    case 1: // The first sub-cluster
      rep(i, 1, map_size()){
        current_spin[size()-1-i] = 0;//-1;
        current_spin[size()-1-i*map_size()] = 0;//-1;
        OptSpins.push_back(map_size()-1-i);//Cities to Optimize. The N-th city is fixed as the last city to visit. ** The order is flipped. 
      }
      current_spin[size()-1] = 1;
      break;
    case -1: // The last sub-cluster
      rep(i, 1, map_size()){
        current_spin[i] = 0;//-1;
        current_spin[i*map_size()] = 0;//-1;
        OptSpins.push_back(i);//Cities can be optimized. The 1st city is fixed as the first city to visit.
      }
      current_spin[0] = 1;
      break;
    default: // The rest of sub-clusters
      rep(i, 1, map_size()){
        current_spin[i] = 0;//-1;
        current_spin[i*map_size()] = 0;//-1;
        current_spin[size()-1-i] = 0;//-1;
        current_spin[size()-1-i*map_size()] = 0;//-1;
        if (i < map_size()-1) {
          OptSpins.push_back(i);//Cities can be optimized. The first/N-th city is fixed as the first/last city to visit.
        }
      }
      current_spin[0] = 1;
      current_spin[size()-1] = 1;
      break;
  }
  shuffled = OptSpins;
  shuffle(shuffled.begin(), shuffled.end(), g);
  rep(i,0, OptSpins.size()) {
    current_spin[OptSpins[i]*map_size()+shuffled[i]] = 1;
  }
  /*
  if (seq_clust!=0) {
    rep (i, 0, map_size()){
      rep (j, 0, map_size()){
        cout << current_spin[i*map_size()+j];
      }
      cout << '\n';
    }
    cout << "##############" << '\n';
  }
  rep(b,BitPrec) {
    cout << b << "bit \n";
    rep(i,map_size()) {
      rep(j,map_size()) {
        cout << Wd[b*size() + i*map_size() + j] << " ";
      }
      cout << "\n";
    }
    cout << "##############" << '\n';
  }*/
  optimal_spin = current_spin;
}
void IsingSolver::init(const IsingSolver::InitMode mode, const double cool_coe, const double initial_active_ratio, const int seq_clust,
  const double VDD, const double RonTr, const double RoffTr, const double RonArr, const double RoffArr, const double Rw, const int BitPrec, const double Factor, const double Threshold, const int Patience, double init_Irand) {
  random_device rd;
  init(mode, rd(), cool_coe, initial_active_ratio, seq_clust, VDD, RonTr, RoffTr, RonArr, Rw, RoffArr, BitPrec, Factor, Threshold, Patience, init_Irand);
}
void IsingSolver::HminMAC() {
  int erase_spin;
  std::vector<int> AvailableSpins;
  for (int spin : OptSpins){
    AvailableSpins.push_back(spin);
  }
  for (int Order: OptSpins) {
    if (AvailableSpins.size() == 1) {
      rep (i, map_size()) {
        current_spin[Order*map_size()+i] = 0;//-1;
      }
      current_spin[Order*map_size()+AvailableSpins[0]] = 1;
      break;
    }
    erase_spin = MAC(Order, AvailableSpins);
    AvailableSpins.erase(AvailableSpins.begin()+erase_spin);
  }
}
int IsingSolver::MAC(const int Rid, std::vector<int> AvailableSpins) {
  // Rid & Cid : Row (visitng order) & Column (city) index
  // Rid Should be between 1 and map_size()-2
  double Iout = 0;
  double Iparas = 0;
  double MAX = 0;
  int Cid = 0;
  int Seq_prev = (Rid-1)*map_size();
  int Seq_curr = Rid * map_size();
  int Seq_next = (Rid+1)*map_size();

  int erase_id = 0;
  int for_id;

  for_id = 0;
  MAX = 0;

  double Iout_rand = 0;
  double MAX_rand = 0;
  int Cid_rand = 0;
  int erase_id_rand = 0;

  //Update input by a MAC operation using the spin storage.
  std::vector<int> MACinput;
  int new_spin;
  rep(i, map_size()) {
    new_spin = 0;
    if (Rid > 0) new_spin += current_spin[Seq_prev+i];
    if (Rid < map_size()-1) new_spin += current_spin[Seq_next+i];
    MACinput.push_back(new_spin);
  }
  //Ising-Macro
  for (int col : AvailableSpins){
    rep(B, BitPrec) rep(i, map_size()){
      if (MACinput[i] == 1) Iout += Wd[B*size()+i*map_size() + col];
      //Parastic current derived by Down-Spins.
      else Iparas += Wd_Paras[B*size()+i*map_size() + col];
    }
    Iout += Iparas;
    //Add current from Spin-devices that are stochastically switched. By the Digital circuit.
    if (rand()%100 < 100/(1+exp(-(Imid-450)*0.047619))) Iout_rand = Iout;
    else Iout_rand = 0;
    if (MAX_rand<Iout_rand) {
      MAX_rand = Iout_rand;
      Cid_rand = col;
      erase_id_rand = for_id;
    }
    if (MAX < Iout) {
      MAX = Iout;
      Cid = col;
      erase_id = for_id;
    }
    ++for_id;
    Iout = 0;
    Iparas = 0;
  }
  ++nMAC;
  
  if (MAX_rand > 0) {
    Cid = Cid_rand;
    erase_id = erase_id_rand;
  }
  //Reset Spin States
  rep (i, map_size()) {
    current_spin[Seq_curr+i] = 0;
  }
  //Program a new Spin State
  current_spin[Seq_curr+Cid] = 1;
  return erase_id;
}
double IsingSolver::calcHbyMAC() {
  double Iout = 0;
  rep(order, map_size()-1) {
    rep (col, map_size()) rep(i, map_size()){
      if (current_spin[order*map_size() + i] == 1) {
        if (current_spin[(order+1)*map_size() + col] == 1) {
          Iout += WdG[i*map_size() + col];
        }
      }
    }
    /*rep(B, BitPrec) rep (col, map_size()) rep(i, map_size()){
      if (current_spin[order*map_size() + i] == 1) {
        if (current_spin[(order+1)*map_size() + col] == 1) {
          Iout += WdG[B*size() + i*map_size() + col];
        }
        else Iout += WdG_Paras[B*size() + i*map_size() + col];
      }
      else Iout += WdG_Paras[B*size() + i*map_size() + col];
    }*/
  }
  return Iout;
}
void IsingSolver::cool() {
  Imid -= Icool;
}
void IsingSolver::updateOptimalSpin() {
  H_curr = calcHbyMAC();
  if (H_curr < H_Optim) {
    H_Optim = H_curr;
    optimal_spin = current_spin;
    OptimCNT = 0;
  }
  else ++OptimCNT;
}
void IsingSolver::step() {
  cool();
  if (current_spin.size() != size()) {
    throw new runtime_error("call init method first");
  }
  HminMAC();
  updateOptimalSpin();
  ++steps;
}
size_t IsingSolver::getActiveNodeCount() const {
  return size_t(floor(size_opt() * active_ratio));
}
size_t IsingSolver::size() const {
  return cf.size();
}
int IsingSolver::map_size() const {
  return sqrt(double(cf.size()));
}
int IsingSolver::opt_size() const {
  switch (seq_clust) {
    case 0: // The topmost cluster
      return map_size();
    case 1: // The first sub-cluster
      return map_size()-1;
    case -1: // The last sub-cluster
      return map_size()-1;
    default: // The rest of sub-clusters
      return map_size()-2;
      break;
  }
}
int IsingSolver::size_opt() const {
  return pow(opt_size(),2);
}
Weight IsingSolver::calcEnergy(const std::vector<int>& spin) const {
  return cf.calcEnergy(getCurrentPer(), spin);
}
Weight IsingSolver::getCurrentEnergy() const {
  return H_curr;
}
Weight IsingSolver::getOptimalEnergy() const {
  return H_Optim;
}
int IsingSolver::getNumberMAC() const {
  return nMAC;
}
int IsingSolver::getNumberRandFlip() const {
  return nRAND;
}
const vector<int>& IsingSolver::SpinsToOptimize() const{
  return OptSpins;
}
const vector<int>& IsingSolver::getCurrentSpin() const {
  return current_spin;
}
const vector<int>& IsingSolver::getOptimalSpin() const {
  return optimal_spin;
}
void IsingSolver::setCurrentSpin(const vector<int>& new_spin) {
  assert(current_spin.size() == new_spin.size());
  current_spin = new_spin;
}
int IsingSolver::getStep() const {
  return steps;
}
int IsingSolver::getTotalStep() const {
  return total_step;
}
double IsingSolver::getCurrentPer() const {
  return min(1.0, double(getStep()) / getTotalStep());
}
