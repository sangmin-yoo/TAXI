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
IsingSolver::IsingSolver(const CostFunction& cf, const int size_opt, const std::vector<double> Wd, const std::vector<double> Wd_Paras)
  : steps(0), total_step(0),
    random_selector(size_opt), active_ratio(0), cf(cf),
    Wd(Wd), Wd_Paras(Wd_Paras), Hmax(0), H_B(0) {}
    
int IsingSolver::calcTotalStep(double initial_active_ratio, double init_Irand) const {
  // return n s.t. initial_active_ratio * size() * cool_coe^n < 1
  //double n = - (log(initial_active_ratio) + log(size())) / log(cool_coe);
  //double n = - (log(initial_active_ratio) + log(size_opt())) / log(cool_coe);
  //double n = - 6800*(log(initial_active_ratio) + log(size_opt())) / log(cool_coe);
  double n = (init_Irand-Istop)/cool_coe;
  assert(n >= 0);
  return int(ceil(n));
}
void IsingSolver::init(const IsingSolver::InitMode mode, const int seed, const double cool_coe, const double update_ratio, const double initial_active_ratio, const int seq_clust,
  const double VDD, const double RonTr, const double RoffTr, const double RonArr, const double RoffArr, const int BitPrec, const double Factor, const double Threshold, const int Patience, double init_Irand) {
  assert(0 <= cool_coe && cool_coe < 1);
  assert(0 <= update_ratio && update_ratio <= 1);
  rnd.seed(seed);
  this->cool_coe = cool_coe;
  this->update_ratio = update_ratio;
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

  this->Factor = Factor;
  this->Threshold = Threshold;
  this->Patience = Patience;

  nMAC = 0;
  nRAND = 0;
  PlateauCNT = 0;
  Imid = init_Irand;
  Icool = cool_coe;
  Istop = -log(100/1-1)+50;

  this->total_step = calcTotalStep(initial_active_ratio, init_Irand);

  switch (mode) {
    case Negative:
      current_spin.assign(size(), -1);
      break;
    case Positive:
      current_spin.assign(size(), 1);
      break;
    case Random:
      /*current_spin.resize(size());
      for (auto&& s : current_spin) {
        s = rnd() % 2 ? 1 : -1;
      }*/
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
        current_spin[size()-1-i] = -1;
        current_spin[size()-1-i*map_size()] = -1;
        OptSpins.push_back(map_size()-1-i);//Cities to Optimize. The N-th city is fixed as the last city to visit. ** The order is flipped. 
      }
      current_spin[size()-1] = 1;
      break;
    case -1: // The last sub-cluster
      rep(i, 1, map_size()){
        current_spin[i] = -1;
        current_spin[i*map_size()] = -1;
        OptSpins.push_back(i);//Cities can be optimized. The 1st city is fixed as the first city to visit.
      }
      current_spin[0] = 1;
      break;
    default: // The rest of sub-clusters
      rep(i, 1, map_size()){
        current_spin[i] = -1;
        current_spin[i*map_size()] = -1;
        current_spin[size()-1-i] = -1;
        current_spin[size()-1-i*map_size()] = -1;
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
void IsingSolver::init(const IsingSolver::InitMode mode, const double cool_coe, const double update_ratio, const double initial_active_ratio, const int seq_clust,
  const double VDD, const double RonTr, const double RoffTr, const double RonArr, const double RoffArr, const int BitPrec, const double Factor, const double Threshold, const int Patience, double init_Irand) {
  random_device rd;
  init(mode, rd(), cool_coe, update_ratio, initial_active_ratio, seq_clust, VDD, RonTr, RoffTr, RonArr, RoffArr, BitPrec, Factor, Threshold, Patience, init_Irand);
}
void IsingSolver::HminMAC_wo_Randomness_parallel() {
  bool forward_path;
  forward_path = (OptSpins[0] == 1)? true : false;
  std::vector<int> prev_spin;
  prev_spin = current_spin;

  std::vector<int> odd_order;
  std::vector<int> even_order;
  for (int Order: OptSpins) {
    if (Order%2==0) even_order.push_back(Order);
    if (Order%2==1) odd_order.push_back(Order);
  }
  std::vector<int> AvailableSpins;
  for (int spin : OptSpins){
    AvailableSpins.push_back(spin);
  }
  std::vector<int> DDSpins;

  DDSpins = MAC_wo_Randomness_parallel(even_order, AvailableSpins, forward_path);
  std::vector<int> AvailableSpins2;
  rep (i, OptSpins.size()){
    int cnt = count(DDSpins.begin(), DDSpins.end(), i);
    if (cnt == 0) AvailableSpins2.push_back(OptSpins[i]);
  }
  DDSpins = MAC_wo_Randomness_parallel(odd_order, AvailableSpins2, forward_path);
}
void IsingSolver::HminMAC_wo_Randomness() {
  std::vector<int> AvailableSpins;
  for (int spin : OptSpins){
    AvailableSpins.push_back(spin);
  }
  bool forward_path = (OptSpins[0] <= 1)? true : false;
  //std::vector<int> prev_spin;
  //prev_spin = current_spin;
  //H_B = 0;
  int erase_spin;
  for (int Order: OptSpins) {
    if (AvailableSpins.size() == 1) {
      //SwapSpinsViolated(Order*map_size(), AvailableSpins[0]);
      rep (i, map_size()) {
        current_spin[Order*map_size()+i] = -1;
      }
      current_spin[Order*map_size()+AvailableSpins[0]] = 1;
      break;
    }
    erase_spin = MAC_wo_Randomness(Order, AvailableSpins, forward_path);
    //erase_spin = MAC_wo_Randomness(Order, AvailableSpins);
    AvailableSpins.erase(AvailableSpins.begin()+erase_spin);
    /*if (getCurrentEnergy() >= H_B_prev) {//retract
    
      current_spin = prev_spin;
    }
    else {//accept
      H_B_prev = getCurrentEnergy();
      prev_spin = current_spin;
      AvailableSpins.erase(AvailableSpins.begin()+erase_spin);
    }*/
  }
}
void IsingSolver::HminMAC() {
  std::vector<int> AvailableSpins;

  for (int spin : OptSpins){
    AvailableSpins.push_back(spin);
  }
  
  bool forward_path;
  forward_path = (OptSpins[0] == 1)? true : false;
  double H_B_prev;
  //H_B_prev = H_B;
  H_B_prev = getCurrentEnergy();
  std::vector<int> prev_spin;
  prev_spin = current_spin;
  //H_B = 0;
  int erase_spin;

  for (int Order: OptSpins) {
    if (AvailableSpins.size() == 1) {
      rep (i, map_size()) {
        current_spin[Order*map_size()+i] = -1;
      }
      current_spin[Order*map_size()+AvailableSpins[0]] = 1;
      break;
    }
    erase_spin = MAC(Order, AvailableSpins, forward_path);
    AvailableSpins.erase(AvailableSpins.begin()+erase_spin);
  }
}
std::vector<int> IsingSolver::MAC_wo_Randomness_parallel(std::vector<int> Rids, std::vector<int> AvailableSpins, const bool forward_path) {
  // Rid & Cid : Row (visitng order) & Column (city) index
  //Rid Should be between 1 and map_size()-2
  //double Iout = 0
  double Iprev = 0;
  double Inext = 0;
  double iout;
  int Seq_prev;
  int Seq_next;
  int Seq_curr;

  std::vector<double> Iout;//(Rids.size()*AvailableSpins.size());
  for (int Rid: Rids) {
    Seq_prev = (Rid-1)*map_size();
    Seq_next = (Rid+1)*map_size();
    Seq_curr = Rid*map_size();
    for (int col : AvailableSpins){
      rep(B, BitPrec) rep(i, map_size()){
        if (Rid > 0) Iprev += (((current_spin[Seq_prev + i]+1)/2) * Wd[B*size()+i*map_size() + col]);
        if (Rid < map_size()-1) Inext += (((current_spin[Seq_next + i]+1)/2) * Wd[B*size()+i*map_size() + col]);
      }
      if (Rid == 0) {
        iout = 2*Inext;
      }
      else if (Rid == map_size()-1) {
        iout = 2*Iprev;
      }
      else iout = Iprev+Inext;
      if (current_spin[Seq_curr+col] != 1) iout = iout - Isub();
      Iout.push_back(iout);
      //Iout.push_back(Iprev+Inext);
      Iprev = 0;
      Inext = 0;
    }
    ++nMAC;
  }
  double MAX = 0;
  int Cid=0;
  int Rid=0;
  std::vector<int> DeleteSpins;
  rep(TT, Rids.size()) {
    rep(i, Rids.size()) rep(j, AvailableSpins.size()) {
      if (Iout[i*AvailableSpins.size()+j] > MAX) {
        MAX = Iout[i*AvailableSpins.size()+j];
        Rid = i;
        Cid = j;
      }
    }
    int count = 0;
    if (DeleteSpins.size() > 0) {
      for (int CHK: DeleteSpins) {
        if (CHK == Cid) ++count;
      }
    }
    if (count == 0) DeleteSpins.push_back(Cid);

    rep (j, AvailableSpins.size()) {
      Iout[Rid*AvailableSpins.size()+j] = 0;
    }
    rep (i, Rids.size()) {
      Iout[i*AvailableSpins.size()+Cid] = 0;
    }
    MAX = 0;
    //SwapSpinsViolated(Rids[Rid]*map_size(), Cid);
    rep (i, map_size()) {
      current_spin[Rids[Rid]*map_size()+i] = -1;
    }
    current_spin[Rids[Rid]*map_size()+AvailableSpins[Cid]] = 1;
  }
  return DeleteSpins;
}
int IsingSolver::MAC_wo_Randomness(const int Rid, std::vector<int> AvailableSpins, const bool forward_path) {
//int IsingSolver::MAC_wo_Randomness(const int Rid, std::vector<int> AvailableSpins) {
  // Rid & Cid : Row (visitng order) & Column (city) index
  //Rid Should be between 1 and map_size()-2
  double Iout = 0;
  double Iprev = 0;
  double Inext = 0;
  double Iparas = 0;
  double MAX = 0;
  double MAX_prev = 0;
  double MAX_next = 0;
  int Cid = 0;
  int Seq_prev = (Rid-1)*map_size();
  int Seq_curr = Rid * map_size();
  int Seq_next = (Rid+1)*map_size();

  int erase_id = 0;
  int for_id;
  
  for_id = 0;
  MAX = 0;
  for (int col : AvailableSpins){
    rep(B, BitPrec) rep(i, map_size()){
      if (Rid > 0) Iprev += (((current_spin[Seq_prev + i]+1)/2) * Wd[B*size()+i*map_size() + col]);
      if (Rid < map_size()-1) Inext += (((current_spin[Seq_next + i]+1)/2) * Wd[B*size()+i*map_size() + col]);
      //Parastic current derived by Down-Spins.
      if (Rid == 0) Iparas += (map_size()-1)*Wd_Paras[B*size()+i*map_size() + col];
      else if (Rid == map_size()-1) Iparas += (map_size()-1)*Wd_Paras[B*size()+i*map_size() + col];
      else Iparas += (map_size()-2)*Wd_Paras[B*size()+i*map_size() + col];
    }
    if (Rid == 0) {
      //Iout = 2*(Inext+Iparas);
      Iout = 2*Inext + Iparas;
    }
    else if (Rid == map_size()-1) {
      //Iout = 2*(Iprev+Iparas);
      Iout = 2*Iprev + Iparas;
    }
    else Iout = Iprev+Inext+Iparas;
    //if (current_spin[Seq_curr+col] != 1) Iout = Iout - Isub();
    if (current_spin[Seq_curr+col] == 1) Iout = Iout + Isub();
    if (MAX < Iout) {
      MAX = Iout;
      MAX_prev = Iprev+Iparas;
      MAX_next = Inext+Iparas;
      Cid = col;
      erase_id = for_id;
    }
    ++for_id;
    Iprev = 0;
    Inext = 0;
    Iparas = 0;
  }
  ++nMAC;
  /*if (forward_path) {
    if (Rid < map_size() -2) H_B += MAX_prev;
    else if (Rid == map_size()-2) H_B += MAX_prev + MAX_next;
  }
  else {
    if (Rid > 1) H_B += MAX_next;
    else if (Rid == 1) H_B += MAX_next + MAX_prev;
  }*/
  //H_B += MAX;
  rep (i, map_size()) {
    current_spin[Seq_curr+i] = -1;
  }
  current_spin[Seq_curr+Cid] = 1;
  
  return erase_id;
}
int IsingSolver::MAC(const int Rid, std::vector<int> AvailableSpins, const bool forward_path) {
  // Rid & Cid : Row (visitng order) & Column (city) index
  //Rid Should be between 1 and map_size()-2
  double Iout = 0;
  double Iprev = 0;
  double Inext = 0;
  double MAX = 0;
  int Cid = 0;
  int Seq_prev = (Rid-1)*map_size();
  int Seq_curr = Rid * map_size();
  int Seq_next = (Rid+1)*map_size();

  int erase_id = 0;
  int for_id;

  bool NewSpin = false;
  bool ON = true;
  
  for_id = 0;
  MAX = 0;
  while (ON) {
    for (int col : AvailableSpins){
      rep(B, BitPrec) rep(i, map_size()){
        if (Rid > 0) Iprev += (((current_spin[Seq_prev + i]+1)/2) * Wd[B*size()+i*map_size() + col]);
        if (Rid < map_size()-1) Inext += (((current_spin[Seq_next + i]+1)/2) * Wd[B*size()+i*map_size() + col]);
      }
      if (Rid == 0) {
        Iout = 2*Inext;
      }
      else if (Rid == map_size()-1) {
        Iout = 2*Iprev;
      }
      else Iout = Iprev+Inext;
      if (current_spin[Seq_curr+col] != 1) Iout = Iout - Isub();
      NewSpin = rand()%100 < 100/(1+exp(-(Iout*5*1e5+Imid-50)));
      if (NewSpin) {
        if (MAX < Iout) {
          MAX = Iout;
          Cid = col;
          erase_id = for_id;
        }  
      }
      ++for_id;
      Iprev = 0;
      Inext = 0;
    }
    ++nMAC;
    if (MAX > 0) {
      break;
    }
  }
  rep (i, map_size()) {
    current_spin[Seq_curr+i] = -1;
  }
  current_spin[Seq_curr+Cid] = 1;
  
  return erase_id;
}
double IsingSolver::calcHbyMAC() {
  double Charge = 0;
  int order_prev;
  int order_next;
  int city_prev;
  int city_next;

  rep(y, map_size()-1) {
    order_prev = y*map_size();
    order_next = (y+1)*map_size();

    rep(x, map_size()) {
      if (current_spin[order_prev+x] == 1) city_prev = x;
    }
    rep(x, map_size()) {
      if (current_spin[order_next+x] == 1) city_next = city_prev*map_size() + x;
    }
    rep(B, BitPrec) {
      Charge += Wd[B*size()+city_next];
    }
  }
  return Charge;
}
void IsingSolver::SwapSpinsViolated(const int Seq_curr, const int Cid) {
  int Rid2=0;
  int Cid2=0;
  bool Found = false;
  rep (i, map_size()) {
    if (current_spin[Seq_curr+i]==1) {
      Cid2 = i;
    }
    if (current_spin[i*map_size()+Cid]==1) {
      Rid2 = i;
      Found = true;
    }
  }
  if (Found) {
    if (Cid2 != Cid) {
      rep (i, map_size()) {
        current_spin[Rid2*map_size()+i] = -1;
      }
      current_spin[Rid2*map_size()+Cid2] = 1;
    }
  }
}
void IsingSolver::randomFlip() {
  vector<int> node_ids = random_selector.select(getActiveNodeCount(), rnd);
  node_ids = SyncNodes(node_ids);
  int Rid;
  int Cid;
  for (auto&& node_id : node_ids) {
    Cid = node_id%map_size();
    Rid = node_id/map_size();
    rep (i, map_size()) {
      current_spin[Rid*map_size() + i] = -1;
      current_spin[i*map_size() + Cid] = -1;
    }
    current_spin[node_id] = 1;
  }
}
void IsingSolver::randomFlip_MRAM() {
  std::vector<int> node_ids;

  rep(i, size_opt()) {
    if (rand()%100 < 100/(1+exp(-(Imid-50)))) node_ids.push_back(i);
  }
  node_ids = SyncNodes(node_ids);

  int Rid;
  int Cid;
  for (auto&& node_id : node_ids) {
    Cid = node_id%map_size();
    Rid = node_id/map_size();
    rep (i, map_size()) {
      current_spin[Rid*map_size() + i] = -1;
      current_spin[i*map_size() + Cid] = -1;
    }
    current_spin[node_id] = 1;
  }
}
std::vector<int> IsingSolver::SyncNodes(std::vector<int> node_ids) {
  std::vector<int> nodes_remap;
  switch (seq_clust) {
    case 0: // The topmost cluster
      return node_ids;
      break;
    case 1: // The first sub-cluster
      for (auto&& node_id : node_ids) {
        nodes_remap.push_back(((node_id)/opt_size())*map_size()+(node_id)%opt_size());
      }
      return nodes_remap;
      break;
    default: // The rest of sub-clusters
      for (auto&& node_id : node_ids) {
        nodes_remap.push_back((((node_id)/opt_size())+1)*map_size()+1+(node_id)%opt_size());
      }
      return nodes_remap;
      break;
  }
}
void IsingSolver::cool() {
  //active_ratio *= cool_coe;
  Imid -= Icool;
  //Imid -= (50-Imid)*Icool;
  //Imid *= Icool;
}
void IsingSolver::cooling_rate_scheduler(){
  //if (steps%Patience == Patience-1) Icool *= Factor;
  double Hcurr = getCurrentEnergy();
  if (H_B - Hcurr < Threshold) ++PlateauCNT;
  if (PlateauCNT >= Patience) {
    //Icool -= Factor;
    Icool *= Factor;
    PlateauCNT = 0;
  }
  H_B = Hcurr;
}
void IsingSolver::updateOptimalSpin() {
  if (getCurrentEnergy() < getOptimalEnergy()) {
    optimal_spin = current_spin;
  }
  //cout << "Hmax: " << Hmax << '\n';
  //cout << "H_B: " << H_B << '\n';
  //double Hcurr = calcHbyMAC();
  //cout << "Hcurr: " << Hcurr << '\n';
  //if (Hmax > H_B) {
  //  Hmax = H_B;
  //  optimal_spin = current_spin;
  //}
}
void IsingSolver::step() {
  cool();
  if (current_spin.size() != size()) {
    throw new runtime_error("call init method first");
  }
  //randomFlip();
  randomFlip_MRAM();
  ++nRAND;
  //HminMAC_wo_Randomness_parallel();
  HminMAC_wo_Randomness();
  //HminMAC();
  updateOptimalSpin();
  cooling_rate_scheduler();
  ++steps;
}
size_t IsingSolver::getActiveNodeCount() const {
  return size_t(floor(size_opt() * active_ratio));
}
size_t IsingSolver::getUpdateNodeCount() const {
  return size_t(floor(size_opt() * update_ratio));
}
size_t IsingSolver::size() const {
  return cf.size();
}
//size_t IsingSolver::map_size() const {
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
double IsingSolver::Isub() const {
  //return (pow(2,BitPrec)-1)/RonArr;
  return (pow(2,BitPrec)-1)*(1/(RonArr+RonTr) + (map_size()-2)/(RonArr+RoffTr));
  //return (pow(2,(BitPrec-1))-1)*(2/(RonArr+RonTr) + (map_size()-2)/(RonArr+RoffTr));
}
Weight IsingSolver::calcEnergy(const std::vector<int>& spin) const {
  return cf.calcEnergy(getCurrentPer(), spin);
}
Weight IsingSolver::getCurrentEnergy() const {
  return calcEnergy(current_spin);
}
Weight IsingSolver::getOptimalEnergy() const {
  return calcEnergy(optimal_spin);
}
//double IsingSolver::getCurrentEnergy() const {
//  return H_B;
//}
int IsingSolver::getNumberMAC() const {
  return nMAC;
}
int IsingSolver::getNumberRandFlip() const {
  return nRAND;
}
double IsingSolver::getImid() const {
  return Imid;
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
