#include "mylib.h"
#include "ising_solver.h"
#include "cost_function.h"
#include <vector>
#include <random>
#include <cassert>
#include <cmath>

using namespace std;

Edge::Edge(int to, Weight weight) : to(to), weight(weight) {}

//cf is costfunction
IsingSolver::IsingSolver(const CostFunction& cf, const int opt_size)
  : steps(0), total_step(0),
    random_selector(opt_size), active_ratio(0), cf(cf) {}
    //random_selector(cf.size()), active_ratio(0), cf(cf) {}
    
int IsingSolver::calcTotalStep(double initial_active_ratio) const {
  // return n s.t. initial_active_ratio * size() * cool_coe^n < 1
  //double n = - (log(initial_active_ratio) + log(size())) / log(cool_coe);
  double n = - (log(initial_active_ratio) + log(size_opt())) / log(cool_coe);
  assert(n >= 0);
  return int(ceil(n));
}

void IsingSolver::init(const IsingSolver::InitMode mode, const int seed, const double cool_coe, const double update_ratio, const double initial_active_ratio, const int seq_clust) {
  assert(0 <= cool_coe && cool_coe < 1);
  assert(0 <= update_ratio && update_ratio <= 1);
  rnd.seed(seed);
  this->cool_coe = cool_coe;
  this->update_ratio = update_ratio;
  this->steps = 0;
  this->active_ratio = initial_active_ratio;
  this->seq_clust = seq_clust;
  this->total_step = calcTotalStep(initial_active_ratio);
  switch (mode) {
    case Negative:
      current_spin.assign(size(), -1);
      break;
    case Positive:
      current_spin.assign(size(), 1);
      break;
    case Random:
      current_spin.resize(size());
      for (auto&& s : current_spin) {
        s = rnd() % 2 ? 1 : -1;
      }
      break;
  }
  
  //cout << "seq_clust in init: " << seq_clust << '\n';
  //cout << "size(): " << size() << '\n';
  switch (seq_clust) {
    case 0: // The topmost cluster
      break;
    case 1: // The first sub-cluster
      rep(i, 1, map_size()){
        current_spin[size()-1-i] = -1;
        current_spin[size()-1-i*map_size()] = -1;
      }
      current_spin[size()-1] = 1;
      break;
    case -1: // The last sub-cluster
      rep(i, 1, map_size()){
        current_spin[i] = -1;
        current_spin[i*map_size()] = -1;
      }
      current_spin[0] = 1;
      break;
    default: // The rest of sub-clusters
      rep(i, 1, map_size()){
        current_spin[i] = -1;
        current_spin[i*map_size()] = -1;
        current_spin[size()-1-i] = -1;
        current_spin[size()-1-i*map_size()] = -1;
      }
      current_spin[0] = 1;
      current_spin[size()-1] = 1;
      break;
  }
  /*if (seq_clust!=0) {
    rep (i, 0, map_size()){
      rep (j, 0, map_size()){
        cout << current_spin[i*map_size()+j];
      }
      cout << '\n';
    }
    cout << "##############" << '\n';
  }*/
  optimal_spin = current_spin;
}

void IsingSolver::init(const IsingSolver::InitMode mode, const double cool_coe, const double update_ratio, const double initial_active_ratio, const int seq_clust) {
  random_device rd;
  init(mode, rd(), cool_coe, update_ratio, initial_active_ratio, seq_clust);
}

void IsingSolver::randomFlip() {
  vector<int> node_ids = random_selector.select(getActiveNodeCount(), rnd);

  //node_ids = SyncNodes(node_ids);
  //cout << "seq_clust in randomFlip: " << seq_clust << '\n';
  node_ids = SyncNodes(node_ids);
  for (auto&& node_id : node_ids) {
    //node_id = SyncNode(node_id);
    //cout << "outside node_id: " << (node_id) << '\n';
    current_spin[node_id] = 1;
    if (current_spin[node_id] > 0) {
      for (auto&& e : cf.J2[node_id]) {
        // if (int(rnd() % 100) * getTotalStep() < getStep() * 100 * 5) {
          current_spin[e.to] = -1;
        // }
      }
    }
  }
}
void IsingSolver::updateNodes() {
  vector<int> node_ids = random_selector.select(getUpdateNodeCount(), rnd);
  node_ids = SyncNodes(node_ids);
  for (auto&& node_id : node_ids) {
    //node_id = SyncNode(node_id);
    //cout << "outside node_id: " << (node_id) << '\n';
    updateNode(node_id);
  }
}
void IsingSolver::updateNode(const int node_id) {
  Weight energy_diff = cf.calcEnergyDiff(getCurrentPer(), current_spin, node_id);
  if (energy_diff > 0) {
    current_spin[node_id] = -1;
  }
  else if (energy_diff < 0) {
    current_spin[node_id] = 1;
  }
  else {
    current_spin[node_id] = rnd() % 2 ? 1 : -1;
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
/*
int IsingSolver::SyncNode(int node_id) {
  //node_id = node_id + 1;
  //cout << "seq_clust in SyncNode: " << seq_clust << "node_id: " << node_id << " " << opt_size() << " " << map_size() << " " << (node_id) << " " << (node_id)/opt_size() << " " << (node_id)%opt_size() << '\n';
  switch (seq_clust) {
    case 0: // The topmost cluster
      break;
    case 1: // The first sub-cluster
      node_id = ((node_id)/opt_size())*map_size()+(node_id)%opt_size();
      break;
    default: // The rest of sub-clusters
      node_id = (((node_id)/opt_size())+1)*map_size()+1+(node_id)%opt_size();
      break;
  }
  //cout << "remapped node_id: " << (node_id) << '\n';
  //node_id = node_id - 1;
  return node_id;
}
*/
void IsingSolver::cool() {
  active_ratio *= cool_coe;
}
void IsingSolver::updateOptimalSpin() {
  if (getCurrentEnergy() < getOptimalEnergy()) {
    optimal_spin = current_spin;
  }
}
void IsingSolver::step() {
  cool();
  if (current_spin.size() != size()) {
    throw new runtime_error("call init method first");
  }
  randomFlip();
  updateNodes();
  updateOptimalSpin();
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
size_t IsingSolver::map_size() const {
  return sqrt(double(cf.size()));
}
size_t IsingSolver::opt_size() const {
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
size_t IsingSolver::size_opt() const {
  return pow(opt_size(),2);
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
