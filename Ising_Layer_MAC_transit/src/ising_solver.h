#ifndef __ising_solver__
#define __ising_solver__

#include <vector>
#include <random>
//#include <bits/stdc++.h>
#include "random_selector.h"
#include "cost_function.h"

// minimize: Σs_i s_j J_{ij} + Σs_i h_i
// s := current_spin
// s_i: {-1, 1}
class IsingSolver {
  int steps, total_step, seq_clust, BitPrec, nMAC, nRAND, Patience;
  double cool_coe, Imid, dRtr, VDD, RonTr, RoffTr, RonArr, RoffArr, Rw, Icool, Factor, Threshold, init_Irand, Istop;
  std::mt19937 rnd;
  RandomSelector random_selector;
  double active_ratio; // temp: [0, 1]
  const CostFunction cf;
  const std::vector<double> Wd;
  const std::vector<double> Wd_Paras;
  const std::vector<double> WdG;
  const std::vector<double> WdG_Paras;
  double H_Optim, H_curr;
  std::vector<double> KeepH;
  int PlateauCNT, OptimCNT;
  std::vector<int> OptSpins, current_spin, optimal_spin;
  Weight calcEnergyDiff(const std::vector<int>& spin, const int node_id) const;
  // H minimization using MAC operations
  void HminMAC();
  // MAC Operation for Energy-minimization
  int MAC(const int Rid, std::vector<int> AvailableSpins);
  double calcHbyMAC();
  // tempLower
  void cool();
  void cooling_rate_scheduler();
  // If you find a better solution optimal_spin Update
  void updateOptimalSpin();
  int calcTotalStep(double initial_active_ratio, double init_Irand) const;
public:
  enum InitMode {
    Negative, Positive, Random
  };
  IsingSolver(const CostFunction& cf, const int opt_size, const std::vector<double> Wd, const std::vector<double> Wd_Paras, const std::vector<double> WdG, const std::vector<double> WdG_Paras, std::vector<double> KeepH);
  Weight getCurrentEnergy() const;
  int getNumberMAC() const;
  int getNumberRandFlip() const;
  Weight getOptimalEnergy() const; // However, it is calculated by the current objective function
  Weight calcEnergy(const std::vector<int>& spin) const;
  const std::vector<int>& SpinsToOptimize() const;
  const std::vector<int>& getCurrentSpin() const;
  const std::vector<int>& getOptimalSpin() const;
  void setCurrentSpin(const std::vector<int>& new_spin);
  void step();
  void init(const InitMode mode, const double cool_coe, const double initial_active_ratio, const int seq_clust, const double VDD,
    const double RonTr, const double RoffTr, const double RonArr, const double RoffArr, const double Rw, const int BitPrec, const double Factor, const double Threshold, const int Patience, double init_Irand);
  void init(const InitMode mode, const int seed, const double cool_coe, const double initial_active_ratio, const int seq_clust, const double VDD,
    const double RonTr, const double RoffTr, const double RonArr, const double RoffArr, const double Rw, const int BitPrec, const double Factor, const double Threshold, const int Patience, double init_Irand);
  size_t getActiveNodeCount() const;
  size_t size() const;
  int map_size() const;
  int opt_size() const;
  int size_opt() const;
  int getStep() const;
  int getTotalStep() const;
  double getCurrentPer() const;
};

#endif
