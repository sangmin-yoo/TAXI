#include "mylib.h"
#include "problem.h"
#include "ising_solver.h"
#include "mid.h"
#include "mid_grid.h"
#include "lib/cmdline.h"
#include "distance_matrix.h"

#include <iostream>
#include <fstream>
#include <cassert>
#include <tuple>

using namespace std;

void run(const cmdline::parser& parser) {
  //const is for unchangeable and read-only data
  // & is reference that is a "reference" to an existing variable.
  // & can be useful when you need to change the value of the arguments:
  //const int& seq_clust = std::stoi(parser.rest()[0]);
  const int& seq_clust = std::stoi(parser.rest()[0].substr(1,parser.rest()[0].size()-1));
  const string& input_file_path = parser.rest()[1];

  //const string ifpath = input_file_path;
  //ifstream infile(ifpath);
  ifstream ifs(input_file_path);
  if (ifs.fail()) {
    cerr << "can't open the file: " << input_file_path << endl;
    exit(1);
  }
  // Mid mid(Problem::fromIstream(ifs));
  Mid mid = parser.get<int>("grid") == 1 ? Mid(Problem::fromIstream(ifs))
    : MidWithGrid(Problem::fromIstream(ifs), parser.get<int>("grid"));
  // solvers[0] is the main solver
  // solvers[1..] is the sub solvers with different cool coefficient
  const CostFunction cf = mid.getCostFunction();
  const int swidth = parser.get<int>("swidth");
  const double base_cool = parser.get<double>("cool");
  const double initial_active_ratio = min(1., 1. / sqrt(double(cf.size())));
  // Hardware specs of Memory Arrays
  const double RonArr = parser.get<double>("Ron-Arr");
  const double RoffArr = parser.get<double>("Roff-Arr");
  const double Rw = parser.get<double>("Rw");
  const int BitPrec = parser.get<int>("BitPrec");
  // Macro-level Hardware specs 
  const double VDD = parser.get<double>("VDD");
  const double RonTr = parser.get<double>("Ron-tr");
  const double RoffTr = parser.get<double>("Roff-tr");
  const double IsotL = parser.get<double>("IsotL");
  const double IsotH = parser.get<double>("IsotH");
  const double Irange = IsotH-IsotL;
  const double tMAC = parser.get<double>("tMAC");
  // For keeping track of the number of cycles
  int nMAC = 0;
  const bool is_realistic = parser.get<bool>("is_realistic");
  //const bool is_realistic = true;
  vector<IsingSolver> solvers;//Vector of IsingSolver: Multiple IsingSolvers are in "solvers"
  int main_solver_idx = -1;
  int size_opt;
  int Oid;
  switch (seq_clust) {
    case 0: // The topmost cluster
      size_opt = cf.size();
      Oid = 0;
      break;
    case 1: // The first cluster
      size_opt = pow(sqrt(cf.size())-1,2);
      Oid = 0;
      break;
    case -1: // The last cluster
      size_opt = pow(sqrt(cf.size())-1,2);
      Oid = 1;
      break;
    default: // The rest of clusters
      size_opt = pow(sqrt(cf.size())-2,2);
      Oid = 1;
      break;
  }
  DMatrix distance_matrix = DMatrix(ifs, RonArr, RoffArr, Rw, BitPrec, cf.size());
  const std::vector<double> Wd = distance_matrix.getDMatrix(ifs, Oid, is_realistic);
  //const std::vector<double> Wd = distance_matrix.getDMatrix(ifs, Oid, false);
  //from -swidth to swidth+1, add a IsingSolver to solvers by changing cool
  rep(i, -swidth, swidth+1) {
    const double cool = base_cool * pow(base_cool, 0.1 * i);//pow(a,b)=a^b
    if (0 <= cool && cool < 1) {
      if (i == 0) main_solver_idx = solvers.size();
      IsingSolver solver(cf, size_opt, Wd);
      solver.init(IsingSolver::InitMode::Random, cool, parser.get<double>("update-ratio"), initial_active_ratio,
      seq_clust, VDD, RonTr, RoffTr, RonArr, RoffArr, IsotL, IsotH, Irange, tMAC, BitPrec, nMAC);
      solvers.push_back(move(solver));
    }
  }
  for (auto&& solver : solvers) {
    solver.step();
  }
  assert(main_solver_idx >= 0);//program will be terminated if main_solver_idx < 0
  const IsingSolver& main_solver = solvers[main_solver_idx];
  // solve
  bool is_detail = parser.exist("detail");
  const int ExtraStepCount = 10;
  bool is_first = true;

  //while (main_solver.getStep() < main_solver.getTotalStep()+ExtraStepCount) {
  while (main_solver.getStep() < 1.5*(main_solver.getTotalStep()+ExtraStepCount)) {
    if (!is_first) {
      for (auto&& solver : solvers) {
        solver.step();
      }
      // solver Share state between
      for (auto&& base_solver : solvers) for (auto&& ref_solver : solvers) {
        if (base_solver.calcEnergy(ref_solver.getCurrentSpin()) < base_solver.getCurrentEnergy()) {
          base_solver.setCurrentSpin(ref_solver.getCurrentSpin());
        //cout << "base solver: " << base_solver.getCurrentEnergy() << '\n';
        //cout << "ref solver: " << ref_solver.getCurrentEnergy() << '\n';
        //cout << "############################\n";
        //if (ref_solver.getCurrentEnergy() < base_solver.getCurrentEnergy()) {
        //  base_solver.setCurrentSpin(ref_solver.getCurrentSpin());
        }
      }
    }
    else is_first = false;
    //cout << "[Step " << main_solver.getStep() << " / " << main_solver.getTotalStep()+ExtraStepCount << "]" << endl;
    //cout << "energy: " << main_solver.getCurrentEnergy() << endl;
    if (is_detail) cout << "spin: " << main_solver.getCurrentSpin() << endl;
    //cout << "flip: " << main_solver.getActiveNodeCount() << " / " << main_solver.size() << endl;
    Answer ans = mid.getAnswerFromSpin(main_solver.getCurrentSpin());
    //ans.output(cout, is_detail);
    //cout << "is_answer: " << boolalpha << ans.verify() << endl;
    //cout << endl;
    //cout << main_solver.getCurrentEnergy() << '\n';
  }
  int n_MAC = 0;
  for (auto&& solver : solvers) {
    if (n_MAC < solver.getNumberMAC()) {
      n_MAC = solver.getNumberMAC();
    }
  }
  //cout << "[Answer]" << endl;
  //cout << "energy: " << main_solver.getOptimalEnergy() << endl;
  if (is_detail) cout << "spin: " << main_solver.getOptimalSpin() << endl;
  Answer ans = mid.getAnswerFromSpin(main_solver.getOptimalSpin());
  ans.output(cout, true);
  cout << "n_MAC " << n_MAC <<'\n';
  //cout << "is_answer: " << boolalpha << ans.verify() << endl; 
}
cmdline::parser get_command_line_parser() {
  cmdline::parser parser;
  parser.add<double>("cool", 'c', "coefficient of cooling", false, 0.999);
  parser.add<double>("update-ratio", 'u', "the ratio of nodes to update in 1 step", false, 0.3);
  parser.add<int>("grid", 'g', "width and height of the grid", false, 8);
  parser.add<int>("swidth", 's', "the max number of sub solvers / 2", false, 2);
  parser.add("detail", 'd', "print log in detail");
  parser.add<double>("VDD", 'v', "Supply voltage", false, 1.5);
  parser.add<double>("Ron-tr", 'T', "On resistance of transistors", false, 1e4);
  parser.add<double>("Roff-tr", 't', "Off resistance of transistors", false, 1e9);
  parser.add<double>("Rw", 'w', "Parasitic resistance (Wire)", false, 1e3);
  //parser.add<double>("Ron-Arr", 'A', "On resistance of Memory in the Array", false, 1e5);
  //parser.add<double>("Ron-Arr", 'A', "On resistance of Memory in the Array", false, 2e5);//69165
  parser.add<double>("Ron-Arr", 'A', "On resistance of Memory in the Array", false, 1.5e7);
  parser.add<double>("Roff-Arr", 'a', "Off resistance of Memory in the Array", false, 1e9);
  parser.add<int>("BitPrec", 'b', "Bit Precision of Memory in the Array", false, 8);
  parser.add<double>("IsotL", 'i', "Lowerbound of variable range in current", false, 4.6e-4);
  parser.add<double>("IsotH", 'I', "Upperbound of variable range in current", false, 5.6e-4);
  parser.add<double>("tMAC", 'm', "time to operate one MAC operation", false, 1e-9);
  parser.add<bool>("is_realistic", 'B', "Consider parasitic components?", false, false);
  parser.footer("filename");
  return parser;
}
int main(int argc, char *argv[]) {
  auto parser = get_command_line_parser();
  parser.parse_check(argc, argv);
  if (parser.rest().size() < 1) {
    cerr << parser.usage();
    exit(1);
  }
  run(parser);
  
  cout<<endl;
}
