#include "mylib.h"
#include "problem.h"
#include "ising_solver.h"
#include "mid.h"
#include "mid_grid.h"
#include "lib/cmdline.h"

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
  vector<IsingSolver> solvers;//Vector of IsingSolver: Multiple IsingSolvers are in "solvers"
  int main_solver_idx = -1;
  int opt_size;
  switch (seq_clust) {
    case 0: // The topmost cluster
      opt_size = cf.size();
      break;
    case 1: // The first cluster
      opt_size = pow(sqrt(cf.size())-1,2);
      break;
    case -1: // The last cluster
      opt_size = pow(sqrt(cf.size())-1,2);
      break;
    default: // The rest of clusters
      opt_size = pow(sqrt(cf.size())-2,2);
      break;
  }
  //from -swidth to swidth+1, add a IsingSolver to solvers by changing cool
  rep(i, -swidth, swidth+1) {
    const double cool = base_cool * pow(base_cool, 0.1 * i);//pow(a,b)=a^b
    if (0 <= cool && cool < 1) {
      if (i == 0) main_solver_idx = solvers.size();
      IsingSolver solver(cf, opt_size);
      solver.init(IsingSolver::InitMode::Random, cool, parser.get<double>("update-ratio"), initial_active_ratio, seq_clust);
      solvers.push_back(move(solver));
    }
  }
  assert(main_solver_idx >= 0);//program will be terminated if main_solver_idx < 0
  const IsingSolver& main_solver = solvers[main_solver_idx];
  // solve
  bool is_detail = parser.exist("detail");
  const int ExtraStepCount = 10;
  bool is_first = true;
  while (main_solver.getStep() < main_solver.getTotalStep()+ExtraStepCount) {
    if (!is_first) {
      for (auto&& solver : solvers) {
        solver.step();
      }
      // solver Share state between
      for (auto&& base_solver : solvers) for (auto&& ref_solver : solvers) {
        if (base_solver.calcEnergy(ref_solver.getCurrentSpin()) < base_solver.getCurrentEnergy()) {
          base_solver.setCurrentSpin(ref_solver.getCurrentSpin());
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
  }
  //cout << "[Answer]" << endl;
  //cout << "energy: " << main_solver.getOptimalEnergy() << endl;
  if (is_detail) cout << "spin: " << main_solver.getOptimalSpin() << endl;
  Answer ans = mid.getAnswerFromSpin(main_solver.getOptimalSpin());
  ans.output(cout, true);
  //cout << "is_answer: " << boolalpha << ans.verify() << endl;
}
cmdline::parser get_command_line_parser() {
  cmdline::parser parser;
  parser.add<double>("cool", 'c', "coefficient of cooling", false, 0.999);
  parser.add<double>("update-ratio", 'u', "the ratio of nodes to update in 1 step", false, 0.3);
  parser.add<int>("grid", 'g', "width and height of the grid", false, 8);
  parser.add<int>("swidth", 's', "the max number of sub solvers / 2", false, 2);
  parser.add("detail", 'd', "print log in detail");
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
