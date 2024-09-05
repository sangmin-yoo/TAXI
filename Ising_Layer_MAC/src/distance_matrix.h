#ifndef __distance_matrix__
#define __distance_matrix__

#include "ising_solver.h"
#include "problem.h"
#include <vector>

class DMatrix {
protected:
  const double DtoR_ratio;
  const std::istream& input_path;
  const double RonArr;
  const double RoffArr;
  const double RonTr;
  const double RoffTr;
  const double Rw;
  const int BitPrec;
  const int ArrSize;
public:
  DMatrix(std::istream& is, const double RonArr, const double RoffArr, const double RonTr, const double RoffTr, const double Rw, const int BitPrec, const int ArrSize);
  virtual ~DMatrix();
  //virtual std::vector<double> getDMatrix(std::istream& is, const int Oid, const bool is_realistic);
  virtual std::pair<std::vector<double>, std::vector<double>> getDMatrix(std::istream& is, const int Oid, const bool is_realistic);
};

#endif
