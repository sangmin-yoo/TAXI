#include "ising_solver.h"
#include "distance_matrix.h"
#include "mylib.h"
#include <vector>
#include <cassert>
#include <cmath>
#include <bits/stdc++.h>

using namespace std;

// initialize variables by "variable1(variable1 input), variable2(variable2 input)".
// variable 1 and 2 should be defined in .h file.
DMatrix::DMatrix(std::istream& is, const double RonArr, const double RoffArr, const double RonTr, const double RoffTr, const double Rw, const int BitPrec, const int ArrSize)
  : DtoR_ratio(1), input_path(is), RonArr(RonArr), RoffArr(RoffArr), RonTr(RonTr), RoffTr(RoffTr), Rw(Rw), BitPrec(BitPrec), ArrSize(ArrSize){}
DMatrix::~DMatrix() {}
//std::vector<double> DMatrix::getDMatrix(std::istream& is, const int Oid, const bool is_realistic) {
std::pair<std::vector<double>, std::vector<double>> DMatrix::getDMatrix(std::istream& is, const int Oid, const bool is_realistic) {
  is.clear();
  is.seekg(1);
  int nCity = sqrt(ArrSize);
  // Reading from an input file
  vector<double> CITY(2,0);
  vector<vector<double>> XY(nCity, CITY);
  double x, y;
  rep (i, Oid, nCity) {
    //cout << i << '\n';
    is >> x >> y;
      XY[i][0] = x;
      XY[i][1] = y;
    //cout << XY[i][0] << "  " << XY[i][1] << '\n';
  }
  // Calculate distances between cities and Map them in resistance.
  vector<double> Rrow(nCity, 0);
  vector<vector<double>> Rm(nCity, Rrow);
  double dX, dY;
  double DistMax = 0;
  double DistMin = INFINITY;
  rep(i,nCity) rep(j, nCity) {
    if (i == j) {
      Rm[i][j] = INFINITY;
    }
    else {
      dX = XY[i][0] - XY[j][0];
      dY = XY[i][1] - XY[j][1];
      Rm[i][j] = sqrt(pow(dX,2) + pow(dY,2));
      if (Rm[i][j] > DistMax) {
        DistMax = Rm[i][j];
      }
      if (Rm[i][j] < DistMin) {
        DistMin = Rm[i][j];
      }
    }
  }
  // Normalize, flip (R to G), and quantize the distance
  vector<int> Grow(nCity, 0);
  vector<vector<int>> Gm(nCity, Grow);
  double FullPrec = pow(2,BitPrec)-1;
  rep(i,nCity) rep(j, nCity) {
    if (i == j) {
      Gm[i][j] = 0;
      continue;
    }
    else {
      //Gm[i][j] = floor((1 - Rm[i][j]/DistMax)*FullPrec);
      Gm[i][j] = floor((DistMin/Rm[i][j])*FullPrec);
      //Gm[i][j] = floor(((DistMax - Rm[i][j])/(DistMax-DistMin))*FullPrec);
    }
  }
  // Mapping the quantized Distances onto (BitPrec) of binary arrays considering the parasitic resistance according to each position.
  std::vector<int> BW (ArrSize*BitPrec);
  //BW.reserve(ArrSize*BitPrec);
  rep(b,BitPrec) {
    rep(i,nCity) rep(j,nCity) {
      BW[b*ArrSize + i*nCity + j] = Gm[i][j] % 2;
      Gm[i][j] = Gm[i][j]/2;
    }
  }
  double GLow = 1/RoffArr;
  double GHigh = 1/RonArr;
  std::vector<double> BWC;
  std::vector<double> BWC_Paras;
  if (is_realistic) {
    rep(b,BitPrec) {
      rep(i,nCity) rep(j,nCity) {
        if (BW[b*ArrSize + i*nCity + j]==1) {
          //BWC.push_back(pow(2,b)*1/(RonTr+RonArr+Rw*(i+nCity-1-j)));
          //BWC_Paras.push_back(pow(2,b)*1/(RoffTr+RonArr+Rw*(i+nCity-1-j)));
          
          //BWC.push_back(pow(2,b)*1/(RonTr+RonArr+Rw*(i+nCity-1-j+(BitPrec-b-1)*12)));
          //BWC_Paras.push_back(pow(2,b)*1/(RoffTr+RonArr+Rw*(i+nCity-1-j+(BitPrec-b-1)*12)));
          BWC.push_back(RonArr);
        }
        else {
          //BWC.push_back(pow(2,b)*1/(RonTr+RoffArr+Rw*(i+nCity-1-j)));
          //BWC_Paras.push_back(pow(2,b)*1/(RoffTr+RoffArr+Rw*(i+nCity-1-j)));
          
          //BWC.push_back(pow(2,b)*1/(RonTr+RoffArr+Rw*(i+nCity-1-j+(BitPrec-b-1)*12)));
          //BWC_Paras.push_back(pow(2,b)*1/(RoffTr+RoffArr+Rw*(i+nCity-1-j+(BitPrec-b-1)*12)));

          BWC.push_back(RoffArr);
        }
      }
    }
  }
  else {
    rep(b,BitPrec) {
      rep(i,nCity) rep(j,nCity) {
        if (BW[b*ArrSize + i*nCity + j]==1) {
          BWC.push_back(pow(2,b)*GHigh);
        }
        else {
          BWC.push_back(pow(2,b)*GLow);
        }
      }
    }
  }
  return make_pair(BWC, BWC_Paras);
}