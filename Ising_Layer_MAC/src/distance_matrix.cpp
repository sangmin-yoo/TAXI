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
DMatrix::DMatrix(std::istream& is, const double RonArr, const double RoffArr, const double Rw, const int BitPrec, const int ArrSize)
  : DtoR_ratio(1), input_path(is), RonArr(RonArr), RoffArr(RoffArr), Rw(Rw), BitPrec(BitPrec), ArrSize(ArrSize){}
DMatrix::~DMatrix() {}
std::vector<double> DMatrix::getDMatrix(std::istream& is, const int Oid, const bool is_realistic) {
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
  /*
  cout << "##############" << '\n';
  cout << DistMax << '\n';
  cout << "##############" << '\n';
  cout << "Rm" << '\n';
  rep (i, 0, nCity){
    rep (j, 0, nCity){
      cout << Rm[i][j] << " ";
    }
    cout << '\n';
  }
  cout << "##############" << '\n';
  cout << "Gm" << '\n';
  */
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
      //Gm[i][j] = floor((DistMin/Rm[i][j])*FullPrec);
      Gm[i][j] = floor(((DistMax - Rm[i][j])/(DistMax-DistMin))*FullPrec);
    }
  }
  //std::vector<double> BWC;
  //rep(i,nCity) rep(j,nCity) {
  //  BWC.push_back(Gm[i][j]);
  //}
  /*
  rep (i, 0, nCity){
    rep (j, 0, nCity){
      cout << Gm[i][j] << " ";
    }
    cout << '\n';
  }
  cout << "##############" << '\n';
  cout << "BW" << '\n';
  */
  
  // Mapping the quantized Distances onto (BitPrec) of binary arrays considering the parasitic resistance according to each position.
  std::vector<int> BW (ArrSize*BitPrec);
  //BW.reserve(ArrSize*BitPrec);
  rep(b,BitPrec) {
    rep(i,nCity) rep(j,nCity) {
      BW[b*ArrSize + i*nCity + j] = Gm[i][j] % 2;
      Gm[i][j] = Gm[i][j]/2;
    }
  }

  /*
  rep(b,BitPrec) {
    cout << b << "bit \n";
    rep(i,nCity) {
      rep(j,nCity) {
        cout << BW[b*ArrSize + i*nCity + j];
      }
      cout << "\n";
    }
    cout << "##############" << '\n';
  }
  */
  
  double GLow = 1/RoffArr;
  double GHigh = 1/RonArr;
  std::vector<double> BWC;
  if (is_realistic) {
    rep(b,BitPrec) {
      rep(i,nCity) rep(j,nCity) {
        if (BW[b*ArrSize + i*nCity + j]==1) {
          BWC.push_back(pow(2,b)*1/(RonArr+Rw*(i+nCity-1-j)));
        }
        else {
          BWC.push_back(pow(2,b)*1/(RoffArr+Rw*(i+nCity-1-j)));
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
  
  /*
  rep(b,BitPrec) {
    cout << b << "bit \n";
    rep(i,nCity) {
      rep(j,nCity) {
        cout << BWC[b*ArrSize + i*nCity + j] << " ";
      }
      cout << "\n";
    }
    cout << "##############" << '\n';
  }
  */
  return BWC;
}