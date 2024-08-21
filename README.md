# Large-scale-Ising
This repository is the space to design a hierarchical Ising Machine (HIM) for the large-scale Combinatorial Optimization Problems. The first demonstration will be made for Traveling Salesman Problems.

In "Large-Scale-Ising.ipynb", a TSPlib is clustered, then solved depending on the type of Ising-Solver configured by "MAC_ising" from the higher-level clusters to the lower-level clusters.

1. ising_RNG is a c++ program coded by "Ising_Layer". It solves the TSPs by generating random numbers by RNG and flipping random spins. (Completed)

2. ising_MAC is a c++ program coded by "Ising_Layer_MAC". It is based on MAC operations for the Hamiltonian minimization. (On-going)
