# TAXI: Traveling Salesman Problem Accelerator with X-bar-based Ising Macros Powered by SOT-MRAMs and Hierarchical Clustering

This repository is the space to design a Taveling Salesman Problem (TSP) Accelerator with X-bar-based Ising Macros for the large-scale TSP.

In "TAXI.ipynb", a TSPlib is clustered, then solved depending on the type of Ising-Solver configured by "MAC_ising" from the higher-level clusters to the lower-level clusters.

1. ising_MAC is a c++ program coded by "Ising_Layer_MAC". It is based on Ising Macros with SOT-MRAM stochastic switching characteristics.

2. ising_RNG is a c++ program coded by "Ising_Layer-RNG". It solves the TSPs by generating random numbers by CMOS-based RNG and flipping random spins.

