# TAXI: Traveling Salesman Problem Accelerator with X-bar-based Ising Macros Powered by SOT-MRAMs and Hierarchical Clustering

This repository is the space to design a Traveling Salesman Problem (TSP) Accelerator with X-bar-based Ising Macros for large-scale TSPs. The backbone of the code is from [Neuro-Ising](https://github.com/souravsanyal06/Neuro-Ising)

![TAXI_Overview](Images/TAXI_main.png)

In "TAXI.ipynb", a TSPlib is clustered, then solved depending on the type of Ising-Solver configured by "MAC_ising" from the higher-level clusters to the lower-level clusters.

1. ising_MAC is a c++ program coded by "Ising_Layer_MAC". It is based on Ising Macros with SOT-MRAM stochastic switching characteristics.

2. ising_RNG is a c++ program coded by "Ising_Layer-RNG". It solves the TSPs by generating random numbers by CMOS-based RNG and flipping random spins.

## Citation

If you find TAXI useful for your work, please cite the following source:
```
@article{sangmin2025TAXI,
        title   =  {TAXI: Traveling Salesman Problem Accelerator with X-bar-based Ising Macros Powered by SOT-MRAMs and Hierarchical Clustering},
        author  =  {Yoo, Sangmin and Holla, Amod and Sanyal, Sourav and Kim, Dong Eun
                    and Iacopi, Francesca and Biswas, Dwaipayan and Myers, James and Roy, Kaushik},
        journal = {Proceedings of the 62nd ACM/IEEE Design Automation Conference (DAC)},
        volume  = {TBD},
        number  = {TBD},
        pages   = {TBD},
        year    = {2025}
}
```

## Acknowledgement

TAXI was initially developed by Sangmin Yoo (joint affiliation in IMEC USA and NRL of Purdue University).

Additional contributions were made by Amond Holla, Sourav Sanyal, and Dong Eun Kim.

This work was funded through the joint MOU between the Indiana Economic Development Corporation, Purdue University, imec with in-kind support from the Applied Research Institute.
