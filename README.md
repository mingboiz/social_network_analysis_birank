## Social network analytics for supervised fraud detection

Implementation of [Oskarsdottir et al (2020)](https://arxiv.org/abs/2009.08313) - Social network analytics for supervised fraud detection

`cookbook` show how I extended [BrianAronson's BiRank](https://github.com/BrianAronson/birankr), implementing support for prior vectors of known fraudulent claims. I also provide explanation on how should one check for correctness of the birank score calculated, through two methods presented in the original [He et al (2017) BiRank](https://arxiv.org/abs/1708.04396) paper. Providing support for known fraudulent parties could also be done in a similar fashion, but it doesn't make sense for this specific use-case and has not been implemented.

Modelling was done on the Sample Network in Figure 1 and verified by comparing Claim C1's features values generated from the algorithm against data from Figure 5, Table 7 and Table 8. `birank` notebook shows provides the extended Bipartite Class in `birank.py` for implementation in your own work, with additional `generate_prior` and `check_birank` methods as discussed in `cookbook`. 
