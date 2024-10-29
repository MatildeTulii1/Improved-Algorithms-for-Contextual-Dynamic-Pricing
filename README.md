This repository contains the code necessary to run the simulations included in Appendix A of the paper 'Improved Algorithms for Contextual Dynamic Pricing'.

The file vape_algorithm.py contains the code for the implementation of VAPE algorithm in the linear valuations case. 
The complete run of the algorithm that resulted in Figure 1 in the paper can be obtained by running the file vape_completereun.py

The two comparisons with the algorithm proposed in Fan et al. 2021, that resulted in the plots of Figure 2, are contained in the files comparison_stochastic.py and comparison_adversarial.py.
The former uses the implementation of VAPE of the aforementioned vape_algorithm.py file, and an implementation of the algorithm proposed in appendix F of Fan et al. 2021, can be found in fan_algorithm.py. 
The latter file is for the comparison in the adversarial case, of which the details are explained in the paper. This relies on vape_algorithm_adversarial.py and fan_algorithmic_adversarial.py 
that are equal to the previous algorithms, but for the way context are received, allowing thus to create an adversarial sequence, which shows two different subsets of the contexts in the eploration 
and exploitaion phases of fan_algorithm.
