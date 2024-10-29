#!/usr/bin/env python
# coding: utf-8

from vape_algorithm_adversarial import *
from fan_algorithm_adversarial import *
from datetime import datetime
import pickle as pk


#  INITIALIZATION HYPERPARAMETERS

# init hyperparameters
d = 3
N_context = 2
B_x = 1
B_theta = 1
B_noise = 1
L_xi = 1
B_y = B_x * B_theta + B_noise

# init parameter theta
theta_star = np.ones(3)
theta_star /= np.linalg.norm(theta_star, 2) 


# COMPLETE RUN 

T_list = [1000, 1400, 4200, 9000]

reg_fanAD = np.zeros((4, 15))
reg_vapeAD = np.zeros((4, 15))


for j in range(15):
    print('Running iteration:', j)
    
    X_context = np.zeros((2,3))
    X_context[0, 0] = np.random.rand()
    X_context[0, 2] = np.random.rand()
    X_context[1, 1] = 1
    X_context /= np.linalg.norm(X_context, axis=1)[:, None]
    
    for i,T in enumerate(T_list):
        print('currently running horizon:', T)
        my_rew, opt_rew, = fan_alg_adv(T=T, 
                                  d=d, 
                                  contexts=X_context, 
                                  N_contexts=N_context, 
                                  theta_star=theta_star, 
                                  bound_x=B_x, 
                                  bound_noise=B_noise, 
                                  bound_theta=B_theta)
        reg_fanAD[i, j] = opt_rew.sum() - my_rew.sum()
        
        epsilon_T = get_epsilon(d, T)
        alpha_T = get_alpha(T, B_noise)
        mu_T = get_mu(d, epsilon_T, B_theta, B_y, B_x, T, alpha_T)
        my_rew1, opt_rew1 = vape_adv(T=T,
                                 theta_star=theta_star,
                                 L_xi=1,
                                 mu=mu_T,
                                 alpha=alpha_T,
                                 epsilon=epsilon_T,
                                 X_context=X_context,
                                 N_contexts=N_context,
                                 d=d,
                                 bound_theta=B_theta,
                                 bound_x=B_x,
                                 bound_noise=B_noise)
        reg_vapeAD[i, j] = opt_rew1.sum() - my_rew1.sum()
        
    now = datetime.now().strftime("%d_%m_%Y_%Hh%Mm%Ss")
    np.save(f'regFanADV_{now}.npy', reg_fanAD[: , j])
    np.save(f'regvapeADV_{now}.npy', reg_vapeAD[: , j])

