#!/usr/bin/env python
# coding: utf-8

from vape_algorithm import *
from fan_algorithm import *
from datetime import datetime
import pickle as pk


# INITIALIZATION GENERAL QUANTITIES
d = 3
N_contexts = 5
B_x = 1
B_theta = 1
B_noise = 1
L_xi = 1
B_y = B_x * B_theta + B_noise

# init time horizons 
T_list = [1000, 1700, 3000, 5000]

reg_fan = np.zeros((4, 15))
reg_vape = np.zeros((4, 15 ))


#  COMPLETE RUN 
for j in range(15):
    print('running iteration:', j)
    
    # init theta
    theta_star = np.random.rand(d)
    theta_star = theta_star * B_theta / np.linalg.norm(theta_star, 2)
    
    # init contexts 
    X_context = np.random.uniform(low=-1, high=1, size=(N_contexts, d))
    X_context /= np.linalg.norm(X_context, axis=1)[:, None]
    
    for i,T in enumerate(T_list):
        print('Running time horizon', T)
        
        # Fan Algorithm
        my_rew, opt_rew = fan_alg(T=T, 
                                 d=d,
                                 contexts=X_context,
                                 N_contexts=N_contexts,
                                 theta_star=theta_star,
                                 bound_x=B_x,
                                 bound_noise=B_noise,
                                 bound_theta=B_theta)
        
        reg_fan[i, j] = opt_rew.sum() - my_rew.sum()
        
        # Vape algorithm
        epsilon_T = get_epsilon(d, T)
        alpha_T = get_alpha(T, B_noise)
        mu_T = get_mu(d, epsilon_T, B_theta, B_y, B_x, T, alpha_T)
        
        my_rew1, opt_rew1 = vape(T=T,
                                theta_star=theta_star,
                                L_xi=L_xi,
                                mu=mu_T,
                                alpha=alpha_T,
                                epsilon=epsilon_T,
                                X_context=X_context,
                                N_context=N_contexts,
                                d=d,
                                bound_theta=B_theta,
                                bound_x=B_x,
                                bound_noise=B_noise
                                )
        
        reg_vape[i, j] = opt_rew1.sum() - my_rew1.sum()
    
    # saving results
    now = datetime.now().strftime("%d_%m_%Y_%Hh%Mm%Ss")
    np.save(f'regFanST_{now}.npy', reg_fan[: , j])
    np.save(f'regVapeST_{now}.npy', reg_vape[:, j])

