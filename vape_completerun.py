#!/usr/bin/env python
# coding: utf-8

from vape_algorithm import *
import pickle as pk
from datetime import datetime


# init hyperparameters
d = 3
N_context = 5
B_x = 1
B_theta = 1
B_noise = 1
L_xi = 1
B_y = B_x * B_theta + B_noise


# VAPE COMPLETE RUN

T_list = [1000, 10000, 50000, 200000, 500000, 800000]
reg = np.zeros((6, 15))

# T_list = [100]
for j in range(15):
    
    # init parameter theta
    theta_star = np.random.rand(d)
    theta_star = theta_star * B_theta / np.linalg.norm(theta_star, 2) 

    # init for context generation
    X_context = np.random.normal(size=(N_context, d))
    X_context /= np.linalg.norm(X_context, axis=1)[:, None]
    
    print('Running iteration:', j )
    
    for i, T in enumerate(T_list):
        epsilon_T = get_epsilon(d, T)
        alpha_T = get_alpha(T, B_noise)
        mu_T = get_mu(d, epsilon_T, B_theta, B_y, B_x, T, alpha_T)

        print('Current time horizon', T)
        my_rew, opt_rew = vape(T=T,
                               theta_star=theta_star,
                               L_xi=L_xi,
                               mu=mu_T,
                               alpha=alpha_T,
                               epsilon=epsilon_T,
                               X_context=X_context,
                               N_context=N_context,
                               d=d,
                               bound_theta=B_theta, bound_x=B_x, bound_noise=B_noise)

        reg[i, j] = opt_rew.sum() - my_rew.sum()
    
    # saving results
    now = datetime.now().strftime("%d_%m_%Y_%Hh%Mm%Ss")
    np.save(f'regVAPE_{now}.npy', reg[: , j])    

