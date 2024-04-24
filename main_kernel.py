import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse
from scipy.special import expit
from scipy.stats import truncnorm, norm  
from scipy.optimize import fsolve, minimize_scalar, newton
import json
import math
import torch
from torch import nn
#from src.data import Data
from src.train import mcmc_train_test
import models
import numpy as np
import random
from numpy import linalg as LA
from sympy import symbols, diff
import matplotlib.pyplot as plt
from pynverse import inversefunc
from datetime import datetime
import time
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, make_interp_spline
#from unique_names_generator import get_random_name

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs='?', type=str, default="config/config.json", required=False)
    args = parser.parse_args()
    try:
        with open(args.config, ) as config:
            config = json.load(config)
            #config['model']["random_name"] = get_random_name().replace(" ", "_")
    except:
        print("Error in config")
    
    print("**** Checking device ****")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Device: ", device)

    # Get current date and time
    current_time = datetime.now()

    # Format the date and time as a string
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Create the folder
    try:
        os.mkdir('result/' + folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    except OSError as e:
        print(f"Error: {e}")
    
    try:
        os.chdir('result/' + folder_name)
        print(f"Changed directory to '{folder_name}'.")
    except OSError as e:
        print(f"Error: {e}")

    def truncated_gaussian(mu, sigma, lower_bound, upper_bound, size=1):
        while True:
            sample = np.random.normal(mu, sigma, size)
            if np.all((sample >= lower_bound) & (sample <= upper_bound)):
                return sample

    print("**** Simulate data ****")

    a_trunc = -0.5  # Lower bound
    b_trunc = 0.5 # Upper bound
    x_trunc = math.sqrt(2/3)
    loc_xi = 0  # Mean
    scale_xi = 0.5  # Standard deviation
    a, b = (a_trunc - loc_xi) / scale_xi, (b_trunc - loc_xi) / scale_xi
    t_eval = np.linspace(-10,10,100)
    def sample_x_density(D):
        samples = []
        while len(samples) < D:
            # Sample a value from a uniform distribution between -sqrt(2/3) and sqrt(2/3)
            x = np.random.uniform(-np.sqrt(2/3), np.sqrt(2/3))
            # Calculate the value of the density function at x
            density_value = (2/3 - x**2)**3
            # Generate a uniform random number between 0 and 1
            u = np.random.uniform(0, 1)
            # If the density_value is greater than u, accept the sample
            if density_value > u:
                samples.append(x)
        return np.array(samples)
    
    def sample_xi_distribution(num_samples):
        samples = []
        while len(samples) < num_samples:
            # Sample a value from a uniform distribution between -0.5 and 0.5
            x = np.random.uniform(-0.5, 0.5)
            # Calculate the value of the density function at x
            density_value = (1/4 - x**2)
            # Generate a uniform random number between 0 and 1
            u = np.random.uniform(0, 1)
            # If the density_value is greater than u, accept the sample
            if density_value > u:
                samples.append(x)
        return np.array(samples).reshape((num_samples, 1))
    
    def build_toy_dataset(N, a, b, D=50, noise_std=0.1):
        # np.random.seed(1234)
        #X = np.random.normal(0, 1, (N, D))
        #W = np.random.random((D, 1))
        # X = np.random.normal(0, 1, (N, D))+1
        X = []
        for ii in range(N):
            x_ii = sample_x_density(D)
            X.append(x_ii)
        X = np.array(X)
        # X = np.array([truncnorm.rvs(-x_trunc, x_trunc, size = D) for _ in range(N)])
        W = np.ones((D, 1)) * math.sqrt(2/3)
        #xi = np.array([truncnorm.rvs(a, b, size=1) for _ in range(N)])
        #xi = sample_xi_distribution(N)
        xi = np.random.normal(0, scale_xi, (N, 1))
        # zero_ind = np.arange(D//4, D)
        # W[zero_ind, :] = 0
        V = X.dot(W) + xi
        y = expit(X.dot(W)+xi)
        #y = binom.rvs(1, y)  
        y[y < 0.5] = 0
        y[y >= 0.5] = 1  
        y = y.flatten()
        # X = torch.tensor(X, dtype=torch.float32)
        # V = torch.tensor(X, dtype=torch.float32)
        return X, y, W, V, xi
    
    N, D = config["data"]["sample_size"], config["data"]["dimension"]
    X, y, W, V, xi = build_toy_dataset(N, a, b, D)
    print("**** Fit spline interpolation ****")

    # X = torch.tensor(X.astype('float32')).to(device)
    # y = torch.tensor(y.astype('float32')).to(device)
    # W = torch.tensor(W.astype('float32')).to(device)
    #plt.scatter(np.arange(W.size), W.flatten())
    B = math.ceil(max(V.squeeze()))
    print(B)
    plt.plot(V)
    plt.plot(X.dot(W))
    plt.show()
    plt.hist(xi)
    plt.show()
    
    def F(u):
        return norm.cdf(u, loc=loc_xi, scale=scale_xi)

    # Define the probability density function (PDF) of the truncated normal distribution
    def f_prime(u):
        return norm.pdf(u, loc=loc_xi, scale=scale_xi)
    print("**** Load Model ****")

    # Define kernel function used to approximate the noise CDF
    # class kernel_est_CDF:
    #     def __init__(self, x_t, beta)
    l_ra = 200 # length of the exploration phase
    m = 2 #the order of the kernel function
    I_k = min((D * l_ra)**((2*m+1)/(4*m-1)), l_ra)
    b_ra = 3 * (1/I_k**(1/(2*m+1)))
    print("I_k: ", I_k)
    print("b_ra: ", b_ra)

    def kernel_2(u):
        return np.where(np.abs(u) <= 1, (35/12) * (1 - u**2)**3, 1e-10)
        
    def d_kernel_2(u):
        return np.where(np.abs(u) <= 1, -35/2 * u * (1 - u**2)**2, 1e-10)
        
    def kernel_4(u):
        return (27/16)*(1-(11/3) * u**2) * kernel_2(u)
    
    def d_kernel_4(u):
        return -(27/16) * (11/3) * 2*u * kernel_2(u) + kernel_4(u) * d_kernel_2(u)
        #return np.where(np.abs(u) <= 1, -36.09375*u*(1 - u**2)**3 - 6*u*(1 - u**2)**2*(4.921875 - 18.046875*u**2), 1e-7)

    def kernel_6(u):
        return (297/128) * (1-26/3 * u**2 + 13 * u**4) * kernel_2(u)

    def h_k(u, w_t, y_t):
        #print((w_t - u)/b_ra)
        return np.array([1/(b_ra * I_k) * np.sum(kernel_2((w_t - uu)/b_ra) * y_t) for uu in u])

    def f_k(u, w_t):
        return np.array([1/(b_ra * I_k) * np.sum(kernel_2((w_t - uu)/b_ra)) for uu in u])

    def d_h_k(u, w_t, y_t):
        return np.array([-1/(b_ra**2 * I_k) * np.sum(d_kernel_2((w_t - uu)/b_ra) * y_t) for uu in u])

    def d_f_k(u, w_t):
        return np.array([-1/(b_ra**2 * I_k) * np.sum(d_kernel_2((w_t - uu)/b_ra)) for uu in u])

    def F_hat(u, w_t, y_t):
        return 1 - h_k(u, w_t, y_t)/f_k(u, w_t) 

    def d_F_hat(u, w_t, y_t):
        return - (d_h_k(u, w_t, y_t) * f_k(u, w_t) - h_k(u, w_t, y_t) * d_f_k(u, w_t))/f_k(u, w_t)**2
    
    def phi_func_est(u, w_t, y_t, phi_t):
        return u - (1-F_hat(u, w_t, y_t))/d_F_hat(u, w_t, y_t) + phi_t
    
    def kernel_function(x, xi, h):
        # Gaussian kernel function
        return np.exp(-0.5 * ((x - xi) / h)**2) / (np.sqrt(2 * np.pi) * h)
    
    def logpi(tt, D_or, D_it, beta_or, beta_t, mu0=0, vsigma=1):
        mu0 = np.zeros(D)
        v2 = 0.5
        x_it = np.array(D_it["X"])
        y_it = np.array(D_it["Y"])
        p_it = np.array(D_it["P"])

        u_it = (p_it - x_it.dot(beta_t)).flatten() #(D_it)
        
        x_or = np.array(D_or["X"])
        y_or = np.array(D_or["Y"])
        p_or = np.array(D_or["P"])
        #F_t = 1 / (1 + np.exp(-2 * (output - 1.5)))
        w_t = (p_or - x_or.dot(beta_or)).flatten()
        F_t = F_hat(u_it, w_t, y_or) 
        f_t = d_F_hat(u_it, w_t, y_or)
        # print("F_t:", len(F_t))
        # print("f_t: ", len(f_t))
        loss = sum([ff/FF * (1-yy/(1-FF)) * xx for ff, FF, yy, xx in zip(f_t, F_t, y_it, x_it)]).reshape((D,1))
        #print("loss:", loss)
        lpi = sum([yy * np.log(1-FF) + (1-yy) * np.log(FF) for yy, FF in zip(y_it, F_t)])
        loss = loss + (beta_t - mu0)/v2
        lpi = lpi - LA.norm(beta_t - mu0)**2/v2
        return loss, lpi

    phi_func = lambda u:  u - (1-F(u))/f_prime(u)
    invfunc = inversefunc(phi_func)
    
    #Initialize beta
    beta_t = np.random.normal(0, 0.1, (D,1))
    alpha = 0.2
    step = 1
    nSple = 10
    eta = 0.1
    beta_list = []
    D_it = {"I":[], "X":[], "Y":[], "P":[]}
    D_or = {"X":[], "Y":[], "P":[]}
    cum_regret_plcy = 0
    cum_regret_random = 0
    # Generate D_0 dataset 
    X_0 = X[0]
    V_0 = V[0]
    #D_it["X"].append(X_0)
    u = X_0.dot(beta_t)
    g_0 = u + invfunc(-u)
    P_0 = min(max(g_0, 0), B)
    #D_it["P"].append(P_0)
    Y_0 = int(V_0 >= P_0)
    #D_it["Y"].append(Y_0)
    cum_regret_random_list = []
    cum_regret_plcy_list = []
    # Run the exploration for length l_ra
    for tt in range(1, N):
        print("tt: ", tt)
        X_t = X[tt]
        V_t = V[tt]
        if tt <= l_ra:
            P_t = np.random.uniform(0, B, size = 1)
            Y_t = int((V_t >= P_t)[0])
            D_or["X"].append(X_t)
            D_or["Y"].append(Y_t)
            D_or["P"].append(P_t)
            D_it["X"].append(X_t)
            D_it["Y"].append(Y_t)
            D_it["P"].append(P_t)
            len_plo = len(D_or["P"])
            
        else:
            P_or = np.array(D_or["P"])
            X_or = np.array(D_or["X"])
            y_or = np.array(D_or["Y"])
            # solve beta using least square
            #A = np.vstack([X_or.T, np.ones(len(X_or))]).T
            beta_or = np.linalg.lstsq(X_or, B*y_or, rcond=None)[0].reshape((D,1))
            #print("lsq_beta:", beta_or)
            # Update estimate of Beta using exploitation dataset
            loss, lpi = logpi(tt, D_or, D_it, beta_or, beta_t)
            print(step)
            if step % 50 == 0:
                for k in range(nSple):
                    epsilon = np.random.normal(size=(D,1))
                    beta_prop = beta_t - eta*loss + math.sqrt(2*eta) * epsilon
                    lossprop, lpiprop = logpi(tt, D_or, D_it, beta_or, beta_prop)
                    r = lpiprop - lpi        
                    prob = min(1, math.exp(r))
                    if np.random.uniform(1) <= prob:
                        beta_t = beta_prop
                        lpi = lpiprop
                        loss = lossprop
                    beta_list.append(beta_t)
            step+=1
            #print(beta_t)
            I_t = 1 if np.random.rand() <= alpha else 0

            if I_t == 1:
                # Explore
                P_t = np.random.uniform(0, B, size = 1)
                Y_t = int((V_t >= P_t)[0])
                D_or["X"].append(X_t)
                D_or["Y"].append(Y_t)
                D_or["P"].append(P_t)
                l_ra += 1

            else:
                # Plot the estimated noise pdf
                I_k = min((D * l_ra)**((2*m+1)/(4*m-1)), l_ra)
                X_or = np.array(D_or["X"])
                P_or = np.array(D_or["P"])
                Y_or = np.array(D_or["Y"])
                w_t = (P_or - X_or.dot(beta_or)).flatten()
                t_val = np.linspace(-2, 2, 100)
                f_hat = d_F_hat(t_val, w_t, Y_or)
                F_hat_est = F_hat(t_val, w_t, Y_or)
                #print("f_hat:", f_hat)
                #sample_xi = sample_xi_distribution(len(f_hat))
                random.seed(42)
                sample_xi = norm.pdf(t_val, loc_xi, scale_xi)
                #print(sample_xi)
                plt.plot(t_val, sample_xi)
                plt.plot(t_val, f_hat)

                plt.legend(["Estimated pdf", "True pdf"])
                plt.savefig(f'est_pdf_{len(D_or["P"])}.png')
                #plt.show()
                plt.close()
                #plt.show()
                # Approximate price
                u_t = X_t.dot(beta_t)
                phi_inv = [float(fsolve(phi_func_est, x0=0, args=(w_t, Y_or, u_t, )))]
                phi_est = -(phi_inv - (1-F_hat(phi_inv, w_t, Y_or))/d_F_hat(phi_inv, w_t, Y_or))
                print("phi_inv: ", phi_inv)
                print("phi_t:", u_t)
                print("phi_est:", phi_est)

                P_t = np.array(min(max([0], u_t + phi_inv), [B]))
                print("P_t: ", P_t)
                Y_t = int((V_t >= P_t)[0])

            D_it["X"].append(X_t)
            D_it["Y"].append(Y_t)
            D_it["P"].append(P_t)

            fig, ax = plt.subplots()
            P_it = np.array(D_it["P"])
            X_it = np.array(D_it["X"])
            #ax.hist(P_it - X_it.dot(beta_t), density=True, bins=50, alpha=0.6)
            #ax.hist(xi, density=True, bins=50, alpha=0.6)
            #ax.plot(t_eval, f_dist, '-r', label='DeCDF')
            # plt.xlabel('x')
            # plt.ylabel('p(x)')
            # plt.legend(["estimated error distribution", "True distribution"])
            # plt.title("Learning a MoG distribution (n=1000)")
            # plt.tight_layout()
            # plt.savefig(f'est_pdf_{len(D_it["P"])}_2.png')
            # plt.close()
                
    

        # Calculate oracle price using beta_* and F_*
        u_star = X_t.dot(W).squeeze()
        
        P_star = u_star+invfunc(-u_star)
        print("P_star: ", P_star)
        P_random = np.random.uniform(0, B, 1)
        # Calculate regret
        regret_plcy = P_star * int((V_t >= P_star)[0]) - P_t * Y_t
        cum_regret_plcy += regret_plcy
        cum_regret_plcy_list.append(cum_regret_plcy[0])
        regret_random = P_star * int((V_t >= P_star)[0]) - P_random * int((V_t >= P_random)[0])
        cum_regret_random += regret_random
        cum_regret_random_list.append(cum_regret_random[0])
        
    plt.plot(cum_regret_plcy_list)
    plt.plot(cum_regret_random_list)
    plt.legend(['plcy', 'random'])
    plt.savefig('regret.png')
    plt.show()

if __name__ == "__main__":
    main()