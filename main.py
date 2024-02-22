import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse
from scipy.special import expit
from scipy.stats import truncnorm  
from scipy.optimize import fsolve
import json
import math
import torch
from torch import nn
#from src.data import Data
from src.train import mcmc_train_test
import models
import numpy as np
import matplotlib.pyplot as plt
from pynverse import inversefunc
from datetime import datetime
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

    def truncated_gaussian(mu, sigma, lower_bound, upper_bound, size=1):
        while True:
            sample = np.random.normal(mu, sigma, size)
            if np.all((sample >= lower_bound) & (sample <= upper_bound)):
                return sample

    print("**** Simulate data ****")

    a_trunc = -5  # Lower bound
    b_trunc = 5  # Upper bound
    loc = 0  # Mean
    scale = 1  # Standard deviation
    a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale

    def build_toy_dataset(N, D=50, noise_std=0.1):
        np.random.seed(1234)
        # X = np.random.random((N, D))
        # W = np.random.random((D, 1))
        # X = np.random.normal(0, 1, (N, D))
        a_trunc = 0  # Lower bound
        b_trunc = 2  # Upper bound
        loc = 0  # Mean
        scale = 1  # Standard deviation
        a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale
        X = np.array([truncnorm.rvs(a, b, size = D) for _ in range(N)])
        W = np.random.random((D, 1))
        b = np.array([truncnorm.rvs(a, b, size=1) for _ in range(N)] )
        #b = np.random.normal(0, 1, (N, 1))
        zero_ind = np.arange(D//4, D)
        W[zero_ind, :] = 0
        V = X.dot(W) + b
        y = expit(X.dot(W)+b)
        #y = binom.rvs(1, y)  
        y[y < 0.5] = 0
        y[y >= 0.5] = 1  
        y = y.flatten()
        # X = torch.tensor(X, dtype=torch.float32)
        # V = torch.tensor(X, dtype=torch.float32)
        return X, y, W, V

    N, D = config["data"]["sample_size"], config["data"]["dimension"]
    X, y, W, V = build_toy_dataset(N, D)
    
    print("**** Fit spline interpolation ****")

    # X = torch.tensor(X.astype('float32')).to(device)
    # y = torch.tensor(y.astype('float32')).to(device)
    # W = torch.tensor(W.astype('float32')).to(device)
    #plt.scatter(np.arange(W.size), W.flatten())
    B = math.ceil(max(V.squeeze()))
    print(B)
    plt.plot(V)
    plt.show()
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def equation_to_solve(x, y):
        return x - (1+np.exp(-x)) - y
    
    def F(u):
        return truncnorm.cdf(u, a, b, loc=loc, scale=scale)

# Define the probability density function (PDF) of the truncated normal distribution
    def f_prime(u):
        return truncnorm.pdf(u, a, b, loc=loc, scale=scale)
    print("**** Load Model ****")
    model_name1 = config["model"]["model_name1"]
    model_name2 = config["model"]["model_name2"]
    model_reference1 = getattr(models, model_name1)
    model_reference2 = getattr(models, model_name2)
    if model_reference1 is not None and callable(model_reference1):
        # Create an instance of the class
        model1 = model_reference1()
        print(f"Found model {model_name1}. ")
    else:
        print(f"Model {model_name1} not found.")

    if model_reference2 is not None and callable(model_reference2):
        # Create an instance of the class
        model2 = model_reference2()
        print(f"Found model {model_name2}. ")
    else:
        print(f"Model {model_name2} not found.")

    # Initialize the neural network with a random dummy batch (Lazy)
    model_logistic = model1.to(device)
    model_cdf = model2.to(device)
    #model_logistic.apply(init_weights)
    #print(list(model_logistic.parameters()))
    #print(model_logistic(X_t))

    config["model"]["logistic_total_par"] = sum(P.numel() for P in model_logistic.parameters() if P.requires_grad)
    print(config["model"]["logistic_total_par"])

    config["model"]["cdf_total_par"] = sum(P.numel() for P in model_cdf.parameters() if P.requires_grad)
    print(config["model"]["cdf_total_par"])
    
    def logpi(D_it, beta_t, mu0=0, vsigma=1):
        x_t = D_it["X"]
        y_t = D_it["Y"]
        p_t = D_it["P"]
        pred = x_t.dot(beta_t).squeeze()
        F_t =  expit(model_cdf.mapping(torch.tensor((p_t - pred).astype('float32'))).detach().numpy())
        f_t = np.exp(-model_cdf.forward(torch.tensor((p_t - pred).astype('float32'))).detach().numpy().squeeze())
        plt.plot(F_t)
        plt.show()
        loss = sum([ff/FF * (1-yy/(1-FF)) * xx for ff, FF, yy, xx in zip(f_t, F_t, y_t, x_t)])
        print(loss)
        lpi = sum([yy * np.log(1-FF) + (1-yy) * np.log(FF) for yy, FF in zip(y_t, F_t)])
        return loss, lpi
    
    #Initialize beta
    # beta_t = np.random.normal(0, 0.1, (D,1))
    # l_t = 0
    # l_ra = 1000
    # l_ta = 500
    # alpha = 0.2
    # step = 0
    # nSple = 10
    # eta0 = 0.5
    # beta_list = []
    # D_it = {"I":[], "X":[], "Y":[], "P":[]}
    # D_or = {"X":[], "Y":[], "P":[]}
    # cum_regret_plcy = 0
    # cum_regret_orcl = 0
    # for tt in range(N):
    #     X_t = X[tt]
    #     V_t = V[tt]
    #     I_t = 1 if np.random.rand() <= alpha else 0
    #     if I_t == 1:
    #         # Exploration phase
    #         #X_b = x_t.dot(beta_t).squeeze()
    #         P_t = np.random.uniform(0, B, size = 1)
    #         D_or["X"].append(X_t)
    #         D_or["P"].append(P_t)
    #         #error_est = P_exp - X_b
    #         # plt.hist(error_est)
    #         # plt.show()
    #     # X_exp = X[l_t:l_t+l_ra]
    #     # V_exp = V[l_t:l_t+l_ra]
    #     # Offer price p_t ~ N(X^Tb,1)
    #     #X_b = model_logistic(torch.tensor(X_exp.astype('float32'))).detach().numpy().squeeze()
       

        
    #     else:
    #         # Approximate noise distribution using exploration dataset
    #         P_or = D_or["P"]
    #         X_or = D_or["X"]
    #         if len(P_or) > 1000:
    #             # If we have enough data point to estimate NN
    #             error_est = P_or - X_or.dot(beta_t)
    #             model_cdf.optimise(torch.tensor(error_est.astype('float32')),1500)
    #         #If not, just use the initialization of the network
    #         t_eval = np.linspace(-10,10,100)
    #             #y = expit(cnet.mapping(torch.tensor(t_eval.astype('float32'))).detach().numpy())
    #         F_dist = expit(model_cdf.mapping(torch.tensor(t_eval.astype('float32'))).detach().numpy())
    #         f_dist = np.exp(-model_cdf.forward(torch.tensor(t_eval.astype('float32'))).detach().numpy().squeeze())
    #         plt.plot(t_eval, F_dist)
    #         plt.show()
    #         fig, ax = plt.subplots()
    #         ax.hist(error_est, density=True, bins=50, alpha=0.6)
    #         ax.plot(t_eval, f_dist, '-r', label='DeCDF')
    #         plt.xlabel('x')
    #         plt.ylabel('p(x)')
    #         plt.legend()
    #         plt.title("Learning a MoG distribution (n=1000)")
    #         plt.tight_layout()
    #         plt.show()

    #     # # Store data
    #     # y_exp = np.array([float(p <= v) for p, v in zip(P_exp, V_exp)])

    #         # Update estimate of Beta using exploitation dataset
    #         eta = eta0/(tt+1)
    #         loss, lpi = logpi(D_it, beta_t)
    #         if step % 50 == 0:
    #             for k in range(nSple):
    #                 epsilon = np.random.normal(size=D)
    #                 beta_prop = beta_t - eta*loss + math.sqrt(2*eta) * epsilon
    #                 lossprop, lpiprop = logpi(D_it, beta_prop)
    #                 r = lpiprop - lpi        
    #                 prob = min(1, math.exp(r))
    #                 if np.random.uniform(1) <= prob:
    #                     beta_t = beta_prop
    #                     lpi = lpiprop
    #                     loss = lossprop
    #                 beta_list.append(beta_t)
        

    #         # Update esimtate of phi and g using spline interpolation
    #         u_t = X_t.dot(beta_t).squeeze()
    #         phi_values = np.linspace(min(u_t), max(u_t), 100)
    #         t_values = []
    #         for phi in phi_values:
    #             solution = float(fsolve(equation_to_solve, x0=0, args=(y,)))
    #             t_values.append(solution)
    #         g_w = model_cdf.dudt(torch.tensor(t_values.astype('float32'))[:,None]).squeeze().detach().numpy()
    #         F0_ut = model_cdf.mapping(torch.tensor(t_values.astype('float32'))).detach().numpy()
    #         phi_values = t_values - (1+np.exp(-F0_ut))/g_w
    #         # Generate some example data
    #         sorted_indices = np.argsort(phi_values)
    #         phi_sorted = phi_values[sorted_indices]
    #         t_sorted = t_values[sorted_indices]
   
    #         # Create a cubic spline interpolation function
    #         spline_interpolation = CubicSpline(phi_sorted, t_sorted)

    #         # Generate more points for a smoother curve
    #         phi_interp = np.linspace(min(phi_sorted), max(phi_sorted), 20)
    #         t_interp = spline_interpolation(phi_interp)

    #         # Plot the original data and the spline interpolation
    #         plt.scatter(phi_sorted, t_sorted, label='Original Data')
    #         plt.plot(phi_interp, t_interp, label='Spline Interpolation', color='r')
    #         plt.xlabel('phi')
    #         plt.ylabel('t')
    #         plt.legend()
    #         plt.show()

    #         P_t = u_t + spline_interpolation(-u_t)
    #         D_it["X"] = X_t
    #         D_it["P"] = P_t

    #         # Calculate oracle price
    #         u_star = X_t.dot(W).squeeze()
    #         phi_star = u_star - (1-norm.cdf(u_star))/norm.pdf(u_star)

        
    #     # Calculate reward
    #     regret_plcy = P_t * int(V_t >= P_t)
        
        #f_t = np.exp(-model_cdf.forward(torch.tensor((P_exp - pred).astype('float32'))).detach().numpy().squeeze())
        # print(max(1-F_t))
        # print(min(1-F_t))
        

        # Update estimate of Beta
        # loss_fn = nn.BCELoss()
        # pred = model_logistic(torch.tensor(X_exp.astype('float32'))).detach().numpy().squeeze()
        # F_t = model_cdf.mapping(torch.tensor((P_exp - pred).astype('float32'))).detach()
        # F_t[F_t < 0] = 0
        # F_t = torch.tensor(F_t, requires_grad=True)
        # f_t = np.exp(-model_cdf.forward(torch.tensor((P_exp - pred).astype('float32'))).detach().numpy().squeeze())
        # # print(max(1-F_t))
        # # print(min(1-F_t))
        # loss = loss_fn(1-F_t, y_exp)
        # loss.backward()

        # Update estimate of phi
        
        ## update episode
        l_t = l_t + l_ra + l_ta
        

    # F_t = model_cdf.mapping(phi_t).detach().numpy()
    # f_t = np.exp(-model_cdf.forward(phi_t).detach().numpy().squeeze())

    # using iteration to find inverse of phi_func
    # t = torch.tensor(0)
    # t_new = 1
    # while t_new - t > 0.0001:
    #     t = t_new
    #     W_t = t - 1/(model_cdf.dudt(t) * (1+model_cdf.mapping(t))) - phi_t
    #     u_prime_t = 1/3
    # phi_func = (lambda u: u - (1-model_cdf.mapping(u).detach().numpy())/np.exp(-model_cdf.forward(u_t).detach().numpy().squeeze()))
    # invfunc = inversefunc(phi_func)
    # u_t = u_t.detach().numpy()
    # g_t = u_t + invfunc(-u_t)
    # p_t = min(max(g_t.squeeze(), 0), B)
    # print(p_t)
    # v_t = V[0]
    # y_t = torch.tensor(v_t >= p_t).float().squeeze()
    # print(y_t)
    # loss_fn = nn.BCELoss()
    # F_t = model_cdf.mapping(torch.tensor((p_t - u_t).astype('float32')))
    # print(F_t)
    # loss = loss_fn(1-F_t, y_t)
    # print(loss)
    # print("**** Create directory ****")
    # # Get the current date and time
    # current_time = datetime.now()

    # # Format the current time as a string (adjust the format as needed)
    # time_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # res_dir = f'./result/{config["sampler"]["sampler"]}_{config["model"]["model_name"]}_{config["data"]["dataset"]}_{config["training"]["gamma"]}_{config["training"]["epoches"]}_{config["training"]["cycles"]}_{time_string}'
    # if not os.path.exists(res_dir):
    #     os.makedirs(res_dir)
    #     print(f"Folder '{res_dir}' created.")
    # else:
    #     print(f"Folder '{res_dir}' already exists.")



    #trainer = mcmc_train_test(device=device, X=X, y=y, theta=W, V = V, config=config, model_cdf=model_cdf, model_logistic=model_logistic)
    
    # # print("**** Start training ****")
    # trainer.train_it()
    
    # np.save(f'{res_dir}/loss.npy',loss)
    # np.save(f'{res_dir}/accuracy.npy', acc)
    
    
    # plt.plot(loss)
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    # plt.title('Testing Loss Curve')
    # plt.savefig(f'{res_dir}/loss.png')
    # plt.close()
    # plt.plot(acc)
    # plt.xlabel('Iterations')
    # plt.ylabel('Accuracy')
    # plt.title('Testing accuracy Curve')
    # plt.savefig(f'{res_dir}/accuracy.png')
    # plt.close()
    # if config["sampler"]["sampler"] == "sasgld" or config["sampler"]["sampler"] == "sacsgld":
    #     plt.plot(sparsity)
    #     plt.xlabel('Iterations')
    #     plt.ylabel('Sparsity')
    #     plt.title('Sparsity Curve')
    #     plt.savefig(f'./{res_dir}/sparsity.png')
    #     np.save(f'{res_dir}/sparsity.npy', sparsity)
    # plt.close()
    



if __name__ == "__main__":
    main()