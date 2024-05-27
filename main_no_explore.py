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
from src.data import Data
from src.train import mcmc_train_test
import src.kernels
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

    

    print("**** Simulate data ****")
    N, D = config["data"]["sample_size"], config["data"]["dimension"]
    data_obj = Data(config)
    data_obj.read()
    X, y, beta, V, xi = data_obj.X, data_obj.y, data_obj.beta, data_obj.V, data_obj.xi
    
    # X = torch.tensor(X.astype('float32')).to(device)
    # y = torch.tensor(y.astype('float32')).to(device)
    
    #plt.scatter(np.arange(W.size), W.flatten())
    B = math.ceil(max(V.squeeze()))
    print("Max value: ", B)
    plt.plot(V)
    plt.savefig("value.pdf")
    plt.close()
    plt.hist(xi)
    plt.savefig("noise_dist.pdf")
    plt.show()
    plt.close()
    
    
    #Initialize beta
    beta_t = np.random.normal(0, 0.1, (D,1))
    alpha = 0.1
    step = 1
    nSple = 20
    eta0 = 0.5
    beta_list = []
    Dataset = {"X":[], "Y":[], "P":[]}
    cum_regret_plcy = 0
    cum_regret_random = 0
    # Generate D_0 dataset 
    X_0 = X[0]
    V_0 = V[0]
    Dataset["X"].append(X_0)
    P_0 = np.array([np.random.uniform(0, B)])
    # u = X_0.dot(beta_t)
    # g_0 = u + invfunc(-u)
    # P_0 = min(max(g_0, 0), B)
    Dataset["P"].append(P_0)
    Y_0 = int(V_0 >= P_0)
    Dataset["Y"].append(Y_0)
    cum_regret_random_list = []
    cum_regret_plcy_list = []

    if config["data"]["noise_dist"] == "Gaussian":
        xi_pdf = lambda x: norm.pdf(x, data_obj.loc_xi, data_obj.scale_xi)
        xi_cdf = lambda x: norm.cdf(x, data_obj.loc_xi, data_obj.scale_xi)
    else:
        xi_pdf = lambda x: data_obj.xi_truncated_pdf(x)
        xi_cdf = lambda x: [0 if t < -0.5 else (quad(xi_pdf, -0.5, x)[0] if t <= 0.5 else 1) for t in x]

    phi_func = lambda u:  u - (1-xi_cdf(u))/xi_pdf(u)
    invfunc = inversefunc(phi_func)
    for tt in range(1, N):
        print("tt: ", tt)
        X_t = X[tt]
        V_t = V[tt]
        P_arr = np.array(Dataset["P"])
        X_arr = np.array(Dataset["X"])
        model_cdf = models.CDFNet2(D, Dataset, device).to(device)
        with torch.no_grad():
            model_cdf.first_layer.bias = nn.Parameter(torch.tensor(P_arr, dtype=torch.float32).to(device)) 
        model_cdf.optimise(torch.tensor(-X_arr.astype('float32')).to(device),1500)

        #Collect beta and throw away W
        beta_trained = np.transpose(model_cdf.first_layer.weight.cpu().detach().numpy())

        #Use collected beta to re-estimate CDF using kernel
        Y_arr = np.array(Dataset["Y"])
        w_t = (P_arr - X_arr.dot(beta_trained)).flatten()
        t_val = np.linspace(-1, 1, 100)
        f_hat = src.kernels.d_F_est(t_val, w_t, Y_arr)
        F_hat = src.kernels.F_est(t_val, w_t, Y_arr)
        # f_hat = xi_pdf(t_val)
        # F_hat = xi_cdf(t_val)
        

        # Plot the noise distribution
        if tt % 10 == 0:
            sample_xi = xi_pdf(t_val)
            plt.plot(t_val, sample_xi, label ="Estimated pdf")
            plt.plot(t_val, f_hat, label="True pdf")
            plt.legend()
            plt.savefig(f'est_pdf_{len(Dataset["P"])}.png')
            plt.show()
            plt.close()
            plt.plot(t_val, F_hat)
            plt.savefig(f'est_cdf_{len(Dataset["P"])}.png')
            plt.close()

        # propose price 
        u_t = X_t.dot(beta_trained)
        phi_inv = [float(fsolve(src.kernels.phi_func_est, x0=0, args=(w_t, Y_arr, u_t, )))]
        phi_est = -(phi_inv - (1-src.kernels.F_est(phi_inv, w_t, Y_arr))/src.kernels.d_F_est(phi_inv, w_t, Y_arr))
        # phi_inv = [float(fsolve(src.kernels.phi_func_known, x0=0, args=(u_t, )))]
        # phi_est = -(phi_inv - (1-xi_cdf(phi_inv))/xi_pdf(phi_inv)) 
        print("phi_inv: ", phi_inv)
        print("phi_t:", u_t)
        print("phi_est:", phi_est)
        P_t = np.array(min(max([0], u_t + phi_inv), [B]))
        #P_t = np.array([min(max(0, P_t[0]), B)])
        print("P_t: ", P_t)
        Y_t = int((V_t >= P_t)[0])

        Dataset["X"].append(X_t)
        Dataset["Y"].append(Y_t)
        Dataset["P"].append(P_t)

        # Calculate oracle price using beta_* and F_*
        u_star = X_t.dot(data_obj.beta).squeeze()

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
    np.save('plcy_output.npy', cum_regret_plcy_list)
    np.save('random_output.npy', cum_regret_random_list)
if __name__ == "__main__":
    main()