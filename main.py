import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse
from scipy.special import expit
import json
import math
import torch
from torch import nn
#from src.data import Data
from src.train import mcmc_train_test
import models
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
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
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Device: ", device)

    print("**** Simulate data ****")
    def build_toy_dataset(N, D=50, noise_std=0.1):
        np.random.seed(1234)
        X = np.random.random((N, D))-0.5
        W = np.random.random((D, 1))-0.5
        b = np.random.normal(0, 1, (N, 1))
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
    X = torch.tensor(X.astype('float32')).to(device)
    y = torch.tensor(y.astype('float32')).to(device)
    W = torch.tensor(W.astype('float32')).to(device)
    V = torch.tensor(V.astype('float32')).to(device)
    #plt.scatter(np.arange(W.size), W.flatten())
    print(math.ceil(max(V)))
    

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
    model_logistic = model1.to(device, torch.float32)
    model_cdf = model2.to(device, torch.float32)
    
    #print(model_logistic(X_t))

    config["model"]["logistic_total_par"] = sum(P.numel() for P in model_logistic.parameters() if P.requires_grad)
    print(config["model"]["logistic_total_par"])

    config["model"]["cdf_total_par"] = sum(P.numel() for P in model_cdf.parameters() if P.requires_grad)
    print(config["model"]["cdf_total_par"])
    
    
    
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