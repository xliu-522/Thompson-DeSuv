import os
import pandas as pd
import torch
import numpy as np
from scipy.special import expit
from scipy.stats import truncnorm, norm  
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# from torchvision import transforms
# from types import SimpleNamespace

class Data(object):
    def __init__(self, config):
        self.D = config['data']['dimension']
        self.N = config['data']['sample_size']
        self.data_path = config['data']['data_path']
        self.X_dist = config['data']['X_dist']
        self.xi_dist = config['data']['noise_dist']
        self.scale_xi = 0.3
        self.loc_xi = 0

    def sample_x_density_trunc(self):
        samples = []
        while len(samples) < self.D:
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
    
    def sample_xi_truncated_distribution(self):
        samples = []
        while len(samples) < self.N:
            # Sample a value from a uniform distribution between -0.5 and 0.5
            xi = np.random.uniform(-0.5, 0.5)
            # Calculate the value of the density function at x
            density_value = (1/4 - xi**2)
            # Generate a uniform random number between 0 and 1
            u = np.random.uniform(0, 1)
            # If the density_value is greater than u, accept the sample
            if density_value > u:
                samples.append(xi)
        return np.array(samples).reshape((self.N, 1))
    
    def xi_truncated_pdf(self, val):
        return np.where(np.abs(val) <= 1/2, (1/4 - val**2)) 

    
    def truncated_gaussian(self, mu, sigma, lower_bound, upper_bound, size=1):
        while True:
            sample = np.random.normal(mu, sigma, size)
            if np.all((sample >= lower_bound) & (sample <= upper_bound)):
                return sample

    def read(self):
        if self.X_dist == 'truncated':
            X = []
            for ii in range(self.N):
                x_ii = self.sample_x_density_trunc()
                X.append(x_ii)
            self.X = np.array(X)
            
        elif self.X_dist == 'Gaussian':
            self.X = np.random.normal(0, 1, (N, D))+1
            
        self.beta = np.random.normal(0, 0.5, (self.D,1))
        
        if self.xi_dist == 'truncated':
            self.xi = self.sample_xi_truncated_distribution()
         
        elif self.xi_dist == 'Gaussian':
            self.xi = np.random.normal(self.loc_xi, self.scale_xi, (self.N, 1))
        
        self.V = self.X.dot(self.beta) + self.xi
        self.y = expit(self.X.dot(self.beta)+self.xi)
        #y = binom.rvs(1, y)  
        self.y[self.y < 0.5] = 0
        self.y[self.y >= 0.5] = 1  
        self.y = self.y.flatten()