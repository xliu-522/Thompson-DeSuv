import torch
import random
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from scipy.special import expit

class CDFNet(nn.Module):
    """Solve conditional ODE. Single output dim."""
    def __init__(self, D, D_or, hidden_dim=32, output_dim=1, device="cpu",
                 nonlinearity=nn.Tanh, n=15, lr=1e-4):
        super().__init__()
        
        self.output_dim = output_dim
        self.D_or = D_or

        if device == "gpu":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"CondODENet: {device} specified, {self.device} used")
        else:
            self.device = torch.device("cpu")
            print(f"CondODENet: {device} specified, {self.device} used")

        self.first_layer = nn.Linear(D, 1)

        self.dudt = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nonlinearity(),

            nn.Linear(hidden_dim, hidden_dim),
            nonlinearity(),

            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()
        )

        self.n = n
        u_n, w_n = np.polynomial.legendre.leggauss(n)
        self.u_n = nn.Parameter(torch.tensor(u_n,device=self.device,dtype=torch.float32)[None,:],requires_grad=False)
        self.w_n = nn.Parameter(torch.tensor(w_n,device=self.device,dtype=torch.float32)[None,:],requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.subNet = LogisticNet()
        
    def mapping(self, t):
        t = t[:,None]
        a = 0
        b = t 
        tau = torch.matmul((b - a)/2, self.u_n) + (b+a)/2 # N x n
        tau_ = torch.flatten(tau)[:,None] # Nn x 1. Think of as N n-dim vectors stacked on top of each other
        f_n = self.dudt(tau_).reshape((*tau.shape, self.output_dim)) # N x n x d_out
        sign_fn = torch.sign(tau).reshape(*tau.shape, self.output_dim)
        sign_fn[sign_fn == 0] = 1
        #print(sign_fn)
        #f_n = f_n * sign_fn
        pred = (b - a)/2 * ((self.w_n[:,:,None] * f_n).sum(dim=1)) # 
        return pred.squeeze()
    
    # def mapping_prime(self, t):
    #     t = t[:,None].to(self.device)
    #     tau = torch.matmul(t/2, 1+self.u_n) # N x n
    #     tau_ = torch.flatten(tau)[:,None] # Nn x 1. Think of as N n-dim vectors stacked on top of each other
    #     f_n = self.dudt(tau_).reshape((*tau.shape, self.output_dim)) # N x n x d_out
    #     tau2_ = torch.tensor(tau_, requires_grad=True)
    #     f_n2 = self.dudt(tau2_).reshape((*tau.shape, self.output_dim))
       
    #     dfn_dx = torch.autograd.grad(f_n2, tau2_, grad_outputs=torch.ones_like(f_n2))[0]
    #     pred_prime = 1/2 * ((self.w_n[:,:,None] * f_n).sum(dim=1)) + t/2 * ((self.w_n[:,:,None] * dfn_dx).sum(dim=1)) + 0.5*(self.u_n + 1)
    #     return pred_prime
    

    def forward(self, t):
        F = self.mapping(t)
        du = self.dudt(t[:,None]).squeeze()
        #return -(torch.log(du) + torch.log(1-F**2))
        return -(torch.log(du) + F - 2*torch.log(1+torch.exp(F)))
        #return -(torch.log(du) + F - 2*torch.log(1+torch.exp(F)))
    
    def sum_forward(self, t):
        Y = torch.tensor(self.D_or["Y"])
        F = self.mapping(t)
        Pi_or = -(Y * torch.log(1-torch.sigmoid(F)) + (1-Y)*torch.log(torch.sigmoid(F))).sum()
        return Pi_or
        #return self.forward(t).sum()
    
    def optimise(self, t, niters):
        for i in range(niters):
            self.optimizer.zero_grad()
            t2 = self.first_layer(t)
            t2 = t2.flatten()
            loss = self.sum_forward(t2)
            loss.backward()
            self.optimizer.step()
            if i % 100 == 0:
                print(loss.item())

    def mapping_logistic(self, x, p):
        t = p - self.subNet(x)
        t = t[:,None].to(self.device)
        tau = torch.matmul(t/2, 1+self.u_n) # N x n
        tau_ = torch.flatten(tau)[:,None] # Nn x 1. Think of as N n-dim vectors stacked on top of each other
        f_n = self.dudt(tau_).reshape((*tau.shape, self.output_dim)) # N x n x d_out
        pred = t/2 * ((self.w_n[:,:,None] * f_n).sum(dim=1))
        pred = torch.tanh(pred).squeeze()
        pred[pred<0] = 0
        return pred # F_0(x)
    
class CDFNet2(nn.Module):
    """Solve conditional ODE. Single output dim."""
    def __init__(self, D, Dataset, device, hidden_dim=32, output_dim=1, nonlinearity=nn.Tanh, n=15, lr=1e-4):
        super().__init__()
        
        self.output_dim = output_dim
        self.D = D
        self.Dataset = Dataset

        if device == "gpu":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"CondODENet: {device} specified, {self.device} used")
        else:
            self.device = torch.device("cpu")
            print(f"CondODENet: {device} specified, {self.device} used")

        self.first_layer = nn.Linear(D, 1)

        self.dudt = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nonlinearity(),

            nn.Linear(hidden_dim, hidden_dim),
            nonlinearity(),

            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()
        )

        self.n = n
        u_n, w_n = np.polynomial.legendre.leggauss(n)
        self.u_n = nn.Parameter(torch.tensor(u_n,device=self.device,dtype=torch.float32)[None,:],requires_grad=False)
        self.w_n = nn.Parameter(torch.tensor(w_n,device=self.device,dtype=torch.float32)[None,:],requires_grad=False)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def mapping(self, t):
        t = t[:,None]
        a = 0
        b = t 
        tau = torch.matmul((b - a)/2, self.u_n) + (b+a)/2 # N x n
        tau_ = torch.flatten(tau)[:,None] # Nn x 1. Think of as N n-dim vectors stacked on top of each other
        f_n = self.dudt(tau_).reshape((*tau.shape, self.output_dim)) # N x n x d_out
        sign_fn = torch.sign(tau).reshape(*tau.shape, self.output_dim)
        sign_fn[sign_fn == 0] = 1
        #print(sign_fn)
        #f_n = f_n * sign_fn
        pred = (b - a)/2 * ((self.w_n[:,:,None] * f_n).sum(dim=1)) # 
        return pred.squeeze()
    
    def forward(self, t):
        F = self.mapping(t)
        du = self.dudt(t[:,None]).squeeze()
        #return -(torch.log(du) + torch.log(1-F**2))
        return -(torch.log(du) + F - 2*torch.log(1+torch.exp(F)))
        #return -(torch.log(du) + F - 2*torch.log(1+torch.exp(F)))
    
    def sum_forward(self, t):
        Y = torch.tensor(self.Dataset["Y"]).to(self.device)
        F = self.mapping(t).to(self.device)
        Post = -(Y * torch.log(1-torch.sigmoid(F)) + (1-Y)*torch.log(torch.sigmoid(F))).sum()
        return Post
        #return self.forward(t).sum()
    
    def optimise(self, t, niters):
        for i in range(niters):
            self.optimizer.zero_grad()
            t2 = self.first_layer(t)
            t2 = t2.flatten()
            loss = self.sum_forward(t2)
            loss.backward()
            self.optimizer.step()
            if i % 100 == 0:
                print(loss.item())
