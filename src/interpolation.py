from scipy.optimize import fsolve
import torch
from torch import nn
import numpy as np
from models import CDFNet
from scipy.interpolate import CubicSpline, make_interp_spline
import matplotlib.pyplot as plt

class spline_interpolation:
    def __init__(self, X, theta, cdf_model) -> None:
        self.X = X
        self.W = theta
        self.model = cdf_model
# Define the equation
    def equation_to_solve(self, x, y):
        return x - 1 / (1 + np.tanh(x)) - y

    def interpolate(self):

        ### First find some starting points to fit by letting g_w(t) = 1
        # Choose a value for y
        phi = self.X.dot(self.theta)
        min_phi = min(self.phi)
        max_phi = max(self.phi)
        phi_value = np.linspace(min_phi,max_phi,100)
        tt_value = []           
        # Use fsolve to find the root numerically
        for phi in phi_value:
            solution = float(fsolve(self.equation_to_solve, x0=0, args=(phi,)))
            tt_value.append(solution)
        tt_value = np.array(tt_value)

        # generate phi from t
        t_1 = torch.tensor(tt_value.astype('float32'))
        t_2 = t_1[:,None]
        g_w = self.dudt(t_2).reshape((*t_2.shape, 1)).detach().numpy().squeeze()
        F0_ut = self.mapping(t_1).detach().numpy()
        phi = tt_value - 1 / (g_w*(1 + F0_ut))

        #fit a spline interpolation
        sorted_indices = np.argsort(phi)
        phi_sorted = phi[sorted_indices]
        t_sorted = t[sorted_indices]
        print(phi_sorted)
        print(t_sorted)
        # Create a cubic spline interpolation function
        spline_interpolation = CubicSpline(phi_sorted, t_sorted)

        # Generate more points for a smoother curve
        phi_interp = np.linspace(min(phi_sorted), max(phi_sorted), 20)
        t_interp = spline_interpolation(phi_interp)

        return t_interp

        # Plot the original data and the spline interpolation
    