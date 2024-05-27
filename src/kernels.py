import numpy as np
import math
from scipy.stats import truncnorm, norm  

b_ra = 0.7
I_k = 200

def kernel_2(u):
    # return np.where(np.abs(u) <= 1, (35/12) * (1 - u**2)**3, 1e-10)
    return np.where(np.abs(u) <=1, (1-u**2)**5, 0)
    
def d_kernel_2(u):
    # return np.where(np.abs(u) <= 1, -35/2 * u * (1 - u**2)**2, 1e-10)
    return np.where(np.abs(u) <= 1, -5*(1-u**2)**4*2*u, 0)
    
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

def F_est(u, w_t, y_t):
    #f_k_est = np.where(f_k(u, w_t) < 1e-5, 1e-5, f_k(u,w_t))
    f_k_est =  f_k(u,w_t)
    return 1 - h_k(u, w_t, y_t)/f_k_est

def d_F_est(u, w_t, y_t):
    #f_k_est = np.where(f_k(u, w_t) < 1e-5, 1e-5, f_k(u,w_t))
    val = - (d_h_k(u, w_t, y_t) * f_k(u, w_t) - h_k(u, w_t, y_t) * d_f_k(u, w_t))/f_k(u, w_t)**2
    return np.where(val < 0 , 0, val)

def phi_func_est(u, w_t, y_t, phi_t):
    return u - (1-F_est(u, w_t, y_t))/d_F_est(u, w_t, y_t) + phi_t

def phi_func_known(u, phi_t):
    return u - (1-norm.cdf(u, 0, 0.3))/norm.pdf(u, 0, 0.3) + phi_t