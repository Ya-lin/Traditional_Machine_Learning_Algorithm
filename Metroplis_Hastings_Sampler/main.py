# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 22:20:59 2024

@author: yalin
"""

import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import uniform
from scipy.stats import multivariate_normal

np.random.seed(42)


#%% 
class mh_gaussian:
    
    def __init__(self, dim, K, num_samples, 
                 target_mu, target_sigma, target_pi, 
                 proposal_mu, proposal_sigma):
        self.dim = dim
        self.K = K
        self.num_samples = num_samples
        self.target_mu = target_mu
        self.target_sigma = target_sigma
        self.target_pi = target_pi
        
        self.proposal_mu = proposal_mu
        self.proposal_sigma = proposal_sigma
        
        self.n_accept = 0
        self.mh_samples = np.zeros((self.num_samples, self.dim))
        
    def target_pdf(self, x):
        prob = 0
        for k in range(self.K):
            prob += self.target_pi[k]*multivariate_normal.pdf(x,
                    self.target_mu[:,k],self.target_sigma[:,:,k])
        return prob

    def sample(self):
        x_init = multivariate_normal.rvs(self.proposal_mu, self.proposal_sigma, 1)
        self.mh_samples[0,:] = x_init
        
        for i in range(self.num_samples-1):
            x_curr = self.mh_samples[i,:]
            x_new = multivariate_normal.rvs(x_curr, self.proposal_sigma, 1)
            alpha = self.target_pdf(x_new)/self.target_pdf(x_curr)
            r = min(1, alpha)
            u = uniform.rvs(loc=0, scale=1, size=1)
            ar = u<r
            self.mh_samples[i+1,:] = ar*x_new + (1-ar)*x_curr
            self.n_accept += ar
        
        print(f"MH acceptance rate: {self.n_accept/self.num_samples}")


#%%
if __name__ =="__main__":
    
    dim= 2
    K = 2
    num_samples = 6000
    target_mu = np.zeros((dim, K))
    target_mu[:,0] = [4,0]
    target_mu[:,1] = [-4,0]
    target_sigma = np.zeros((dim, dim, K))
    target_sigma[:,:,0] = [[2,1],[1,1]]
    target_sigma[:,:,1] = [[1,0],[0,1]]
    target_pi = np.array([0.4, 0.6])
    
    proposal_mu = np.zeros((dim, 1)).flatten()
    proposal_sigma = 10*np.eye(dim)
    
    mhg = mh_gaussian(dim, K, num_samples, target_mu, target_sigma, 
                      target_pi, proposal_mu, proposal_sigma)
    mhg.sample()

    fig = plt.figure()
    plt.scatter(mhg.mh_samples[:,0], mhg.mh_samples[:,1], label='MH samples')
    plt.grid(True)
    plt.legend()
    plt.title("Metropolis-Hastings Sampling of 2D Gaussian Mixture")
    plt.xlabel("d1")
    plt.ylabel("d2")
    plt.show()
    plt.close()
    fig.savefig("mh_sample.png")
   
