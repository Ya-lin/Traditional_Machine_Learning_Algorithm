

from types import SimpleNamespace
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad
from scipy.stats import multivariate_normal

np.random.seed(42)


#%%
class importance_sampler:

    def __init__(self, k=1.5, mu=0.8, sigma=np.sqrt(1.5), c=3):
        self.k = k
        self.mu = mu
        self.sigma = sigma
        self.c = c

    def target_pdf(self, x):
        return x**(self.k-1)*np.exp(-x**2/2)

    def proposal_pdf(self, x):
        scalor = 1/np.sqrt(2*np.pi*1.5)
        unnormalized = np.exp(-(x-self.mu)**2/(2*self.sigma**2))
        return self.c*scalor*unnormalized

    def fx(self, x):
        return 2*np.sin(np.pi*x/1.5)

    def sample(self, num_samples):
        x = multivariate_normal.rvs(self.mu, self.sigma, num_samples)
        idx = np.where(x>=0)
        x_pos = x[idx]

        isw = self.target_pdf(x_pos)/self.proposal_pdf(x_pos)
        fw = isw/np.sum(isw)*self.fx(x_pos)
        f_est = np.sum(fw)
        return isw, f_est

    def true_value(self):  # numerical integration
        I_get, _ = quad(lambda x:self.fx(x)*self.target_pdf(x),0,5)
        return I_get


#%%
if __name__ == "__main__":

    num_samples = [10, 100, 1000, 10000, 100000, 1000000]

    result = SimpleNamespace(est=[], weights_var=[])
    for k in num_samples:
        IS = importance_sampler()
        IS_weights, F_est = IS.sample(k)
        result.est.append(F_est)
        IS_weights_var = np.var(IS_weights/np.sum(IS_weights))
        result.weights_var.append(IS_weights_var)

    I_get = IS.true_value()
    
    fig = plt.figure()
    xx = np.linspace(0, 8, 100)
    plt.plot(xx, IS.target_pdf(xx), '-r',
             label='target pdf p(x)')
    plt.plot(xx, IS.proposal_pdf(xx), '-b',
             label='proposal pdf q(x)')
    plt.plot(xx, IS.fx(xx)*IS.target_pdf(xx), '-k',
             label='p(x)f(x) integrand')
    plt.grid(True)
    plt.xlabel('Input')
    plt.ylabel('Function')
    plt.title('Importance Sampling Components')
    plt.legend()
    plt.show()
    plt.close()
    fig.savefig("IS_components.png")
    
    fig = plt.figure()
    plt.hist(IS_weights, label = "IS weights")
    plt.grid(True)
    plt.title("Importance Weights Histogram")
    plt.legend()
    plt.show()
    plt.close()
    fig.savefig("IS_histogram.png")

    fig = plt.figure()
    plt.semilogx(num_samples, result.est,
                 label='IS estimate of E[f(x)]')
    plt.semilogx(num_samples, I_get*np.ones(len(num_samples)),
                 label='Ground Truth')
    plt.grid(True)
    plt.xlabel('Num_samples')
    plt.ylabel('E[f(x)] estimate')
    plt.title('IS estimate of E[f(x)]')
    plt.show()
    plt.close()
    fig.savefig("IS_estimate.png")
   

     
