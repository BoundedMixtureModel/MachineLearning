'''
This is an implementation of the Metropolis Hastings algorithm.

the target distribution is asymmetric gaussian distribution which is introduced by
"Background subtraction using finite mixtures of asymmetric Gaussian distributions and shadow detection" by Tarek Elguebaly,
Nizar Bouguila
'''
import numpy as np
from matplotlib.pylab import *
from scipy.stats import norm
import matplotlib.pyplot as plt


def Asymmetric_Gassian_Distribution(xjk, mu_jk, s_ljk, s_rjk):
    if xjk < mu_jk:
        return np.sqrt(2/np.pi)/(np.power(s_ljk, -0.5) + np.power(s_rjk, -0.5)) * np.exp(- 0.5 * s_ljk * (xjk- mu_jk)**2)
    else:
        return np.sqrt(2/np.pi)/(np.power(s_ljk, -0.5) + np.power(s_rjk, -0.5)) * np.exp(- 0.5 * s_rjk * (xjk- mu_jk)**2)


def Metropolis_Hastings_Sampling_AGD(mu_jk, s_ljk, s_rjk):
    n = 25000
    x = norm.rvs(0, 2.5, 1)[0]
    vec = []
    vec.append(x)
    for i in range(n):
        # proposed distribution make sure 25%-40% accept
        # random_walk algorithm, using symmetric Gaussian distribution, so it's simplified to Metropolis algoritm
        # the parameter is mu: the previous state of x and variation
        candidate = norm.rvs(x, 2.5, 1)[0]
        # acceptance probability
        alpha = min([1., Asymmetric_Gassian_Distribution(candidate, mu_jk, s_ljk, s_rjk)/
                    Asymmetric_Gassian_Distribution(x, mu_jk, s_ljk, s_rjk)])
        u = np.random.uniform(0,1)
        if u < alpha:
            x = candidate
            vec.append(x)
    return vec


vec = Metropolis_Hastings_Sampling_AGD(0,1,3)
print(len(vec))

#plotting the results:
#theoretical curve
x = np.arange(-3,3,.1)
y = [Asymmetric_Gassian_Distribution(x_value,0,1,3) for x_value in x]
subplot(211)
title('Metropolis-Hastings')
plot(vec)
subplot(212)

hist(vec, bins=30, normed=1)
plot(x,y,'ro')
ylabel('Frequency')
xlabel('x')
legend(('PDF','Samples'))
show()