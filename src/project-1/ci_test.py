
import pandas as pd
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

## amount of data
data = pd.read_csv("data.csv")




## Firstly X is independent of all else
X = data.drop(["y", "A", "foreign_A202"], axis = 1)
Y = data["y"].as_matrix()
Z = data["foreign_A202"].as_matrix()
A = data["A"].as_matrix()



## Define a function for calculating the marginal likelihood $P(D|model)$
## This gives you the probability of all the data under the prior.
## Now you can use this to calculate a posterior over two possible beta priors.
## $P(model | D) = P(D | model) P(model) / sum_i P(D | model_i) P(model_i)$
def marginal_posterior(data, alpha, beta):
    n_data = data.shape[0]
    total_probability = 1
    log_probability = 0
    for t in range(n_data):
        p = alpha / (alpha + beta)
        if (data[t] > 0):
            #total_probability *= p
            log_probability += np.log(p)
            alpha += 1
        else:
            #total_probability *= (1 - p)
            log_probability += np.log(1 - p)
            beta +=1
    return np.exp(log_probability)

            
## Now measure the distribution of A for each value of Y, for different values of Z
##
n_figures = 0
for y in [1, 2]:
    ## P(A | Y, Z = 1)
    positive = (Y==y) & (Z==1)
    positive_alpha = sum(A[positive]==1)
    positive_beta = sum(A[positive]==0)
    positive_ratio = positive_alpha / (positive_alpha + positive_beta)


    ## P(A | Y, Z = - 1)
    negative = (Y==y) & (Z==0)
    negative_alpha = sum(A[negative]==1)
    negative_beta = sum(A[negative]==0)
    negative_ratio = negative_alpha / (negative_alpha + negative_beta)
    

    print("y: ", y, "Deviation: ", abs(positive_ratio - negative_ratio))

    print ("Calculate the marginals for each model")
    P_D_positive = marginal_posterior(A[positive], 1, 1)
    P_D_negative = marginal_posterior(A[negative], 1, 1)
    P_D = marginal_posterior(A[(Y==y)], 1, 1)
    
    
    print("Marginal likelihoods: ", P_D, P_D_negative, P_D_positive)
    ## Now you need to calculate the probability of either the
    ## dependent or independent model by combining all of the above
    ## into a single number.  This is not completely trivial, as you
    ## need to combine the negative and positive Z into it, but I
    ## think you can all work it out.
    
    
    print ("Now calculate a posterior distribution for the relevant Bernoulli parameter. Focus on just one value of y for simplicity")

    
    # First plot the joint distribution
    prior_alpha = 1
    prior_beta = 1
    xplot = np.linspace(0, 1, 200)
    pdf_p = beta.pdf(xplot, prior_alpha + positive_alpha, prior_beta + positive_beta)
    pdf_n = beta.pdf(xplot, prior_alpha + negative_alpha, prior_beta + negative_beta)
    pdf_m = beta.pdf(xplot, prior_alpha + positive_alpha + negative_alpha, prior_beta + positive_beta + negative_beta)
    n_figures+=1
    plt.figure(n_figures)
    plt.clf()
    plt.plot(xplot, pdf_p)
    plt.plot(xplot, pdf_n)
    plt.plot(xplot, pdf_m) 
    plt.legend(["z=1", "z=-1", "marginal"])
    plt.title("y=" + str(y))
    
#plt.show()
