# coding: utf-8
# In[1]:


import pandas


# In[2]:


features = ['checking account balance', 'duration', 'credit history',
            'purpose', 'amount', 'savings', 'employment', 'installment',
            'marital status', 'other debtors', 'residence time',
            'property', 'age', 'other installments', 'housing', 'credits',
            'job', 'persons', 'phone', 'foreign']
target = 'repaid'


# Data taken from https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29

# In[3]:


df = pandas.read_csv('../../data/credit/german.data', sep=' ',
                     names=features+[target])


# In[4]:


numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'duration', 'persons', 'credits']
quantitative_features = list(filter(lambda x: x not in numerical_features, features))


# In[5]:


X = pandas.get_dummies(df, columns=quantitative_features, drop_first=True)


# In[6]:


encoded_features = list(filter(lambda x: x != target, X.columns))


# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer


# In[8]:


cv_accuracy = cross_val_score(LogisticRegression(),
                X[encoded_features],
                X[target],
                scoring=make_scorer(accuracy_score),
                cv=10)


# In[9]:


#cv_accuracy.mean(), cv_accuracy.std()


# # Add noise by flipping random indicators

# In[10]:


[a ^ b for a, b in zip((0,1,0,1), (0,0, 1,1))]


# In[11]:
import numpy as np
import matplotlib.pyplot as plt


k = 2 
epsilons = np.array((0.0001, 0.001, 0.01, 0.1, 1, 10,  100, 1000))
cv_accuracy_noises = np.zeros(len(epsilons))

i = 0 
for epsilon in epsilons: 


    flip_fraction = 1/ (1  + np.exp(epsilon/k)) # All non-numerical features are binary. What fraction should we flip to be epsilon-DP, given that there are k attributes in total? If we want to be epsilon-DP overall, then each feature should be \epsilon / k - DP.


    # In[12]:


   

    # In[13]:


    X_noise = X.copy()
    X_noise.iloc[0].loc['duration']


    # In[14]:

    # Since flip = 1 / flip
    n_data = len(X_noise)
    for t in range(n_data):
        for c in X_noise.columns:
            # We can use the same random response mechanism for all binary features
            if any(c.startswith(i) for i in quantitative_features):
                w = np.random.choice([0, 1], p=[1 - flip_fraction, flip_fraction])
                X_noise.loc[t,c] = (X_noise.loc[t,c] + w) % 2

            # For numerical features, it is different. The scaling factor should depend on k, \epsilon, and the sensitivity of that particular attribite. In this case, it's simply the range of the attribute.
            if any(c.startswith(i) for i in numerical_features):
                # calculate the range of the attribute and add the laplace noise to the original data
                M = np.max(X.loc[t,c]) - np.min(X.loc[t,c])
                l = M*k/(epsilon)
                w = np.random.laplace(0, l)
                X_noise.loc[t,c] += w
              


                


    # In[15]:
    a = cross_val_score(LogisticRegression(),
                    X_noise[encoded_features],
                    X_noise[target],
                    scoring=make_scorer(accuracy_score),
                    cv=10)


    cv_accuracy_noises[i] = (cross_val_score(LogisticRegression(),
                    X_noise[encoded_features],
                    X_noise[target],
                    scoring=make_scorer(accuracy_score),
                    cv=10)).mean()
    i += 1

    # In[16]:
    #print ("epsilon = %f" %(epsilon))

    #print("     noise mean score: " )
    #print (cv_accuracy_noise.mean())
    #print("     difference between actual and noisy")
    #print(cv_accuracy.mean() - cv_accuracy_noise.mean())
    #print("     standard deviation of noisy cv- accuracy")
    #print (cv_accuracy_noise.std())


    # In[17]:
plt.plot(np.log(epsilons), cv_accuracy_noises)
plt.show()



