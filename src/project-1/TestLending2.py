#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import pandas
import numpy as np
import matplotlib.pyplot as plt

# We use seaborn to get better looking plots
import seaborn as sb
sb.set()

# Import SKlearn models
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
#from sklearn.neural_network import MLPClassifier

# Import our policies/models
import random_banker
import NameBankerCVKNN
import NameBanker

## Test function
def test_decision_maker(X_test, y_test, interest_rate, decision_maker):
    
    n_test_examples = len(X_test)
    utility = 0
    ## Example test function - this is only an unbiased test if the data has not been seen in training
    for t in range(n_test_examples):
        action = decision_maker.get_best_action(X_test.iloc[t])
        good_loan = y_test.iloc[t] # assume the labels are correct
        duration = X_test['duration'].iloc[t]
        amount = X_test['amount'].iloc[t]
        # If we don't grant the loan then nothing happens
        if (action==1):
            if (good_loan == 2):
                utility -= amount
            else:    
                utility += amount*(pow(1 + interest_rate, duration) - 1)
    return utility

### Function that adds noise to the data
### The code is based on what we dd in lab.
def create_noisy_data(X, epsilon, k, numerical_features, categorical_features):
    
    # We find the flipfraction by solving \epsilon = \ln\frac{1-p}{p} with respect to p
    flip_fraction = 1 / (1 + np.exp(epsilon/k)) 
    
    # Make a copy of the X table
    X_noise = X.copy()
    
    # Find the number of entries
    n = len(X_noise)
    
    # We can use the same random response mechanism for all categorical features
    w = np.random.choice([0, 1], 
                         size=(n, len(categorical_features)), 
                         p=[1 - flip_fraction, flip_fraction])
    X_noise[categorical_features] = (X_noise[categorical_features] + w) % 2
    
    # For the quantitative features, the noise depends on k, \epsilon, 
    # and the sensitivity of the given attribute. This is just the range of
    # the values of attribute.
    for c in numerical_features:
        M = (X[c].max()-X[c].min())
        l = (M*epsilon)/k
        w = np.random.laplace(scale=l, size=n)
        X_noise[c] += w
    return X_noise


features = ['checking account balance', 'duration', 'credit history',
            'purpose', 'amount', 'savings', 'employment', 'installment',
            'marital status', 'other debtors', 'residence time',
            'property', 'age', 'other installments', 'housing', 'credits',
            'job', 'persons', 'phone', 'foreign']
target = 'repaid'

df = pandas.read_csv('german.data', sep=' ', names=features+[target])
numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'duration', 'persons', 'credits']
categorical_features = list(filter(lambda x: x not in numerical_features, features))

X = pandas.get_dummies(df, columns=categorical_features, drop_first=True)
encoded_features = list(filter(lambda x: x != target, X.columns))


# Create the models
bagging = BaggingClassifier(KNeighborsClassifier(), n_estimators=10)
knn = KNeighborsClassifier()
logistic = linear_model.LogisticRegression()


n_tests = 10
interest_rate = 0.005

n_eps = 20
epsilon = np.logspace(-2, 2, n_eps)
categorical_features = [f for f in encoded_features if f not in numerical_features]
accuracy_std = np.zeros(n_eps)
accuracy_mean = np.zeros(n_eps)

utility_RB = np.zeros(n_eps)
utility_CVKNN = np.zeros(n_eps)
utility_CVKNN_no_noise = np.zeros(n_eps)
utility_log = np.zeros(n_eps)

# Create the models
decision_maker_RB = random_banker.RandomBanker()
decision_maker_CVKNN = NameBankerCVKNN.NameBankerCVKNN()
decision_maker_log = NameBanker.NameBanker(logistic)

# We set the interest rates
decision_maker_RB.set_interest_rate(interest_rate)
decision_maker_CVKNN.set_interest_rate(interest_rate)
decision_maker_log.set_interest_rate(interest_rate)

index = 0
for eps in epsilon:
    print("Calculating eps number: ", index+1)
    for iter in range(n_tests):
        # Split the data in training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.2)
        
        # We fit the models. Trained on the         
        decision_maker_RB.fit(X_train, y_train)
        decision_maker_CVKNN.fit(X_train, y_train)
        decision_maker_log.fit(X_train, y_train)
        
        # We add noise to features of the test data set
        X_test_noise = create_noisy_data(X_test, epsilon=eps, k=len(encoded_features),
                                numerical_features=numerical_features,
                                categorical_features=[f for f in encoded_features if f not in numerical_features])
        
        utility_RB[index] += test_decision_maker(X_test_noise, y_test, interest_rate, decision_maker_RB)
        utility_CVKNN[index] += test_decision_maker(X_test_noise, y_test, interest_rate, decision_maker_CVKNN)
        utility_log[index] += test_decision_maker(X_test_noise, y_test, interest_rate, decision_maker_log)
        
        utility_CVKNN_no_noise[index] += test_decision_maker(X_test, y_test, interest_rate, decision_maker_CVKNN)
        
    index += 1
    

        
plt.plot(epsilon, utility_RB, epsilon, utility_CVKNN,epsilon, utility_CVKNN_no_noise, epsilon, utility_log)
plt.xlabel("Epsilon")
plt.ylabel("Utility")

