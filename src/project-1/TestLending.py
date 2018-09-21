import pandas
import numpy as np
from sklearn.preprocessing import StandardScaler

## Set up for dataset
features = ['checking account balance', 'duration', 'credit history',
            'purpose', 'amount', 'savings', 'employment', 'installment',
            'marital status', 'other debtors', 'residence time',
            'property', 'age', 'other installments', 'housing', 'credits',
            'job', 'persons', 'phone', 'foreign']
target = 'repaid'
df = pandas.read_csv('german.data', sep=' ',
                     names=features+[target])




import matplotlib.pyplot as plt
numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'duration', 'persons', 'credits']
categorical_features = list(filter(lambda x: x not in numerical_features, features))
X = pandas.get_dummies(df, columns=categorical_features, drop_first=True)
encoded_features = list(filter(lambda x: x != target, X.columns))


def add_noise(X, numerical_features, categorical_features):
    #coinflip for categorical variables
    epsilon = 100
    k = np.shape(X)[1]

    flip_fraction = 1/ (1  + np.exp(epsilon/k))

    X_noise = X.copy()
    for t in list(X_noise.index):
         for c in X_noise.columns:
            # We can use the same random response mechanism for all binary features
            if any(c.startswith(i) for i in categorical_features):
                w = np.random.choice([0, 1], p=[1 - flip_fraction, flip_fraction])
                X_noise.loc[t,c] = (X_noise.loc[t,c] + w) % 2
            # For numerical features, it is different. The scaling factor should depend on k, \epsilon, and the sensitivity of that particular attribite. In this case, it's simply the range of the attribute.
            if any(c.startswith(i) for i in numerical_features):
                # calculate the range of the attribute and add the laplace noise to the original data
                M = np.max(X.loc[:,c]) - np.min(X.loc[:,c])
                l = M*k/(epsilon)
                w = np.random.laplace(0, l)   
                X_noise.loc[t,c] += w 

    return X_noise            




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


## Main code


### Setup model
#import logistic_banker
#decision_maker = logistic_banker.LogisticBanker()
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier


mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(16, 4, 2), random_state=1)
bagging = BaggingClassifier(KNeighborsClassifier(), n_estimators=10)
knn = KNeighborsClassifier()
logistic = linear_model.LogisticRegression()
n_tests = 10


interest_rate = 0.005


def calculate_utility(n_tests, X, encoded_features, numerical_features, categorical_features, target, decision_maker, interest_rate, noise = False):
    from sklearn.model_selection import train_test_split
    utility = 0.0
    for iter in range(n_tests):
        X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.2)
        decision_maker.set_interest_rate(interest_rate)
        decision_maker.fit(X_train, y_train)
        if noise == True: 
            X_test = add_noise(X_test, numerical_features, categorical_features)
        utility += test_decision_maker(X_test, y_test, interest_rate, decision_maker)
    return utility    


import random_banker
import NameBankerCVKNN
import NameBanker


utility_random = calculate_utility(n_tests, X, encoded_features, numerical_features, categorical_features, target, random_banker.RandomBanker(), interest_rate, noise = True)
utility_CVKNN = calculate_utility(n_tests, X, encoded_features, numerical_features, categorical_features, target, NameBankerCVKNN.NameBankerCVKNN(), interest_rate, noise = True)
utility_generic = calculate_utility(n_tests, X, encoded_features, numerical_features, categorical_features, target, NameBanker.NameBanker(logistic), interest_rate, noise = True)



print("NameBanker CV KNN: %.2f" % (utility_CVKNN/ n_tests))
print("NameBanker Generic: %.2f" % (utility_generic/ n_tests))
print("Random Banker: %.2f" % (utility_random/ n_tests))




