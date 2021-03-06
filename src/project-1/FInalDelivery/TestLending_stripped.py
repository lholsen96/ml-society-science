# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 19:51:44 2018

@author: Lars
"""

import pandas
from statsmodels.formula.api import Logit, logit, mnlogit

## Set up for dataset
features = ['checking account balance', 'duration', 'credit history',
            'purpose', 'amount', 'savings', 'employment', 'installment',
            'woman', 'other debtors', 'residence time',
            'property', 'age', 'other installments', 'housing', 'credits',
            'job', 'persons', 'phone', 'foreign']
target = 'repaid'
df = pandas.read_csv('german-manwoman.data', sep=' ',
                     names=features+[target])

import matplotlib.pyplot as plt
numerical_features = ['duration', 'age', 'residence time', 'installment', 'amount', 'duration', 'persons', 'credits']
quantitative_features = list(filter(lambda x: x not in numerical_features, features))
X = pandas.get_dummies(df, columns=quantitative_features, drop_first=True)
encoded_features = list(filter(lambda x: x != target, X.columns))

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
            if (good_loan != 1):
                utility -= amount
            else:    
                utility += amount*(pow(1 + interest_rate, duration) - 1)
    return utility


## Main code


### Setup model
from sklearn import linear_model
logistic = linear_model.LogisticRegression()

import NameBanker

decision_maker = NameBanker.NameBanker(logistic)

interest_rate = 0.005

### Do a number of preliminary tests by splitting the data in parts
from sklearn.model_selection import train_test_split
n_tests = 1
utility = 0
for iter in range(n_tests):
    X_train, X_test, y_train, y_test = train_test_split(X[encoded_features], X[target], test_size=0.5,random_state = 314159)
    decision_maker.set_interest_rate(interest_rate)
    decision_maker.fit(X_train, y_train)
    
    a = []
    for i in range(X_test.shape[0]):
        a.append(decision_maker.get_best_action(X_test.iloc[i,:]))
    
    utility += test_decision_maker(X_test, y_test, interest_rate, decision_maker)

#print(utility / n_tests)
se = pandas.Series(a)
X_test["A"] = se.values
X_test["y"] = y_test.values

print ("Number of people granted loan: %d" % sum(a))
X_test.to_csv("data_manwoman3.csv")

data_manwoman = pandas.read_csv("data_manwoman3.csv")

relative_counts_manwoman = pandas.DataFrame(
    {i: d.A.value_counts() / d.A.count() for i, d in data_manwoman.groupby(['woman_A92', 'y'])})

print(relative_counts_manwoman)

relative_counts_manwoman.plot.bar().legend(bbox_to_anchor = (1,1))
