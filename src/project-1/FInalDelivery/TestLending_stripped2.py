# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 13:18:02 2018

@author: Lars
"""

import pandas

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

    #Make test set with opposite value of foreign_A202, everything else the same. 
    X_test_opposite_foreign = X_test.copy() 
    X_test_opposite_foreign["foreign_A202"] = [1 if v == 0 else 0 for v in X_test_opposite_foreign["foreign_A202"]]    


    decision_maker.set_interest_rate(interest_rate)


    decision_maker.fit(X_train, y_train)
    
    a = []
    a_opposite_foreign = []

    for i in range(X_test.shape[0]):
        a.append(decision_maker.get_best_action(X_test.iloc[i,:]))
        a_opposite_foreign.append(decision_maker.get_best_action(X_test_opposite_foreign.iloc[i,:]))
    
    utility += test_decision_maker(X_test, y_test, interest_rate, decision_maker)

#print(utility / n_tests)
se = pandas.Series(a)
se_opposite_foreign = pandas.Series(a_opposite_foreign)

X_test["A"] = se.values
X_test["y"] = y_test.values

X_test_opposite_foreign["A"] = se_opposite_foreign.values
X_test_opposite_foreign["y"] = y_test.values





#print(X_test)
X_test.to_csv("data10.csv")
X_test_opposite_foreign.to_csv("data_opposite_foreign10.csv")

df = pandas.read_csv("data10.csv")
relative_counts = pandas.DataFrame(
    {i: d.A.value_counts() / d.A.count() for i, d in df.groupby(['foreign_A202', 'y'])})
print(relative_counts)

relative_counts.plot.bar().legend(bbox_to_anchor = (1,1))


data = pandas.read_csv("data10.csv")
data_switch = pandas.read_csv("data_opposite_foreign10.csv")

# Calculate the variation
variation = sum(abs(data["A"].subtract(data_switch["A"])))
print("Variation: %d" % variation)

# The direction of the varation
variation_2 = sum(data["A"].subtract(data_switch["A"]))
print("Direction: %d" % variation_2)