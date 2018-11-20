# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:40:59 2018

@author: Lars



import HistoricalRecommender
policy_factory_historical = HistoricalRecommender.HistoricalRecommender
res = np.zeros(1000)
for i in range(1000):
    policy_historical = policy_factory_historical(generator.get_n_actions(), generator.get_n_outcomes())
    policy_historical.set_reward(reward_function)
    policy_historical.fit_treatment_outcome(features, actions, outcome)
    res[i] = test_policy(generator, policy_historical, reward_function, n_tests)[0]
"""

import numpy as np   
import LogisticRecommender
import pandas

import data_generation
generator = data_generation.DataGenerator()

features = pandas.read_csv('historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv('historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv('historical_Y.dat', header=None, sep=" ").values

def reward_function(action, outcome):
    # CHanged the reward. Assume all medicins have the same cost/must be 10%
    # more efficient than placebo.
    if (action >= 1): 
        return -0.1 + outcome
    else: 
        return outcome
   

def test_policy(generator, policy, reward_function, T):
    policy.set_reward(reward_function)
    u = 0
    
    # Not placebo
    total_given_treatment = 0
    for t in range(T):
        
        # We generate 1 new pasient
        x = generator.generate_features()
        # Find the best action, from our model
        a = policy.recommend(x)

        if (a != 0): total_given_treatment += 1
        # Generate the outcome based on user_data and action
        y = generator.generate_outcome(x, a)
        #print(a)
        # Add the utility/reward
        u += reward_function(a, y)
        # Let the policy now about the result
        # Refit the model.
        policy.observe(x, a, y)
        #print(a)
        #print("x: ", x, "a: ", a, "y:", y, "r:", r)
        #print("Iteration: %4d \t Current mean reward: %7.4f" %(t, u/(t+1)))
    return [u, total_given_treatment] 

n_tests = 500
policy_factory_logistic = LogisticRecommender.LogisticRecommender
policy_logistic_no_update = policy_factory_logistic(generator.get_n_actions(), generator.get_n_outcomes(), 0)
res = np.zeros(100)
for i in range(100):
    policy_logistic_no_update = policy_factory_logistic(generator.get_n_actions(), generator.get_n_outcomes(), 0)
    policy_logistic_no_update.set_reward(reward_function)
    policy_logistic_no_update.fit_treatment_outcome(features, actions, outcome)
    res[i] = test_policy(generator, policy_logistic_no_update, reward_function, n_tests)[0]
np.percentile(res, 2.5)
np.percentile(res, 97.5)