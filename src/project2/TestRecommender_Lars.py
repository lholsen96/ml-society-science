import numpy as np
import pandas
def default_reward_function(action, outcome):
    return outcome

def reward_function(action, outcome):
    return -0.1*action + outcome


def test_policy(generator, policy, reward_function, T):
    policy.set_reward(reward_function)
    u = 0
    for t in range(T):
        # We generate 1 new pasient
        x = generator.generate_features()
        #x = np.array(features[0].reshape(1,130))
        # Find the best action, from our model
        a = policy.recommend(x)
        # Generate the outcome based on user_data and action
        y = generator.generate_outcome(x, a)
        # Add the utility
        u += reward_function(a, y)
        # Let the policy now about the result
        # Refit the model.
        policy.observe(x, a, y)
        f.write("Iteration: %4d \t Current mean reward: %7.4f\n" %(t, u/(t+1)))
        print("Iteration: %4d \t Current mean reward: %7.4f\n" %(t, u/(t+1)))
    return u

features = pandas.read_csv('historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv('historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv('historical_Y.dat', header=None, sep=" ").values
observations = features[:, :8]
labels = features[:,128] + features[:,129]*2

import reference_recommender
policy_factory = reference_recommender.ReferenceRecommender
policy = policy_factory()

import random_recommender_lars
policy_factory = random_recommender_lars.RandomRecommender
policy = policy_factory(2, 2)

import data_generation
generator = data_generation.DataGenerator()

## Fit the policy on historical data first
policy.fit_treatment_outcome(features, actions, outcome)

## Run an online test
n_tests = 500
#result = test_policy(generator, policy, default_reward_function, n_tests)
f = open("idag.txt","w+")
result = test_policy(generator, policy, reward_function, n_tests)
print(result)
f.close()


"""
Det vi skal levere:
    

features = pandas.read_csv('historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv('historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv('historical_Y.dat', header=None, sep=" ").values

import random_recommender_lars
policy_factory = random_recommender_lars.RandomRecommender
policy = policy_factory(2, 2)


def reward_function(action, outcome):
    return -0.1*action + outcome

# Må sette reward først
policy.set_reward(reward_function)

## Fit the policy on historical data first
policy.fit_treatment_outcome(features, actions, outcome)

# Historisk:
policy.estimate_utility(features, actions, outcome)

# Vår model med NN
policy.estimate_utility(features, None, None, policy)
"""