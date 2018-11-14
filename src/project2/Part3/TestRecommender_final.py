import numpy as np
import pandas
def default_reward_function(action, outcome):
    return outcome

# This is the reward function given to us in the second part of the project
def reward_function(action, outcome):
    return -0.1*action + outcome


def test_policy(generator, policy, reward_function, T):
    policy.set_reward(reward_function)
    u = 0
    for t in range(T):
        # We generate 1 new pasient
        x = generator.generate_features()
        # Find the best action, from our model
        a = policy.recommend(x)
        # Generate the outcome based on user_data and action
        y = generator.generate_outcome(x, a)
        # Add the utility/reward
        u += reward_function(a, y)
        # Let the policy now about the result
        # Refit the model.
        policy.observe(x, a, y)
        #print(a)
        #print("x: ", x, "a: ", a, "y:", y, "r:", r)
        print("Iteration: %4d \t Current mean reward: %7.4f\n" %(t, u/(t+1)))
    return u

features = pandas.read_csv('historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv('historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv('historical_Y.dat', header=None, sep=" ").values
observations = features[:, :128]
labels = features[:,128] + features[:,129]*2

import data_generation
generator = data_generation.DataGenerator()

import LogisticRecommender
policy_factory = LogisticRecommender.LogisticRecommender
#import NNRecommender
#policy_factory = NNRecommender.NNRecommender
import HistoricalRecommender
policy_factory = HistoricalRecommender.HistoricalRecommender
#import reference_recommender
#policy_factory = reference_recommender.RandomRecommender
policy = policy_factory(generator.get_n_actions(), generator.get_n_outcomes())


## Fit the policy on historical data first
policy.fit_treatment_outcome(features, actions, outcome)

## Run an online test with the same number of actions
n_tests = 500
result = test_policy(generator, policy, reward_function, n_tests)
print("Total reward:", result)

#policy.final_analysis()