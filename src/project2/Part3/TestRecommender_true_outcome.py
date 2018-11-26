import numpy as np
import pandas

from sklearn.model_selection import train_test_split
def default_reward_function(action, outcome):
    return outcome

def test_policy(generator, policy, reward_function, T):
    print("Testing for ", T, "steps")
    policy.set_reward(reward_function)
    u = 0
    for t in range(T):
        x = generator.generate_features()
        a = policy.recommend(x)
        y = generator.generate_outcome(x, a)
        r = reward_function(a, y)
        u += r
        policy.observe(x, a, y)
        #print(a)
        #print("x: ", x, "a: ", a, "y:", y, "r:", r)
    return u

features = pandas.read_csv('data/medical/historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv('data/medical/historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv('data/medical/historical_Y.dat', header=None, sep=" ").values
observations = features[:, :128]
labels = features[:,128] + features[:,129]*2


features_train, features_test, actions_train, actions_test, outcome_train, outcome_test = train_test_split(features, actions, outcome, test_size = 0.3)

import data_generation
import NNRecommender

policy_factory_NN = NNRecommender.NNRecommender
policy_NN = policy_factory_NN(2,2)

def reward(action, outcome):
    return -0.1*action + outcome


policy_NN.set_reward(reward)


policy_NN.fit_treatment_outcome(features_train, actions_train, outcome_train)
policy_NN.estimate_utility(features_test, policy = policy_NN)

new_actions = np.zeros(len(features_test), dtype = int)



for i in range(len(features_test)): 
    new_actions[i] = policy_NN.recommend(features_test[i,:].reshape(1,130))


generator = data_generation.DataGenerator()


true_outcomes = np.zeros(len(features_test), dtype  = int)
for i in range(len(features_test)):
    true_outcomes[i] = int(generator.generate_outcome(X = features_test[i,:].reshape(1,130), A  = new_actions[i]))

print(policy_NN.estimate_utility(data = features_test, actions = new_actions, outcome = true_outcomes))



#import reference_recommender
#policy_factory = reference_recommender.RandomRecommender

## First test with the same number of treatments
print("---- Testing with only two treatments ----")

print("Setting up simulator")
generator = data_generation.DataGenerator(matrices="./generating_matrices.mat")
print("Setting up policy")
policy = policy_factory(generator.get_n_actions(), generator.get_n_outcomes())
## Fit the policy on historical data first
print("Fitting historical data to the policy")
policy.fit_treatment_outcome(features, actions, outcome)
## Run an online test with a small number of actions
print("Running an online test")
n_tests = 1000
result = test_policy(generator, policy, default_reward_function, n_tests)
print("Total reward:", result)
print("Final analysis of results")
policy.final_analysis()

## First test with the same number of treatments
print("--- Testing with an additional experimental treatment and 126 gene silencing treatments ---")
print("Setting up simulator")
generator = data_generation.DataGenerator(matrices="./big_generating_matrices.mat")
print("Setting up policy")
policy = policy_factory(generator.get_n_actions(), generator.get_n_outcomes())
## Fit the policy on historical data first
print("Fitting historical data to the policy")
policy.fit_treatment_outcome(features, actions, outcome)
## Run an online test with a small number of actions
print("Running an online test")
n_tests = 1000
result = test_policy(generator, policy, default_reward_function, n_tests)
print("Total reward:", result)
print("Final analysis of results")
policy.final_analysis()




