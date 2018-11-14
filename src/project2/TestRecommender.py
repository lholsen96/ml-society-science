import pandas

features = pandas.read_csv('historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv('historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv('historical_Y.dat', header=None, sep=" ").values

import NNRecommender
policy_factory_NN = NNRecommender.NNRecommender
policy_NN = policy_factory_NN(2, 2)

import LogisticRecommender
policy_factory_logistic = LogisticRecommender.LogisticRecommender
policy_logistic = policy_factory_logistic(2, 2)



def reward_function(action, outcome):
    return -0.1*action + outcome

# First we need to set the reward
policy_NN.set_reward(reward_function)
policy_logistic.set_reward(reward_function)

## Fit the policy on historical data 
policy_NN.fit_treatment_outcome(features, actions, outcome)
policy_logistic.fit_treatment_outcome(features, actions, outcome)

# Utility of the historical policy. Deterministic, so no difference if we call with policy_NN or policy_logistic
utility_hist = policy_NN.estimate_utility(features, actions, outcome)

# Utility of new policy
utility_new_policy_NN = policy_NN.estimate_utility(features, None, None, policy_NN)
utility_new_policy_logistic = policy_logistic.estimate_utility(features, None, None, policy_logistic)


# Utility of improved policy, using a neural network trained on the historical data
print("The historical utility was %2f" %(utility_hist))
print("The estimated utility of the improved NN policy is: %2f " % (utility_new_policy_NN))
print("The estimated utility of the improved logistic policy is: %2f"  % (utility_new_policy_logistic))

