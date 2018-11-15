import numpy as np
import pandas
def default_reward_function(action, outcome):
    return outcome

# This is the reward function given to us in the second part of the project
def reward_function(action, outcome):
    return -0.1*action + outcome


def test_policy(generator, policy, reward_function, T):
    policy.set_reward(reward_function)
    
    # Set 
    #np.random.seed(314159)
    #np.random.seed(123)
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
        print("Iteration: %4d \t Current mean reward: %7.4f" %(t, u/(t+1)))
    return u

features = pandas.read_csv('historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv('historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv('historical_Y.dat', header=None, sep=" ").values
observations = features[:, :128]
labels = features[:,128] + features[:,129]*2

import data_generation
generator = data_generation.DataGenerator()


# We import the different recommenders
# Initialize an instance of each of them
# Fit the policy on historical data

import LogisticRecommenderBoot
policy_factory_logistic_boot = LogisticRecommenderBoot.LogisticRecommenderBoot
policy_logistic_boot = policy_factory_logistic_boot(generator.get_n_actions(), generator.get_n_outcomes(),0)
policy_logistic_boot.fit_treatment_outcome(features, actions, outcome)

# Run an online test with the same number of actions
n_tests = 1000
# Set seed for reproducibility
#np.random.seed(314159)


# Check the other policies
result_logistic_boot = test_policy(generator, policy_logistic_boot, default_reward_function, n_tests)
print("Logistic boot       %7.4f" % result_logistic_boot)




#policy.final_analysis()