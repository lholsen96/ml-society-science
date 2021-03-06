import numpy as np
import pandas
def default_reward_function(action, outcome):
    return outcome

# This is the reward function given to us in the second part of the project
def reward_function(action, outcome):
    # CHanged the reward. Assume all medicins have the same cost/must be 10%
    # more efficient than placebo.
    if (action >= 1): 
        return -0.1 + outcome
    else: 
        return outcome
    
#def reward_function(action, outcome):
#    return -0.1*action + outcome
   


def test_policy(generator, policy, reward_function, T):
    policy.set_reward(reward_function)
    
    # Set 
    #seed_number = 3567
    #seed_number = 367482
    #seed_number = 367
    #seed_number = 150
    #seed_number = 157157
    #seed_number = 199618
    #seed_number = 100018
    #seed_number = 530018
    seed_number = 43018
    seed_number = 73018
    seed_number = 93018
    seed_number = 113018
    seed_number = 153018
    seed_number = 143018
    seed_number = 177700
    seed_number = 197700
    seed_number = 217700
    seed_number = 317700
    seed_number = 337700
    seed_number = 357700
    seed_number = 457700
    
    
    
    u = 0
    
    # Not placebo
    total_given_treatment = 0
    for t in range(T):
        np.random.seed(seed_number + t)
        
        # We generate 1 new pasient
        x = generator.generate_features()
        # Find the best action, from our model
        a = policy.recommend(x)

        if (a != 0): total_given_treatment += 1
        # Generate the outcome based on user_data and action
        y = generator.generate_outcome(x, a)
        print(a)
        # Add the utility/reward
        u += reward_function(a, y)
        # Let the policy now about the result
        # Refit the model.
        policy.observe(x, a, y)
        #print(a)
        #print("x: ", x, "a: ", a, "y:", y, "r:", r)
        #print("Iteration: %4d \t Current mean reward: %7.4f" %(t, u/(t+1)))
    return [u, total_given_treatment] 

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
import HistoricalRecommender
policy_factory_historical = HistoricalRecommender.HistoricalRecommender
policy_historical = policy_factory_historical(generator.get_n_actions(), generator.get_n_outcomes())
policy_historical.fit_treatment_outcome(features, actions, outcome)

import LogisticRecommender
policy_factory_logistic = LogisticRecommender.LogisticRecommender
policy_logistic = policy_factory_logistic(generator.get_n_actions(), generator.get_n_outcomes())
policy_logistic.fit_treatment_outcome(features, actions, outcome)

# We allow no update of the policy. THis in Exercise 3 subquestion 2
policy_logistic_no_update = policy_factory_logistic(generator.get_n_actions(), generator.get_n_outcomes(), 0)
policy_logistic_no_update.fit_treatment_outcome(features, actions, outcome)

import LogisticRecommenderBoot
policy_factory_logistic_boot = LogisticRecommenderBoot.LogisticRecommenderBoot
policy_logistic_boot = policy_factory_logistic_boot(generator.get_n_actions(), generator.get_n_outcomes())
policy_logistic_boot.fit_treatment_outcome(features, actions, outcome)

# We allow no update of the policy. THis in Exercise 3 subquestion 2
policy_logistic_boot_no_update = policy_factory_logistic_boot(generator.get_n_actions(), generator.get_n_outcomes(),0)
policy_logistic_boot_no_update.fit_treatment_outcome(features, actions, outcome)

import LogisticRecommenderForce
policy_factory_logistic_force = LogisticRecommenderForce.LogisticRecommenderForce
policy_logistic_force = policy_factory_logistic_force(generator.get_n_actions(), generator.get_n_outcomes(), force_alt_med = 0, force_action = 2)
policy_logistic_force.fit_treatment_outcome(features, actions, outcome)


import NNRecommender
policy_factory_NN = NNRecommender.NNRecommender
policy_NN = policy_factory_NN(generator.get_n_actions(), generator.get_n_outcomes())
policy_NN.fit_treatment_outcome(features, actions, outcome)

# We allow no update of the policy. THis in Exercise 3 subquestion 2
policy_NN_no_update = policy_factory_NN(generator.get_n_actions(), generator.get_n_outcomes(), 0)
policy_NN_no_update.fit_treatment_outcome(features, actions, outcome)


import NNRecommenderForce
policy_factory_NN_force = NNRecommenderForce.NNRecommenderForce
policy_NN_force = policy_factory_NN_force(generator.get_n_actions(), generator.get_n_outcomes(), force_alt_med = 0, force_action = 2)
policy_NN_force.fit_treatment_outcome(features, actions, outcome)


# For comparision we want to look at the extreme cases
# Where always use placebo. Always use drug 1 and random
import FixedTreatmentRecommender
import RandomRecommender
policy_random = RandomRecommender.RandomRecommender(generator.get_n_actions(), generator.get_n_outcomes())

# Here we look at the clustered versions of the two classes above
# Currently wirght now they are hardcoded to support 2 clusters,
# which is the number of clusters we were told that there were.
import LogisticRecommenderCluster
policy_factory_logistic_cluster = LogisticRecommenderCluster.LogisticRecommenderCluster
policy_logistic_cluster = policy_factory_logistic_cluster(generator.get_n_actions(), generator.get_n_outcomes())
policy_logistic_cluster.fit_treatment_outcome(features, actions, outcome)

import NNRecommenderCluster
policy_factory_NN_cluster = NNRecommenderCluster.NNRecommenderCluster
policy_NN_cluster = policy_factory_NN_cluster(generator.get_n_actions(), generator.get_n_outcomes())
policy_NN_cluster.fit_treatment_outcome(features, actions, outcome)

import CheatRecommender
policy_factory_cheat = CheatRecommender.CheatRecommender
policy_cheat = policy_factory_cheat(generator.get_n_actions(), generator.get_n_outcomes())
policy_cheat.fit_treatment_outcome(features, actions, outcome)



# Run an online test with the same number of actions
n_tests = 100
# Set seed for reproducibility
#np.random.seed(314159)

# Check what happens when we always stick to one medicin, try all of them
results_fixed_treatment = np.zeros((generator.get_n_actions(), 2))
for i in range(len(results_fixed_treatment)):
    policy_temp = FixedTreatmentRecommender.FixedTreatmentRecommender(generator.get_n_actions(), generator.get_n_outcomes(), i)
    results_fixed_treatment[i] = test_policy(generator, policy_temp, reward_function, n_tests)

# Let's only look at the ten best gene treatments
best_indices = np.argsort(-results_fixed_treatment[:,0])[:10]

# Check what happens when we chose a randomly between placebo and drug1 and drug2
result_random = test_policy(generator, policy_random, reward_function, n_tests)

# Check the other policies
result_historical = test_policy(generator, policy_historical, reward_function, n_tests)

result_logistic = test_policy(generator, policy_logistic, reward_function, n_tests)
result_logistic_no_update = test_policy(generator, policy_logistic_no_update, reward_function, n_tests)
result_logistic_cluster = test_policy(generator, policy_logistic_cluster, reward_function, n_tests)
result_logistic_boot = test_policy(generator, policy_logistic_boot, reward_function, n_tests)
result_logistic_boot_no_update = test_policy(generator, policy_logistic_boot_no_update, reward_function, n_tests)
result_logistic_force = test_policy(generator, policy_logistic_force, reward_function, n_tests)


result_NN = test_policy(generator, policy_NN, reward_function, n_tests)
result_NN_no_update = test_policy(generator, policy_NN_no_update, reward_function, n_tests)
result_NN_cluster = test_policy(generator, policy_NN_cluster, reward_function, n_tests)
result_NN_force = test_policy(generator, policy_NN_force, reward_function, n_tests)

result_cheat = test_policy(generator, policy_cheat, reward_function, n_tests)

print("Total rewards (max is %d):" %n_tests)
for i in best_indices:
    print("Treatment %3d:      %7.4f\t# of Treatments: %3d" % (i, results_fixed_treatment[i,0],results_fixed_treatment[i,1]))

print("Random:             %7.4f\t# of Treatments: %3d" % (result_random[0], result_random[1]))
print("Historical:         %7.4f\t# of Treatments: %3d" % (result_historical[0], result_historical[1]))
print("Logistic:           %7.4f\t# of Treatments: %3d" % (result_logistic[0], result_logistic[1]))
print("Logistic no update: %7.4f\t# of Treatments: %3d" % (result_logistic_no_update[0], result_logistic_no_update[1]))
print("Logistic boot:      %7.4f\t# of Treatments: %3d" % (result_logistic_boot[0], result_logistic_boot[1]))
print("Logistic boot no:   %7.4f\t# of Treatments: %3d" % (result_logistic_boot_no_update[0], result_logistic_boot_no_update[1]))
print("Logistic cluster:   %7.4f\t# of Treatments: %3d" % (result_logistic_cluster[0], result_logistic_cluster[1]))
print("Logistic force:     %7.4f\t# of Treatments: %3d" % (result_logistic_force[0], result_logistic_force[1]))
print("Neural Network:     %7.4f\t# of Treatments: %3d" % (result_NN[0], result_NN[1]))
print("NN no update:       %7.4f\t# of Treatments: %3d" % (result_NN_no_update[0], result_NN_no_update[1]))
print("NN cluster:         %7.4f\t# of Treatments: %3d" % (result_NN_cluster[0], result_NN_cluster[1]))
print("NN force:           %7.4f\t# of Treatments: %3d" % (result_NN_force[0], result_NN_force[1]))
print("Cheat Recommender:  %7.4f\t# of Treatments: %3d" % (result_cheat[0], result_cheat[1]))



#policy.final_analysis()

      
