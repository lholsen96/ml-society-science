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


def test_policy(generator, policy, reward_function, T):
    policy.set_reward(reward_function)
    
    # Set 
    #seed_number = 3567
    #seed_number = 367482
    #seed_number = 367
    #seed_number = 180
    #seed_number = 190
    seed_number = 12345
    u = 0
    
    # Not placebo
    number_treated_gen0 = 0
    number_treated_gen1 = 0
    number_failed_gen0  = 0
    number_failed_gen1  = 0
    for t in range(T):
        np.random.seed(seed_number + t)
        
        # We generate 1 new pasient
        x = generator.generate_features()
        
        # Find the best action, from our model
        # Fixed model, always give a fixed number
        a = policy.recommend(x)
        
        # Generate the outcome based on user_data and action
        y = generator.generate_outcome(x, a)
        
        # If success or failure
        if(y == 1): 
            if (x[a] == 0):
                number_treated_gen0 += 1
            
            if (x[a] == 1):
                number_treated_gen1 += 1
        else:
            if (x[a] == 0):
                number_failed_gen0 += 1
            
            if (x[a] == 1):
                number_failed_gen1 += 1
                
        # Add the utility/reward
        u += reward_function(a, y)
        
        # Let the policy now about the result
        # Refit the model.
        policy.observe(x, a, y)
        #print(a)
        #print("x: ", x[127], "a: ", a, "y:", y, "r:", y)
        #print("Iteration: %4d \t Current mean reward: %7.4f" %(t, u/(t+1)))
    return [u, number_treated_gen0, number_treated_gen1, number_failed_gen0, number_failed_gen1] 

features = pandas.read_csv('historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv('historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv('historical_Y.dat', header=None, sep=" ").values
observations = features[:, :128]
labels = features[:,128] + features[:,129]*2

import data_generation
generator = data_generation.DataGenerator()


# For comparision we want to look at the extreme cases
# Where always use placebo. Always use drug 1 and random
import FixedTreatmentRecommender

# Run an online test with the same number of actions
n_tests = 1000
# Set seed for reproducibility
#np.random.seed(314159)

# Get the results for each medicin
results_fixed_treatment = np.zeros((generator.get_n_actions(), 5))
for i in range(len(results_fixed_treatment)):
    policy_temp = FixedTreatmentRecommender.FixedTreatmentRecommender(generator.get_n_actions(), generator.get_n_outcomes(), i)
    results_fixed_treatment[i] = test_policy(generator, policy_temp, reward_function, n_tests)

k = 20
# Let's only look at the k best gene treatments
best_indices = np.argsort(-results_fixed_treatment[:,0])[:k]

print("Total rewards (max is %d):" %n_tests)
print("Treatment  Utility  Total success: %gen0   %gen1   Total fail: %gen0   %gen1")
for i in best_indices:
    util = results_fixed_treatment[i,0]
    suc0 = results_fixed_treatment[i,1]
    suc1 = results_fixed_treatment[i,2]
    fail0 = results_fixed_treatment[i,3]
    fail1 = results_fixed_treatment[i,4]
    tot_suc = suc0 + suc1
    tot_fail = fail0 + fail1
    tot = tot_suc + tot_fail
    if (i in [0,1,2]): 
        print("%9d  %7.3f        %4d                          %3d      " % (i, util, tot_suc, tot_fail))
    else:
        percentage_suc0 = suc0 / tot_suc
        percentage_suc1 = suc1 / tot_suc
        percentage_fail0 = fail0 / tot_fail
        percentage_fail1 = fail1 / tot_fail
        print("%9d  %7.3f        %4d     %4.3f   %4.3f        %3d  %7.3f  %6.3f" % (i, util, tot_suc, percentage_suc0, percentage_suc1, tot_fail, percentage_fail0, percentage_fail1))




"""
aa = sum(succ[0:results_fixed_treatment[2], 127])/len(succ[0:results_fixed_treatment[2], 127])
print("Percentege og people which drug)


# Check what happens when we always stick to one medicin, try all of them
results_fixed_treatment = np.zeros((generator.get_n_actions(), 2))
for i in range(len(results_fixed_treatment)):
    policy_temp = FixedTreatmentRecommender.FixedTreatmentRecommender(generator.get_n_actions(), generator.get_n_outcomes(), i)
    results_fixed_treatment[i] = test_policy(generator, policy_temp, reward_function, n_tests)

# Let's only look at the ten best gene treatments
best_indices = np.argsort(-results_fixed_treatment[:,0])[:10]

print("Total rewards (max is %d):" %n_tests)
for i in best_indices:
    print("Treatment %3d:      %7.4f\t# of Treatments: %3d" % (i, results_fixed_treatment[i,0],results_fixed_treatment[i,1]))

"""

#policy.final_analysis()