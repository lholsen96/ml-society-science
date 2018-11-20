# -*- Mode: python -*-
# A simple reference recommender
#
#
# This is a medical scenario with historical data. 
#
# General functions
#
# - set_reward
# 
# There is a set of functions for dealing with historical data:
#
# - fit_data
# - fit_treatment_outcome
# - estimate_utiltiy
#
# There is a set of functions for online decision making
#
# - predict_proba
# - recommend
# - observe

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas
import scipy.io as sio

class CheatRecommender:

    #################################
    # Initialise
    #
    # Set the recommender with a default number of actions and outcomes.  This is
    # because the number of actions in historical data can be
    # different from the ones that you can take with your policy.
    def __init__(self, n_actions, n_outcomes, allow_update = 1):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

        # These variables are for the third part of the project. To be continued.
        # It is stupid to store the data in the model
        # but it makes the code more elegant in the
        # refiting procedure
        self.data = None
        self.actions = None
        self.outcome = None
        self.fitted = False
        
        # If we are allowed to update the model. That is
        # if we call fit_treatment_outcome in observe with the new data
        # or not. Default is allowed, but in Exercise 3 subquestion 2 we do not
        # allow this
        self.allow_update = allow_update
        
        # THe cheating part
        self.v = sio.loadmat("./big_generating_matrices.mat")['V']

    ## By default, the reward is just equal to the outcome, as the actions play no role.
    def _default_reward(self, action, outcome):
        return outcome


    # Set the reward function r(a, y)
    def set_reward(self, reward):
        self.reward = reward
    
    ##################################
    # Fit a model from patient data.
    #
    # This will generally speaking be an
    # unsupervised model. Anything from a Gaussian mixture model to a
    # neural network is a valid choice.  However, you can give special
    # meaning to different parts of the data, and use a supervised
    # model instead.
    def fit_data(self, data):
        print("Preprocessing data")
        return None


    ## Fit a model from patient data, actions and their effects
    ## Here we assume that the outcome is a direct function of data and actions
    ## This model can then be used in estimate_utility(), predict_proba() and recommend()
    def fit_treatment_outcome(self, data, actions, outcome):
        return None


    ## Estimate the utility of a specific policy from historical data (data, actions, outcome),
    ## where utility is the expected reward of the policy.
    ##
    ## If policy is not given, simply use the average reward of the observed actions and outcomes.
    ##
    ## If a policy is given, then you can either use importance
    ## sampling, or use the model you have fitted from historical data
    ## to get an estimate of the utility.
    def estimate_utility(self, data, actions, outcome, policy=None):
        if (policy == None):
            # We compute the estimated utility from the data
            # reawrd is a function that takes in actions and outcome
            # This is the same as in the jupyter notebook
            return sum(self.reward(actions, outcome))
        
        else:    
            # We have a policy
            # we assume it is an object of class LogisticRecommender, which has been fitted
            # If the policy has not been fitted we return -infinity
            if (policy.fitted == False):
                return -float('inf')

            # We assume that we only get data, actions = None, and Outcome = None
            # And that we have to use the model to find the best actions
            # Then we use predict_proba to find the probability for the outcomes
            estimated_utility = 0
            
            # For each action in data we want to find the estimated reward
            for row in range(data.shape[0]):
                # Fancy printout
                if ((row + 1) % 100) == 0:
                    print("Logistic model estimating utility %6d of %d" % (row + 1, data.shape[0]))

                # This iteration's data
                iter_data = np.array(data[row].reshape(1,130))
                
                # Start by finding the recommended action
                recommended_action = policy.recommend(iter_data)
                
                # Find the probabilities for the outcome
                predict_proba_recommended_action = policy.predict_proba(iter_data, recommended_action)
                
                # E[f(X)] = \sum_x p(x)*f(x), X is a discrete RV
                # We use self.reward since we want to use this instance version of reward to calculate the reward
                # Policy can be another instance of Recommender (with the same interface) which calculates the best 
                # action in another way and based on another reward
                estimated_reward = predict_proba_recommended_action[0,0]*self.reward(recommended_action, 0) + predict_proba_recommended_action[0,1]*self.reward(recommended_action, 1)
           
                # Add the reward
                estimated_utility += estimated_reward
                
            return estimated_utility
            
        
    # Return a distribution of effects for a given person's data and a specific treatment.
    # This should be an numpy.array of length self.n_outcomes
    def predict_proba(self, data, treatment):
        return None
    
    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    def recommend(self, user_data):
        # We want to maximize the utility
        # So we chose the action which yields the best estimated reward
        ans = np.dot(self.v, user_data)
        ret = np.argmax(ans)
        if(ans[ret] < 0):
            return 0
        else: 
            return np.argmax(ans)
        


    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        
        return None

    # After all the data has been obtained, do a final analysis. This can consist of a number of things:
    # 1. Recommending a specific fixed treatment policy
    # 2. Suggesting looking at specific genes more closely
    # 3. Showing whether or not the new treatment might be better than the old, and by how much.
    # 4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
    def final_analysis(self):
        return None