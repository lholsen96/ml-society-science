# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 09:29:41 2018

@author: Lars
"""

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
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas

class HistoricalRecommender:

    #################################
    # Initialise
    #
    # Set the recommender with a default number of actions and outcomes.  This is
    # because the number of actions in historical data can be
    # different from the ones that you can take with your policy.
    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward
        
        # By default we set the probabilites to be a half. We update these in
        # fit_treatment_outcome().
        self.prob_actions = np.array([0.5, 0.5])
    
        # It is stupid to store the data in the model
        # but it makes the code more elegant in the
        # refiting procedure
        self.data = None
        self.actions = None
        self.outcome = None
        self.fitted = False
        

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
        # First time we fit the model
        if (self.fitted == False):
            self.data = data
            self.actions = actions
            self.outcome = outcome
            self.fitted = True
        else:
            # Refitting the model. Add the new data
            self.data = np.concatenate((self.data, data), axis=0)
            self.actions = np.concatenate((self.actions, np.array([actions]).reshape(1,1)), axis=0)
            self.outcome = np.concatenate((self.outcome, np.array([outcome]).reshape(1,1)), axis=0)
            
        print("Fitting treatment outcomes")
    
    
    
    
    
        # Find number of patients and number of patients on 
        # the experimenta drug
        number_of_patients = len(actions)
        number_of_experimental_drug = np.sum(actions)
        
        # Here we look at the percentages of people who get
        ##placebo and experimental drug
        percentage_experimenta_drug = number_of_experimental_drug / number_of_patients
        percentage_placebo = 1 - percentage_experimenta_drug
        
        # We store these results
        self.prob_actions = np.array([percentage_placebo, percentage_experimenta_drug])
        
        
        
        
        
        
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
            return sum(self.reward(actions, outcome))
        else:
            
            # We have a policy
            # we assume it is an object of class random_recommender_lars
            # this is a fitted model 
            # We assume that we only get data, actions = None, and Outcome=None
            # And that we have to use the model to find the best actions
            # Then we use predict proba to find prob for outcome
            
            estimated_utility = 0
            
            # For each action in data we want to find the reward
            for row in range(data.shape[0]):
                print("Estimating data %6d of %d" % (row, data.shape[0]))
                # This iter data
                iter_data = np.array(data[row].reshape(1,130))
                #np.array(features[0].reshape(1,130))
                                                
                # Start by finding the recommended action
                recommended_action = policy.recommend(iter_data)
                
                # Find the probabilities for the outcome
                predict_proba_recommended_action = policy.predict_proba(iter_data, recommended_action)
                
                # E[X] = \sum_x p(x)*x
                estimated_reward = predict_proba_recommended_action[0,0]*self.reward(recommended_action, 0) + predict_proba_recommended_action[0,1]*self.reward(recommended_action, 1)
           
                # Add the reward
                estimated_utility += estimated_reward
                
            return estimated_utility
            
        
    # Return a distribution of effects for a given person's data and a specific treatment.
    # This should be an numpy.array of length self.n_outcomes
    def predict_proba(self, data, treatment):
        # This is not necassary here since we have static probabilities
        return None
    
    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    def recommend(self, user_data):
        # A bit unsure here
        # Should we 1) just draw a uniform, and if it is less than prob_action[0]
        # then we choose action 0?
        # Or should we 2) look at the two probabilites and loook at which maximize 
        # the estimated reward?
        
        # 1)
        # We draw a uniform RV. If it is less than prob_action[0], then we
        # choose action 0(placebo), otherwise we choose action 1(experimental drug).
        prob = np.random.uniform(0,1,1)
        if (prob <= self.prob_actions[0]):
            return 0
        else:
            return 1
        
    
        
    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        # Here we do nothing
        # since we interpret HistoricalRecommender to be a static model
        # which only uses the historical data to make a decission.
        return None
    
    # After all the data has been obtained, do a final analysis. This can consist of a number of things:
    # 1. Recommending a specific fixed treatment policy
    # 2. Suggesting looking at specific genes more closely
    # 3. Showing whether or not the new treatment might be better than the old, and by how much.
    # 4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
    def final_analysis(self):
        print("\nFinal analysis of HistoricalRecommender:")
        print("Probability for action 0: %.4f \nProbability for action 1: %.4f" % (self.prob_actions[0], self.prob_actions[1]))
        return None