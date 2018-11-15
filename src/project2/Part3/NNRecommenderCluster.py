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
from sklearn.cluster import KMeans
import numpy as np
import pandas

class NNRecommenderCluster:

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
        # It is stupid to store the data in the model
        # but it makes the code more elegant in the
        # refiting procedure
        self.data = None
        self.actions = None
        self.outcome = None
        self.fitted = False
        
        
        # This we need for the kmenas
        self.centers = np.zeros((2,128))
        self.labels = None
        

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
        kmeans_model = KMeans(n_clusters=2, random_state=1).fit(data)
        self.centers = kmeans_model.cluster_centers_
        self.labels = kmeans_model.labels_
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
            self.fit_data(data)
        else:
            # Refitting the model. Add the new data
            self.data = np.concatenate((self.data, data), axis=0)
            self.actions = np.concatenate((self.actions, np.array([actions]).reshape(1,1)), axis=0)
            self.outcome = np.concatenate((self.outcome, np.array([outcome]).reshape(1,1)), axis=0)
            self.fit_data(self.data)
            
        print("Fitting treatment outcomes")
        self.model0 = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                   hidden_layer_sizes=(5, 2), random_state=1)
        
        self.model1 = MLPClassifier(solver='lbfgs', alpha=1e-5,
                                   hidden_layer_sizes=(5, 2), random_state=1)
        
        # Add actions to the dataset
        data2 = pandas.DataFrame(self.data)
        data2['a'] = self.actions
        data2 = data2.values
        
        data0 = data2[self.labels == 0, :]
        data1 = data2[self.labels == 1, :]
        self.model0.fit(data0, np.ravel(self.outcome[self.labels == 0]))
        self.model1.fit(data1, np.ravel(self.outcome[self.labels == 1]))
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
            return self.reward(actions, outcome)
        else:
            # We have a policy
            # So we assume actions and outcome are 'None'
            # We look at what the policy recommends
            recommended_action = policy.recommend(data)
            predict_proba_recommended_action = self.predict_proba(data, recommended_action)
            
            # E[X] = \sum_x p(x)*x
            estimated_reward = predict_proba_recommended_action[0,0]*self.reward(recommended_action, 0) + predict_proba_recommended_action[0,1]*self.reward(recommended_action, 1)
            return estimated_reward
            
        
    # Return a distribution of effects for a given person's data and a specific treatment.
    # This should be an numpy.array of length self.n_outcomes
    def predict_proba(self, data, treatment):
        # We assume we have a model here. NN
        data2 = pandas.DataFrame(data)
        data2['a'] = treatment 
        
        # Check which group the new patient belongs to:
        dist_to_center_0 = np.sum(np.square(data + self.centers[0]))
        dist_to_center_1 = np.sum(np.square(data + self.centers[1]))
        
        if (dist_to_center_0 <= dist_to_center_1):
            print("Use model 0")
            return self.model0.predict_proba(data2)
        else:
            print("Use model 1")
            return self.model1.predict_proba(data2)
    
    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    def recommend(self, user_data):
        # We want to maximize the utility
        # So we chose the action which yields the best reward
        
        # Due to Christos' change in the generator file in the latest
        # update we have to reshape the data.
        user_data = user_data.reshape(1,130)
        
        # Find the probabilities for the outcome given user_data and action
        placebo_prob_outcomes = self.predict_proba(user_data, 0)
        drug_prob_outcomes = self.predict_proba(user_data, 1)
        
        #print(placebo_prob_outcomes)
        #print(placebo_prob_outcomes[0,0])
        #print(placebo_prob_outcomes[0,1])
        
        # Estimate the reward.
        # Reward takes action and outcome
        placebo_estimate_reward = placebo_prob_outcomes[0,0]*self.reward(0, 0) + placebo_prob_outcomes[0,1]*self.reward(0, 1)
        drug_estimate_reward = drug_prob_outcomes[0,0]*self.reward(1, 0) + drug_prob_outcomes[0,1]*self.reward(1, 1)
        
        if (placebo_estimate_reward >= drug_estimate_reward):
            return 0
        else:
            return 1
        


    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        self.fit_treatment_outcome(user.reshape(1,130), action, outcome)
        return None
    
    
        
    # After all the data has been obtained, do a final analysis. This can consist of a number of things:
    # 1. Recommending a specific fixed treatment policy
    # 2. Suggesting looking at specific genes more closely
    # 3. Showing whether or not the new treatment might be better than the old, and by how much.
    # 4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
    def final_analysis(self):
        return None