
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

class NameBanker: 

    def __init__(self, model):
        self.model = model
    
    # Fit the model to the data.  You can use any model you like to do
    # the fit, however you should be able to predict all class
    # probabilities
    def fit(self, X, y):
        self.model = self.model.fit(X,y)
          
         
    # set the interest rate
    def set_interest_rate(self, rate):
        self.rate = rate


    # Predict the probability of success for a specific person with data x
    def predict_proba(self, x):
        # We use the predict_proba function in the sklearn.neighbors.KNeighborsClassifier class.
        # Note that we add '[0,0]' at the end to only get the prob of success
        
        #print("Probablilities for suc and fail: ", self.model.predict_proba(x)[0])
        return  self.model.predict_proba(x)[0,0]
        

    # THe expected utility of granting the loan or not. Here there are two actions:
    # action = 0 do not grant the loan
    # action = 1 grant the loan
    #
    # Make sure that you extract the length_of_loan from the
    # 2nd attribute of x. Then the return if the loan is paid off to you is amount_of_loan*(1 + rate)^length_of_loan
    # The return if the loan is not paid off is -amount_of_loan.
    def expected_utility(self, x, action):
                
        # We do not grant the loan. Thus, the utility it set equal to zero.
        if action == 0:
            return 0
        
        # We grant the loan
        else:
            success = self.predict_proba([x])
            failure = 1-success
            amount = x['amount']
            duration = x['duration']
            returnValue = -failure*amount + success*amount*(pow(1 + self.rate, duration) - 1) 
            #print ("Expected util: %.2f \t ", % returnValue, "Success: ", success, "Amount: ", amount, "Duration: ", duration, "\n")
            print ("Util: %.2f \t Success: %.2f Amount: %d Duration: %d \n" % (returnValue, success, amount, duration))
            return returnValue

    # Here we calculate the best action based on which action returns the highest
    # expected utility
    def get_best_action(self, x):
        if self.expected_utility(x, 0) > self.expected_utility(x, 1): 
            return 0
        else: 
            return 1     
    