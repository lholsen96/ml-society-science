import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class NameBanker:

    #def __init__(self, model):
     #   self.model = model

    
    # Fit the model to the data.  You can use any model you like to do
    # the fit, however you should be able to predict all class
    # probabilities
    def fit(self, X, y):
        #self.data = [X, y]
        #Her skal vi legge inn metode for modell
        neigh = KNeighborsClassifier(n_neighbors = 3)
        neigh.fit = fit(X, y)

        return neigh.fit
        
       




    # set the interest rate
    def set_interest_rate(self, rate):
        self.rate = rate
        return

    # Predict the probability of failure for a specific person with data x
    def predict_proba(self, x):
        return 0

    # THe expected utility of granting the loan or not. Here there are two actions:
    # action = 0 do not grant the loan
    # action = 1 grant the loan
    #
    # Make sure that you extract the length_of_loan from the
    # 2nd attribute of x. Then the return if the loan is paid off to you is amount_of_loan*(1 + rate)^length_of_loan
    # The return if the loan is not paid off is -amount_of_loan.
    def expected_utility(self, x, action):
        if action == 0:
            return 0
        else:
            prob = predict_proba(self, x)
            return -p*x[4] + (1-p)*x[4]*((1 + x[7])**(x[1]) -1)  


        print("Expected utility: Not implemented")
        
    def get_best_action(self, x):
        return np.random.choice(2,1)[0]
