import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

class NameBanker: 

    #def __init__(self, model):
     #   self.model = model


    
    # Fit the model to the data.  You can use any model you like to do
    # the fit, however you should be able to predict all class
    # probabilities
    def fit(self, X, y):
        
        # Define the upper limit of possible k's
        max_k = 100
        
        # Define a set of possible k-values.
        ks = range(1,max_k)
        
        # Create mak_x number of untrained KNN-classifiers for each possible value of k
        untrained_models = [KNeighborsClassifier(n_neighbors=k) for k in ks]
        
        # From STK-IN4300 we know that 5 or 10 folds are the standard choices
        folds = 5;
        
        # Calculate the cross validations scores for each of the k's with 5 folds
        k_fold_scores = [cross_val_score(estimator = model, X=X, y=y, cv = folds) for model in untrained_models] 
        
        #skal vi ha med scaled over?
        
        # Take the mean score for each of the k's.
        # That is, originally each k has 5 cv-scores. We take the average of them.
        mean_cv_scores = [score.mean() for score in k_fold_scores] 
        
        # Plot of the errorbars to illustrate wich k is the optimal k for KNN
        #print("Plot of the mean cross-validation scores for the different values of k")
        #plt.errorbar(ks, mean_cv_scores, yerr=[score.std() for score in k_fold_scores])
        #plt.show()
        
        # Find the value of k that maximize the cross-validation score
        knn_best_k_cv = np.asarray(mean_cv_scores).argmax()
        print("The best k-value for KNN was: ", knn_best_k_cv)

        # Create a KNN classifier with the optimum k-value
        neigh = KNeighborsClassifier(n_neighbors = knn_best_k_cv)
        
        # Fit the KNN to our data
        self.model = neigh.fit(X,y)

        
        
    
    # set the interest rate
    def set_interest_rate(self, rate):
        self.rate = rate


    # Predict the probability of success for a specific person with data x
    def predict_proba(self, x):
        # We use the predict_proba function in the sklearn.neighbors.KNeighborsClassifier class.
        # Note that we add '[0,0]' at the end to only get the prob of success
        
      #  print("Probablilities for suc and fail: ", self.model.predict_proba(x)[0])
        return  self.model.predict_proba(x)[0,0]
        

    # THe expected utility of granting the loan or not. Here there are two actions:
    # action = 0 do not grant the loan
    # action = 1 grant the loan
    #
    # Make sure that you extract the length_of_loan from the
    # 2nd attribute of x. Then the return if the loan is paid off to you is amount_of_loan*(1 + rate)^length_of_loan
    # The return if the loan is not paid off is -amount_of_loan.
    def expected_utility(self, x, action):
        
        # Har han skrevet feil formel over? Han har glemt -1
        
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
            #print ("Util: %.2f \t Success: %.2f Amount: %d Duration: %d \n" % (returnValue, success, amount, duration))
            return returnValue

    # Here we calculate the best action based on which action returns the highest
    # expected utility
    def get_best_action(self, x):
        if 0 > self.expected_utility(x, 1): 
            return 0
        else: 
            return 1     
    