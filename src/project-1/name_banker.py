import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

class NameBanker: 

    #def __init__(self, model):
     #   self.model = model


    
    # Fit the model to the data.  You can use any model you like to do
    # the fit, however you should be able to predict all class
    # probabilities
    def fit(self, X, y):
       neighbor_ks = range(1,100)
       untrained_models = [KNeighborsClassifier(n_neighbors=k) for k in neighbor]
        
        features = ['checking account balance', 'duration', 'credit history',
            'purpose', 'amount', 'savings', 'employment', 'installment',
            'marital status', 'other debtors', 'residence time',
            'property', 'age', 'other installments', 'housing', 'credits',
            'job', 'persons', 'phone', 'foreign']
        target = 'repaid'
        
        k_fold_scores = [cross_val_score(estimator=m), X = features, y = target, cv = 5] #skal vi ha med scaled?
        
        mean_cv_scores = [score.mean() for score in k_fold_scores] 
        plt.errorbar(neighbor_ks, mean_cv_scores, yerr=[score.std() for score in k_fold_scores])
        
        knn_best_k_cv = numpy.asarray(mean_cv_scores).argmax()
        knn_best_k_train = numpy.asarray(train_scores).argmax()
        knn_best_k_test = numpy.asarray(test_scores).argmax()
        print(ks[knn_best_k_cv], ks[knn_best_k_train], ks[knn_best_k_test])
        plt.semilogx(ks, train_scores, ks, test_scores, ks, mean_xv_scores)
        plt.legend(["Train", "Test", "XV"])

        # Let's select the best model on the basis of the XV score, as we must, since the 'test' result is invisible to us
        knn_best_model_cv = models[knn_best_k_cv]
        




        neigh = KNeighborsClassifier(n_neighbors = knn_best_k_cv)
        self.model = neigh.fit(X,y)
        return self.model
        
        
    
    # set the interest rate
    def set_interest_rate(self, rate):
        self.rate = rate
        return

    # Predict the probability of success for a specific person with data x
    def predict_proba(self, x):
        return  self.model.predict_proba(x)[0,0]
        

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
            p = self.predict_proba(x)
            return -(1-p)*x[4] + p*x[4]*((1 + x[7])**(x[1]) - 1)  

        
    def get_best_action(self, x):
        if self.expected_utility(x, 0) > self.expected_utility(x, 1): 
            return 0
        else: 
            return 1     
    
