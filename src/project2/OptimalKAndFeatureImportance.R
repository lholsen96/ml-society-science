setwd("C:/Users/lars9/OneDrive/UiO-Høst2018/IN-STK5000/GITHUB/Proj2")
features = read.csv('historical_X.dat', sep = ' ', header = FALSE)

observations = features[,1:128]
labels = features[,129] + features[,130]*2

## 75% of the sample size
smp_size <- floor(0.75 * nrow(features))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(features)), size = smp_size)

train_obs <- observations[train_ind, ]
train_lab <- labels[train_ind]
test_obs <- observations[-train_ind, ]
test_lab <- labels[-train_ind]

### Find number of clusters
library(factoextra)
library(NbClust)
df <- scale(train_obs)
head(df)

# Elbow method
fviz_nbclust(df, kmeans, method = "wss") +
  geom_vline(xintercept = 4, linetype = 2)+
  labs(subtitle = "Elbow method")

# Silhouette method
fviz_nbclust(df, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")
# We see that two clusters are the optimal number of cluster

# Find the most important features
library(randomForest)
model <- randomForest(y=as.factor(train_lab),x=train_obs, ntree=250)
summary(model)
model$importance

# Look at how good it is to predict
pred = predict(model, test_obs)
sum(pred == test_lab) # 1712 of 2500 correct.

# Find the 10 best features
aa = order(model$importance, decreasing = T)[1:10]
bestfeatures = cbind(order(model$importance, decreasing = T)[1:10], model$importance[aa])

train_obs_fr <- train_obs[ ,aa[1:10]]
test_obs_fr <- test_obs[ ,aa[1:10]]


# Check number of clusters when we have reduced
# the number of features to ten. Then we get the
# same results. If we had lowerd it to 4 we would
# have gotten 9 as the optimal k. Now we get the
# same as before.
df2 <- scale(train_obs_fr)
head(df2)

# Elbow method
fviz_nbclust(df2, kmeans, method = "wss") +
  geom_vline(xintercept = 4, linetype = 2)+
  labs(subtitle = "Elbow method")

# Silhouette method
fviz_nbclust(df2, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")



### Let's look at how the best features predict
model <- randomForest(y=as.factor(train_lab), x=train_obs_fr, ntree=250)
summary(model)
model$importance

# They do a little bit better
pred = predict(model, test_obs_fr)
sum(pred == test_lab) # 1730 of 2500 correct.


# See if we can remove anything else:
library(nnet)
fit2 = multinom(train_lab~ ., data=cbind(train_obs_fr, train_lab))
step <- stepAIC(fit2, direction="both")
step$anova # display results 
# V6 + V4 + V114 + V12 + V84 + V2 + V56

# Predictions with multinom yields 
pred = predict(fit2, test_obs_fr)
sum(pred == test_lab) # 1720 of 2500 correct.


# Train 
fitrf = randomForest(x = train_obs_fr, y=as.factor(train_lab),  xtest=test_obs_fr, 
             ytest=as.factor(test_lab), ntree=500)
# See that we are bad at predicting label 2 and 3.
fitrf$confusion
#      0    1 2 3 class.error
# 0 2938  695 1 0   0.1915245
# 1  788 2362 0 0   0.2501587
# 2  360   42 0 0   1.0000000
# 3  117  197 0 0   1.0000000


# look at the parwise plot. The data is located in the cornes as
# mentioned before.
plot(dff[1:20,])


# So look at the prediction without the important features.
train_obs_fr_stupied <- train_obs[ ,-aa[1:10]]
test_obs_fr_stupied <- test_obs[ ,-aa[1:10]]
fitrf = randomForest(x = train_obs_fr_stupied, y=as.factor(train_lab), ntree=500)

# This method perform worse. Which is good.
pred = predict(fitrf, test_obs_fr_stupied)
sum(pred == test_lab) # 1657 of 2500 correct.
fitrf$confusion