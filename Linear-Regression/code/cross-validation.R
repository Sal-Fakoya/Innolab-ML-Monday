
# Cross-Validation

## Coffee Example

coffee <- read.csv("./dataset/coffee_arabica.csv")

# create our own indicator variables
coffee$wet <- ifelse(coffee$Processing.Method == "Washed / Wet", 1, 0) # 1 = wet, 0 otherwise
coffee$semi <- ifelse(coffee$Processing.Method == "Semi-washed / Semi-pulped", 1, 0) # 1 = semi, 0 otherwise
coffee$Processing.Method <- NULL # this variable is redundant so we remove it

N <- nrow(coffee)

## Train and Validation set split
set.seed(12345678)
trainInd <- sample(1:N, round(N * 0.8), replace = FALSE) # select a random sample: 80% for train data
trainSet <- coffee[trainInd, ] # select the rows in the training indices
validSet <- coffee[-trainInd, ] # remaining observations for Validation Set

attach(trainSet)

# Calculate RMSE on three models with different variables included
m1 <- lm(Flavor ~ wet + semi + Aroma + Aftertaste + Body)
pred1 <- predict(m1, newdata = validSet)
sqrt(mean((validSet$Flavor - pred1) ^ 2)) # RMSE 
mean(abs(validSet$Flavor - pred1)) # MAE: we can also use MAE to further check

m2 <- lm(Flavor ~ wet + Aroma + Aftertaste + Body + Acidity + Balance +
           Sweetness + Uniformity + Moisture)
pred2 <- predict(m2, newdata = validSet)
sqrt(mean((validSet$Flavor - pred2) ^ 2)) # RMSE

m3 <- lm(Flavor ~ Aroma + Aftertaste)
pred3 <- predict(m3, newdata = validSet)
sqrt(mean((validSet$Flavor - pred3) ^ 2))# RMSE



# K-fold cross-validation
K <- 5
testSetSplits <- sample((1:N) %% 5 + 1) # allocate k-fifth to each fold: use mod to get remainders 0 to 5, and add 1 
RMSE1 <- c()
RMSE2 <- c()
RMSE3 <- c()

for (k in 1:K) {
  testSet <- coffee[testSetSplits==k, ]
  trainSet <- coffee[testSetSplits != k, ]
  
  # for each model, we fit the model, predict and store the RMSE for that fold
  
  m1 <- lm(Flavor ~ wet + semi + Aroma + Aftertaste + Body, data = trainSet)
  pred1 <- predict(m1, newdata = testSet)
  RMSE1[k] <- sqrt(mean((testSet$Flavor - pred1) ^ 2)) # RMSE 
  
  m2 <- lm(Flavor ~ wet + Aroma + Aftertaste + Body + Acidity + Balance +
             Sweetness + Uniformity + Moisture, data = trainSet)
  pred2 <- predict(m2, newdata = testSet)
  RMSE2[k] <- sqrt(mean((testSet$Flavor - pred2) ^ 2)) # RMSE
  
  m3 <- lm(Flavor ~ Aroma + Aftertaste, data = trainSet)
  pred3 <- predict(m3, newdata = testSet)
  RMSE3[k] <- sqrt(mean((testSet$Flavor - pred3) ^ 2))# RMSE
  
  
}


RMSE1
RMSE2
RMSE3


# Expected performance based on 5-fold validation: gives us more confidence that model 2 is the best in terms of ability to generalize for prediction on new data
mean(RMSE1)
mean(RMSE2)
mean(RMSE3)









