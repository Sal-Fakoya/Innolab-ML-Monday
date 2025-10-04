

# Python data revisited
# Python data re-visited
python <- read.csv("dataset/FLpython.csv")

# create dummy variables for male gender, female is the baseline
python$male <- ifelse(python$sex == "M", 1, 0) # 1 = M, 0 = F
head(python)
attach(python)
mpf2 <- lm(fat ~ male + mass + svl)
summary(mpf2)


# Last time we used a Box-Cox Transformation
library(MASS) # MASS library has the boxcox function

bc <- boxcox(mpf2) # plots the log-likelihood
lambda <- bc$x[which.max(bc$y)] # take the lambda that corresponds to the highest log-likehood
mpf3 <- lm((fat^lambda - 1) / lambda ~ male + mass + svl) # using this formula since lambda is not 0. Check latex-notes.pdf
summary(mpf3)

plot(mpf3$fitted.values, mpf3$residuals)
plot(mass, mpf3$residuals)
plot(svl, mpf3$residuals)

qqnorm(mpf3$residuals)
qqline(mpf3$residuals, col="blue", lwd = 2)



# Now for outliers
# Quantities for individual observations
studres(mpf3) # calculates the studentized residals
hatvalues(mpf3) # calculates leverage
cooks.distance(mpf3) # calculates Cook's distance 


# Residual plots with studentized residuals
plot(mpf3$fitted.values, studres(mpf3), xlab = "Fitted Values", ylab="Studentized residuals") # high leverage points will change 
abline(h = c(-3, 3), col = "red", lty=2) # majority of data points should lie withing 3 standard deviations of 0

which(abs(studres(mpf3)) > 3) # returns the index of observations greater than 3
qqnorm(studres(mpf3))
qqline(studres(mpf3), col="blue", lwd=2)


# Leverage
plot(hatvalues(mpf3), ylab="Leverage")
abline(h = 2 * mean(hatvalues(mpf3)), col="red", lty=2)
which(hatvalues(mpf3) > 2 * mean(hatvalues(mpf3)))
python[which(hatvalues(mpf3) > 2 * mean(hatvalues(mpf3))), ]


# Cook's distance: combines studentized residuals and leverage to determine most influential point
plot(cooks.distance(mpf3), ylab = "Cook's distance")
abline(h = 0.5, col="red", lty=2)
which(cooks.distance(mpf3) > 0.5)


# Let's look at actual changes in beta estimates
summary(mpf3)
mpf3$coefficients # extract the coefficients: with all data

# e.g fit without cook's distance observation 248: influential observations change the Beta estimates
mpf4 <- lm((fat^lambda - 1) / lambda ~ male + mass + svl, data=python[-248, ])
mpf4$coefficients # extract the coefficients: without observation 248
# we see the beta hats change quite a bit considering we remove one observation


# e.g fit without observation without non-influential observation 50
mpf5 <- lm((fat^lambda - 1) / lambda ~ male + mass + svl, data=python[-50, ])
mpf5$coefficients # we see that the Beta hat estimates do not change as much




