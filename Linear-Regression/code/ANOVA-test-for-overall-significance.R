
# R demo
# Plotting functions and histograms, F distribution
## ANOVA tables, F tests, MLR with categorical variables

# Plotting functions (e.g Probability density functions)
x <- seq(-4, 4, 0.01) # grid of x values to evaluate
y <- dnorm(x, 0, 1) # function that evaluates normal PDF with mean 0 and SD 1
plot(x, y, type="l") # "l" indicates line plot

y <- x^2
plot(x, y, type="l")


# F distribution examples
x <- seq(0, 5, 0.01)
plot(x, y = df(x, df1 = 1, df2 = 2),
     type = "l", xlab = "x", ylab = "density")
plot(x, y = df(x, df1 = 1, df2 = 1),
     type = "l", col = "black", 
     xlab = "x", ylab = "density",
     ylim = c(0, 2.5), lwd = 2)
lines(x, y = df(x, df = 1, df2 = 100), 
      type = "l", col = "green", lwd = 2)
lines(x, y = df(x, df = 5, df2 = 1), 
      type = "l", col = "blue", lwd = 2)
lines(x, y = df(x, df = 5, df2 = 100), 
      type = "l", col = "purple", lwd = 2)
lines(x, y = df(x, df = 10, df2 = 1), 
      type = "l", col = "red", lwd = 2)
lines(x, y = df(x, df = 10, df2 = 100), 
      type = "l", col = "orange", lwd = 2)

legend("topright", 
       legend = c("df1 = 1, df2 = 1",
                  "df1 = 1, df2 = 100",
                  "df1 = 5, df2 = 1",
                  "df1 = 5, df2 = 100",
                  "df1 = 10, df2 = 1",
                  "df1 = 10, df2 = 100"),
       lty = 1, col = c("black", "green", 
                        "blue", "purple", 
                        "red", "orange"))


# Random numbers from F distribution
set.seed(12345678)
randF <- rf(1000, 5, 100)
hist(randF)
hist(randF, freq = FALSE)

# add true density
lines(x, y = df(x, df1 = 5, df2 = 100), type = "l", col = "purple", lwd = 2)

# set y-axis limits and more detailed histogram bins
hist(randF, freq = FALSE, ylim = c(0, 0.8), breaks = 25)
# add true density
lines(x, y = df(x, df1 = 5, df2 = 100), type = "l", col = "purple", lwd = 2)

# we can create a smoother F-distribution curve
randF <- rf(10000, 5, 100)
hist(randF)
# set y-axis limits and more detailed histogram bins
hist(randF, freq = FALSE, ylim = c(0, 0.8), breaks = 25)
# add true density
lines(x, y = df(x, df1 = 5, df2 = 100), type = "l", col = "purple", lwd = 2)


## Revisit rocket example
rocket <- read.csv(file = "./dataset/rocket.csv")
attach(rocket)
m1 <- lm(thrust ~ nozzle + propratio)
summary(m1)
anova(m1) # compare with ANOVA from table
anova(m1)$`Sum Sq`
sum(anova(m1)$`Sum Sq`[1:2]) # SS(Reg)
SSRes <- anova(m1)$`Sum Sq`[3]

# Tes of overall significance
m_red <- lm(thrust ~ 1) 
summary(m_red)
anova(m_red)
SSRes_A <- anova(m_red)$`Sum Sq`[1]

# F-statistic
l <- 2
n <- nrow(rocket)
p <- 2
Fstat <- ((SSRes_A - SSRes)/l) / (SSRes/(n-p-1))
pval <- 1 - pf(Fstat, df1 = l, df2 = n-p-1)


# Coffee example (Coffee Quality Institute 2018)
coffee <- read.csv("dataset/coffee_arabica.csv")
head(coffee)
attach(coffee)

mfull <- lm(Flavor ~ factor(Processing.Method) + 
               Aroma + Aftertaste + Body + Acidity +
               Balance + Sweetness + Uniformity + 
               Moisture) # wrapping the categorical in factor makes R treat Processing method as indicator variables. R automatically takes care of creating k - 1 indicator variables and the baseline relative.

summary(mfull)
anova(mfull)
SSRes <- anova(mfull)$`Sum Sq`[10]


## Reduced model without uniformity and moisture (beta9 = beta10 = 0)
m_red <- lm(Flavor ~ factor(Processing.Method) + 
               Aroma + Aftertaste + Body + Acidity +
               Balance + Sweetness)
summary(m_red)
anova(m_red)
SSRes_A <- anova(m_red)$`Sum Sq`[8]

# F-statistic
l <- 2 # number of constraints i.e beta9 and beta10
n <- nrow(coffee)
p <- 10 # 2 indicators and 8 numerical predictors
Fstat <- ((SSRes_A - SSRes)/l) / (SSRes/(n-p-1))
pval <- 1 - pf(Fstat, df1 = l, df2 = n-p-1)

if (pval >= 0.05) {
   print("Fail to reject null hypothesis at the 5% level that the regression coefficients for uniformity and moisture can possibly be zero")
} else if (pval < 0.05) {
   print("Reject null hypothesis at the 5% level that the regression coefficients for uniformity and moisture can possibly be zero")
}


## Reduced model without uniformity and moisture and
## setting effect of Dry = Semi (beta1 = beta9 = beta10 = 0)
coffee$method2 <- ifelse(coffee$Processing.Method %in% c("Natural / Dry", 
                                                         "Semi-washed / Semi-pulped"), 0, 1)
# use the ifelse to set the baseline and the indicator variable we want to test for to 0 else 1
# coffee$wet <- ifelse(coffee$Processing.Method == "Washed / Wet", 0, 1) # if we want to test for wet
attach(coffee)
m_red2 <- lm(Flavor ~ method2 + Aroma + Aftertaste + Body + Acidity + Balance + Sweetness)
summary(m_red2)
anova(m_red2)

SSRes_A <- anova(m_red2)$`Sum Sq`[8]

# F-statistic
l <- 3 # number of constraints i.e beta9 and beta10
n <- nrow(coffee)
p <- 10 # 2 indicators and 8 numerical predictors
Fstat <- ((SSRes_A - SSRes)/l) / (SSRes/(n-p-1))
pval <- 1 - pf(Fstat, df1 = l, df2 = n-p-1)


if (pval >= 0.05) {
   print("Fail to reject null hypothesis at the 5% level that the regression coefficients for uniformity and moisture can possibly be zero")
} else if (pval < 0.05) {
   print("Reject null hypothesis at the 5% level that the regression coefficients for uniformity and moisture can possibly be zero")
}

# easier method of comparing two models 
anova(mfull, m_red2)

# In practice, we may carry out several different f-tests however, to know the predictors to actually put in the model, we carry out "variable or model selection methods".
