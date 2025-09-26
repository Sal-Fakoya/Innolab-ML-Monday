## Coffee example (Coffee Quality Institute, 2018) continued.
coffee <- read.csv("./dataset/coffee_arabica.csv")

# Full model with all predictors
mfull <- lm(
  Flavor ~ factor(Processing.Method) + Aroma + Aftertaste +
    Body + Acidity + Balance + Sweetness + Uniformity + Moisture,
  data = coffee
)

# Check model metrics
summary(mfull)$adj.r.squared
AIC(mfull)
BIC(mfull)

# Exhaustive, brute-force search using leaps package
library(leaps)
all_regs <- regsubsets(
  Flavor ~ .,
  data = coffee,
  nvmax = 10,
  nbest = 2^10,
  really.big = TRUE
)
all_regs_summ <- summary(all_regs)

# View model selection results (commented out for brevity)
# all_regs_summ$which
# all_regs_summ$adjr2
# all_regs_summ$bic

# Organize results according to number of variables in model
p <- 10
k <- c(
  rep(1, choose(p, 1)),
  rep(2, choose(p, 2)),
  rep(3, choose(p, 3)),
  rep(4, choose(p, 4)),
  rep(5, choose(p, 5)),
  rep(6, choose(p, 6)),
  rep(7, choose(p, 7)),
  rep(8, choose(p, 8)),
  rep(9, choose(p, 9)),
  rep(10, choose(p, 10))
)

# Plot adjusted R-squared by number of predictors
boxplot(
  all_regs_summ$adjr2 ~ k,
  xlab = "Number of predictors",
  ylab = "Adjusted R-squared",
  ylim = c(0, 1)
)
abline(h = c(0, 1), lty = 2, col = "red")

# Plot BIC by number of predictors
boxplot(all_regs_summ$bic ~ k, 
        xlab = "Number of predictors", 
        ylab = "BIC")

# Find best models by criteria
max(all_regs_summ$adjr2)
bestR2adj <- which.max(all_regs_summ$adjr2)
min(all_regs_summ$bic)
bestBIC <- which.min(all_regs_summ$bic)

# Check which predictors are in the best models
all_regs_summ$which[bestR2adj,]
all_regs_summ$which[bestBIC,]

# Create dummy variables for processing method
coffee$wet <- ifelse(coffee$Processing.Method == 'Washed / Wet', 1, 0) # 1 = wet, 0 otherwise
coffee$semi <- ifelse(coffee$Processing.Method == 'Semi-washed / Semi-pulped', 1, 0) # 1 = semi/dry, 0 otherwise
coffee$Processing.Method <- NULL

# Fit best adjusted R-squared model
m_bestr2adj <- lm(
  Flavor ~ wet + Aroma + Aftertaste +
    Body + Acidity + Balance + Sweetness + Uniformity + Moisture,
  data = coffee
)
summary(m_bestr2adj)
AIC(m_bestr2adj)
BIC(m_bestr2adj)

# Fit best BIC model (more parsimonious)
m_bestBIC <- lm(Flavor ~ wet + Aroma + Aftertaste +
                  Body + Acidity + Sweetness, data = coffee)
summary(m_bestBIC)
AIC(m_bestBIC)
BIC(m_bestBIC)

# Stepwise methods using MASS package
library(MASS)

# Define full and empty models for stepwise selection
full <- lm(Flavor ~ ., data = coffee)
empty <- lm(Flavor ~ 1, data = coffee)

# Forward selection using AIC (default)
m_f_AIC <- stepAIC(
  object = empty,
  scope = list(upper = full, lower = empty),
  direction = "forward",
  trace = 0
)

# Forward selection using BIC (k = log(n))
m_f <- stepAIC(
  object = empty,
  scope = list(upper = full, lower = empty),
  direction = "forward",
  trace = 0,
  k = log(nrow(coffee))
)
summary(m_f)

# Backward elimination using BIC
m_b <- stepAIC(
  object = full,
  scope = list(upper = full, lower = empty),
  direction = "backward",
  trace = 0,
  k = log(nrow(coffee))
)
summary(m_b)

# Hybrid forward-backward selection using BIC
m_h <- stepAIC(
  object = empty,
  scope = list(upper = full, lower = empty),
  direction = "both",
  trace = 0,
  k = log(nrow(coffee))
)
summary(m_h)

# Note: With 10 variables, all 3 stepwise approaches identify 
# the same BIC-based model as the exhaustive search

# It seems we choose the model with 6 predictors 