
## Coffee example (Coffee Quality Institute, 2018) continued
coffee <- read.csv("dataset/coffee_arabica.csv") 
head(coffee)
attach(coffee)

cor(coffee) # doesn't work since there is a categorical variable
cor(coffee[, -1]) # e.g remove first column
pairs(coffee[, -1]) # scatterplot matrix for all the pairs
pairs(~ Flavor + Aroma + Aftertaste + Body + Acidity + 
        Balance + Sweetness + Uniformity + Moisture) # another way to call pairs()


# Code our own indicators, so we can more easily interpret VIFs
coffee$wet <- ifelse(coffee$Processing.Method == "Washed / Wet", 1, 0) # wet = 1, 0 otherwise
coffee$semi <- ifelse(coffee$Processing.Method == "Semi-washed / Semi-pulped", 1, 0) # 1 = semi, 0 otherwise

attach(coffee)

# Get full model
mfull <- lm(Flavor ~ wet + semi + Aroma + Aftertaste + Body + 
              Acidity + Balance + Sweetness + Uniformity + Moisture)
summary(mfull)

# manually calculate vif for each variable
# step 1: we get the multiple R-squared of individual x variables fitted on other predictors
# step 2: we calculate VIF = 1 / (1 - R-squared)
wet_reg <- lm(wet ~ semi + Aroma + Aftertaste + Body + 
                Acidity + Balance + Sweetness + Uniformity + Moisture)
r2_wet <- summary(wet_reg)$r.squared
VIF_wet <- 1 / (1 - r2_wet)


Aroma_reg <- lm(Aroma ~ semi + wet + Aftertaste + Body + 
                Acidity + Balance + Sweetness + Uniformity + Moisture)
r2_Aroma <- summary(Aroma_reg)$r.squared
VIF_Aroma <- 1 / (1 - r2_Aroma)


Aftertaste_reg <- lm(Aftertaste ~ semi + wet + Aroma + Body + 
                  Acidity + Balance + Sweetness + Uniformity + Moisture)
r2_Aftertaste <- summary(Aftertaste_reg)$r.squared
VIF_Aroma <- 1 / (1 - r2_Aftertaste)



library(car) # this library has the vif() function

vif(mfull) # we use the vif() on the full model to calculate the vif of each variable

# Python in FL everglades example (2017)
## Sex, length, total mass, fat mass, and specimen condition data for
## 248 Burmese pythons (Python Bivittatus) collected in the Florida Everglades

python <- read.csv("./dataset/FLpython.csv")
head(python)

python$male <- ifelse(python$sex == "M", 1, 0) # 1 = M, 0 = F

cor(python[, -1])
pairs(python[, -1])
attach(python)

mpf <- lm(fat ~ male + svl + mass + length)
summary(mpf) # we see that the SE(svl) is greatly inflated due to multicollinearity
vif(mpf)

# we see that length variable is the largest variable with vif >= 10 so we remove "length" based on VIF
mpf2 <- lm(fat ~ male + mass + svl)
summary(mpf2) # the SE for svl has greatly decreased after removing "length" variable
vif(mpf2)










