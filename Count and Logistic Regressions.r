
install.packages("MixAll") # contain the count data
install.packages("pscl") # count data models (zero inflated models)
install.packages("MASS") # negative binomial count model 
install.packages("rsq") # generalized r-square measures
isntall.packages("dplyr") # data manipulation package 

install.packages("caret") # classification pacakge (confusion matrix)
install.packages('e1071')

# graphical packages
 library(ggplot2)
 library(GGally)

# library for count models
 library(MixAll)
 library(pscl)
 library(MASS) 
 library(rsq)

# data manipulation
 library(dplyr)
 library(gplots) # package for additional plotting functions

# lib for confusion matrix 
library(caret)
library(e1071)

#####################################################
# 1. Count data regression models
#####################################################
## 1.1. Poisson regression model
## 1.2. Negative Binomial regression model
data(DebTrivedi) # load data; 
# description of data:
# Demand for medical care for elderly (>66 years old)
# Variables: ofp, ofnp: physician/non physician office visits
#            opp, opnp: hospital outpatient visits physician/non physician
#            emr: emergency room visits 
#            hosp: # of hospital stays
#            health: self-perceived health
#            adldiff: =1 has a condition that limits daily activities
#            school: years of education
#            faminc: family income in $10k
#            privins: =1 covered by private insurance
#            medicaid: =1 covered by medicaid
#we now predict physician office visits
dt <- DebTrivedi[,c(1,7:19)]
colnames(dt)

# histogram of the dependent variable: physician office visits
hist(dt$ofp)
plot(table(dt$ofp))

#####################################################
## 1.1 Poisson regression
#####################################################
pois1 <- glm(ofp ~ numchron, data = dt, family = poisson)

summary(pois1)

# interpretation: 1 unit increase in numchron leads to 21.9% increase in ofp
exp(pois1$coefficients[2])
exp(0.198)
# goodness of fit: 
# AIC (smaller is better), BIC (smaller is better), logLik (larger is better), Pseudo R2 (there are many; one commonly used is McFadden's Pseudo R2)
AIC(pois1)
BIC(pois1)
logLik(pois1)
# Pseudo R2: there are many, commonly used: McFadden Pseudo R2, defined as following
1 - pois1$deviance/pois1$null.deviance 

# prediction
newdata <- data.frame(numchron = c(1:10))
predict(pois1, newdata, type="response") # response means prediction is on the scale of response variable
# is it okay if the prediction take non-integer numbers? 

# multivariate poisson regression
pois1m <- glm(ofp ~ ., data = dt, family = poisson)
summary(pois1m)

# assess multicolinearity
library(fastDummies)
dt_dummy<-dummy_columns(dt,select_columns = c("health", "adldiff", "region", "age","black", "gender", "married", "employed", "privins", "medicaid"), 
                        remove_first_dummy = TRUE, remove_selected_columns = TRUE)
usdm::vifstep(dt_dummy[,-1])

## compare model fit
models <- list("Pois1" = pois1, "Pois1m" = pois1m)
rbind(AIC = sapply(models, function(x) AIC(x)), 
      BIC = sapply(models, function(x) BIC(x)), 
      logLik = sapply(models, function(x) logLik(x)))


hist(dt$ofp)
dt2 <- DebTrivedi[,c(5,7:19)]
pois2 <- glm(emer ~ ., data = dt2, family = poisson)

summary(pois2)

exp(0.223184)






# Poisson assumes mean = variance, is it ture in this data? any sign of over dispersion? 
# a quick&dirty check: variance ?= mean
mean(dt$ofp)
var(dt$ofp)
# overdispersion if the following value is significantly greater than 1; 
# use deviance residual; deviance residual describes deviation in log likelihood: extent to which the likelihood of the saturated model exceeds the likelihood of the proposed model
pois1$deviance/pois1$df.residual
## use pearson residual
## sum(residuals(pois1,type ="pearson")^2)/pois1$df.residual

# Energetic learner: formal test of over dispersion 
install.packages("AER")
library(AER)
dispersiontest(pois1) # Null: Var = Mean, no overdispersion (poisson is a good model)

#####################################################
## 1.2 Negative binomial regression
#####################################################
nbin <- MASS::glm.nb(ofp ~ ., data = dt)

# interpretation: 
exp(pois1m$coefficients) 
exp(nbin$coefficients) 

# how good is the dispersion now
nbin$deviance/nbin$df.residual

## Model comparison
# compare coefficients
cbind("Pois" =pois1m$coefficients,"NB" = nbin$coefficients)

models <- list("Pois" = pois1m, "NB" = nbin) # the results are stored in R as lists, use typeof() to check
sapply(models, function(x) coef(x)) # apply a function over a list or vector; user can define the function

# compare model fit
rbind(AIC = sapply(models, function(x) AIC(x)), 
      BIC = sapply(models, function(x) BIC(x)), 
      logLik = sapply(models, function(x) logLik(x))) 


# typically don't compare Pseudo R2 across different models (Poisson vs NB) as there is no universal definition of Pseudo R2 and the null is different under different models
#####################################################
# Energetic learners: quasi-poisson and zero-inflated models
# quasi-poisson addresses overdispersion
# zero inflated model addresses excessive zeros in data

# quasi-poisson grammar: family = quasipoisson
qpois<- glm(ofp ~ ., data = dt, family = quasipoisson)
summary(qpois)
# Zero inflated models (poisson and negative binomial)
fm_zinp <- zeroinfl(ofp ~ ., data = dt, dist = "poisson")
summary(fm_zinp)

fm_zinb <- zeroinfl(ofp ~ ., data = dt, dist = "negb")
summary(fm_zinb)



#####################################################
# 2. Logistic Regression 
#####################################################
# binary outcome

# load airline delay data
delay <- read.csv("OTP_2021_July_v2.csv", header=TRUE)

# dependent variable: DOT (Department of Transportation Definition of Flight Delay: >15min)
delay$arr_delay_DOT <- delay$arr_delay>15
table(delay$arr_delay_DOT)

prop.table(table(delay$arr_delay_DOT)) # tabulate percentage instead of frequency
# summarize by group: delay by airlines
summarise(group_by(delay,op_unique_carrier), perc_delay_DOT=mean(arr_delay_DOT))
# previously, we also used plotmeans() to visualize data by groups
plotmeans(arr_delay_DOT ~ op_unique_carrier, data=delay)
plotmeans(arr_delay_DOT ~ day_of_week, data=delay)
plotmeans(arr_delay_DOT ~ crs_arr_hour, data=delay)

# calculate scheduled air time (HW3)
delay$crs_air_time=60*(delay$crs_arr_hour- delay$crs_dep_hour) + (delay$crs_arr_min-delay$crs_dep_min)

########################################################################
# Logistic regression 

# Step 1: Run logistic regression model
# If outcome variable is a string, R code categorical variables alphabetically by default. 
log<-glm(arr_delay_DOT ~ as.factor(day_of_week), data=delay, family= binomial)

# how to interpret the results
summary(log)

# include more variables
log_1 <- glm(arr_delay_DOT ~ crs_air_time + distance + as.factor(op_unique_carrier) + as.factor(crs_arr_hour)
             + as.factor(day_of_week),  data=delay, family =binomial)
summary(log_1)

# Goodness of fit: AIC, BIC, logLikelihood Pseudo R2 
logLik(log_1)
AIC(log_1)
BIC(log_1)
1- log_1$deviance/log_1$null.deviance

# Step 2: Prediction: P(diagnosis=1 (M)) 
# result <- predict(log_1, delay) # linear prediction: XB
delay$pred_prob <- predict(log_1, delay, type="response") # same scale as the response variable, which gives us a probability between 0, 1
head(delay$pred_prob)

# Step 3: Convert prediction to (0,1) prediction
# suppose we use cutoff 0.5. If the predicted probability is greater than 0.5, then we predict 1 (DOT delay)
delay$pred <- 0
delay$pred[delay$pred_prob>0.5] <- 1 # predicted DOT delay = 1
str(delay)
# the above saved the variable as character (string); now factorize it
delay$pred <- as.logical(delay$pred) 
str(delay)

# Step 4: quality of prediction/model fit: Confusion matrix. Only takes factors as input
conf_table <- confusionMatrix(as.factor(delay$pred), as.factor(delay$arr_delay_DOT),positive="TRUE")
conf_table


#######################################################################
# Energetic learners:
# Effects of cutoff on classification accuracy 
install.packages("ROCR") # ROC (receiver operating characteristic) curve is a classification model at all classification thresholds
library(ROCR)
# create a prediction object first: containing the predicted values and the true values
pred<-ROCR::prediction(delay$pred_prob,delay$arr_delay_DOT) 
# plot accuracy against cutoff
acc<-performance(pred, "acc")
plot(acc)

# how cutoff affects sensibility and specificity
sens<-performance(pred, "sens")
spec<-performance(pred, "spec")
plot(sens, col="red")
plot(spec, col="blue")

# show three figures together
par(mfrow=c(1,3))
plot(acc, main="Accuracy vs. cuttoff")
plot(sens, col="red", main="Sensitivity vs. cuttoff")
plot(spec, col="blue", main="Specificity vs. cuttoff")
par(mfrow=c(1,1))


# plot sensitivity against specificity 
sens_spec<-performance(pred, "sens", "spec")
plot(sens_spec,colorize=TRUE)

# guess what does the following line of code do?
plot(performance(pred, "tpr", "fpr"),colorize=TRUE)



#######################################################################
# Probit model: assume normal distribution of the error term
log_2 <- glm(arr_delay_DOT ~ crs_air_time + distance + as.factor(op_unique_carrier) + as.factor(crs_arr_hour)
             + as.factor(day_of_week),  data=delay, family =binomial(link="probit"))
summary(log_2)


#Excercise
dt2<-DebTrivedi[, c(5,7:19)]


# run poisson regression
pois2 <- glm(emer ~ ., data = dt2, family = poisson)
summary(pois2)

# interpretation: 
exp(pois2$coefficients)

# fit
AIC(pois2)
BIC(pois2)
logLik(pois2)
1- pois2$deviance/pois2$null.deviance



delay$arr_delay_extreme <- delay$arr_delay>120

log_2 <- glm(arr_delay_extreme ~ crs_air_time + distance + as.factor(op_unique_carrier) + as.factor(crs_arr_hour)
             + as.factor(day_of_week),  data=delay, family =binomial)
summary(log_2)

# Goodness of fit: AIC, BIC, logLikelihood Pseudo R2 
logLik(log_2)
AIC(log_2)
BIC(log_2)
1- log_2$deviance/log_2$null.deviance

# Step 2: Prediction 
# result <- predict(log_1, delay) # linear prediction: XB
delay$pred_extreme_prob <- predict(log_2, delay, type="response") # same scale as the response variable, which gives us a probability between 0, 1
head(delay$pred_extreme_prob)

# Step 3: Convert prediction to (0,1) prediction
# suppose we use cutoff 0.1. If the predicted probability is greater than 0.5, then we predict 1 (DOT delay)
delay$pred_extreme <- 0
delay$pred_extreme[delay$pred_extreme_prob>0.1] <- 1 # predicted DOT delay = 1
str(delay)
# the above saved the variable as character (string); now factorize it
delay$pred_extreme <- as.logical(delay$pred_extreme) 
str(delay)

# Step 4: quality of prediction/model fit: Confusion matrix. Only takes factors as input
conf_table <- confusionMatrix(as.factor(delay$pred_extreme), as.factor(delay$arr_delay_extreme),positive="TRUE")
conf_table

# create a prediction object first: containing the predicted values and the true values
pred<-ROCR::prediction(delay$pred_extreme_prob,delay$arr_delay_extreme) 
# plot accuracy against cutoff
acc<-performance(pred, "acc")
plot(acc)


