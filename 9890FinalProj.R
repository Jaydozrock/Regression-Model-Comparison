# Install necessary packages
install.packages("glmnet")
install.packages("tidyverse")
install.packages("randomForest")
install.packages("gridExtra")
# Load packages
library(randomForest)
library(tidyverse)
library(glmnet)
library(gridExtra)

# Set working directory to the folder in which you dowloading all files and read the csv from the data folder in this repo.

# Read in data file and store in std.data variable
std.data<- read.csv("Put folder address for working director here")

##### Partioning and Modeling Data #####
# Ridge Model alpha = 0 
# Elastic Net Model alpha = 0.5
# Lasso Model  alpha = 1
set.seed(22)
n        =    dim(std.data)[1] # number of observations
p        =    dim(std.data)[2]-1
y        =   data.matrix(std.data %>% select(1)) # select target/response
X        =   data.matrix(std.data%>% select(2:52)) # select predictors
hist(y)
n.train        =     0.8*n
n.test         =     n-n.train

M              =     100
Rsq.test.rid   =     rep(0,M)  # rid = Ridge
Rsq.test.en    =     rep(0,M)  # en = Elastic net
Rsq.test.las   =     rep(0,M)  # las = Lasso
Rsq.test.rf    =     rep(0,M)  # rf= RandomForest
Rsq.train.rid  =     rep(0,M)
Rsq.train.en   =     rep(0,M)
Rsq.train.las  =     rep(0,M)
Rsq.train.rf   =     rep(0,M)

# Create empty vectors for residuals
Res.test.rid  =     rep(0,M)
Res.train.rid =     rep(0,M)
Res.test.en   =     rep(0,M)
Res.train.en  =     rep(0,M)
Res.test.las  =     rep(0,M)
Res.train.las =     rep(0,M)
Res.test.rf   =     rep(0,M)
Res.train.rf  =     rep(0,M)

# Empty Vectors to store time
Total.ridtime =     rep(0,M)
Total.entime  =     rep(0,M)
Total.lastime =     rep(0,M)
Total.rftime  =     rep(0,M)

start.entire = Sys.time()
start.models  = Sys.time()

for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train,]
  y.train          =     y[train]
  X.test           =     X[test,]
  y.test           =     y[test]
  
  # fit ridge and calculate and record the train and test R squares 
  a=0 # Ridge
  start.rid         <-  Sys.time()
  Ridcv.fit         =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  fit               =     glmnet(X.train, y.train, alpha = a, lambda = Ridcv.fit$lambda.min)
  end.rid          <-   Sys.time()
  y.train.hat       =     predict(fit, newx = X.train, type = "response") 
  y.test.hat        =     predict(fit, newx = X.test, type = "response") 
  Rsq.test.rid[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rid[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  Res.test.rid[m]   =     y.test - y.test.hat
  Res.train.rid[m]  =     y.train - y.train.hat
  Total.ridtime[m] <-   abs(Total.ridtime+(start.rid-end.rid)) # Calcs abs total time of Ridge
  
  # fit elastic-net and calculate and record the train and test R squares 
  a=0.5 # elastic-net
  start.en         <-  Sys.time()
  Encv.fit         =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train, alpha = a, lambda = Encv.fit$lambda.min)
  end.en          <-   Sys.time()
  y.train.hat      =     predict(fit, newx = X.train, type = "response") 
  y.test.hat       =     predict(fit, newx = X.test, type = "response") 
  Rsq.test.en[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.en[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  Res.test.en[m]   =     y.test - y.test.hat
  Res.train.en[m]  =     y.train - y.train.hat
  Total.entime[m] <-  abs(Total.entime+(start.en-end.en))
  
  # fit Lasso and calculate and record the train and test R squares 
  a=1 # Lasso
  start.las         <-   Sys.time()
  Lascv.fit         =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  fit               =     glmnet(X.train, y.train, alpha = a, lambda = Lascv.fit$lambda.min)
  end.las           <-   Sys.time()
  y.train.hat       =     predict(fit, newx = X.train, type = "response") 
  y.test.hat        =     predict(fit, newx = X.test, type = "response") 
  Rsq.test.las[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.las[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  Res.test.las[m]   =     y.test - y.test.hat
  Res.train.las[m]  =     y.train - y.train.hat
  Total.lastime[m] <-   abs(Total.lastime+(start.las-end.las))
  
  # fit RF and calculate and record the train and test R squares 
  start.rf         <-   Sys.time()
  rf               =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  end.rf           <-   Sys.time()
  y.test.hat       =     predict(rf, X.test)
  y.train.hat      =     predict(rf, X.train)
  Rsq.test.rf[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rf[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2) 
  Res.test.rf[m]   =     y.test - y.test.hat
  Res.train.rf[m]  =     y.train - y.train.hat
  Total.rftime[m] <-   abs(Total.rftime+(start.rf-end.rf))

  cat(sprintf("m=%3.f| Rsq.test.rid=%.2f, Rsq.test.en=%.2f| Rsq.test.las=%.2f,Rsq.test.rf=%.2f| \n",
              m,  Rsq.test.rid[m], Rsq.test.en[m],  Rsq.train.las[m], Rsq.train.rf[m]))
}
end.models = Sys.time()
total.models = abs(start.models - end.models)

##### BoxPlots of R-Square Test and R-quare train ##### ########
# Create dataframes to store values for plots
rtt = data.frame(Model = "Ridge", value = Rsq.test.rid )
ent = data.frame(Model = "Elastic-Net", value = Rsq.test.en )
lat = data.frame(Model = "Lasso", value = Rsq.test.las )
rft = data.frame(Model = "Random Forest", value = Rsq.test.rf )

rtr = data.frame(Model = "Ridge", value = Rsq.train.rid )
entr = data.frame(Model = "Elastic-Net", value = Rsq.train.en )
latr = data.frame(Model= "Lasso", value = Rsq.train.las )
rftr = data.frame(Model = "Random Forest", value = Rsq.train.rf )

plot.test = rbind(rtt,ent,lat,rft) # this function will bind or join the rows.
plot.train = rbind(rtr,entr,latr,rftr)


box.test <-ggplot(plot.test, aes(x=Model, y=value, fill=Model)) + 
          geom_boxplot()+    
          labs(title = "Boxplots of Models",
               subtitle = "based on R-Square Test")

box.train <-ggplot(plot.train, aes(x=Model, y=value, fill=Model)) +  
            geom_boxplot() + labs(title = "Boxplots of Models",
                                  subtitle = "based on R-Square Train")
grid.arrange(box.train, box.test, nrow = 1) # Arrange Boxplots 

#### Cross Validation Plots ###############
par(mfrow=c(2,2))
r <- plot(Ridcv.fit, sub="Ridge",
          cex.main=2.1, cex.lab=1)
e <- plot(Encv.fit, sub="Elastic Net",
          cex.main=1.9, cex.lab=1)
l <- plot(Lascv.fit, sub="Lasso",
          cex.main=2.1, cex.lab=1)

#### Residuals BatPlots####
#graphics.off()
res.rd.te = data.frame(Model = "Ridge", value = Res.test.rid)
res.en.te = data.frame(Model = "Elastic-Net", value = Res.test.en )
res.la.te = data.frame(Model = "Lasso", value = Res.test.las )
res.rf.te = data.frame(Model = "Random Forest", value = Res.test.rf )

res.rd.tr = data.frame(Model = "Ridge", value = Res.train.rid )
res.en.tr = data.frame(Model = "Elastic-Net", value = Res.train.en )
res.la.tr = data.frame(Model= "Lasso", value = Res.train.las )
res.rf.tr = data.frame(Model = "Random Forest", value = Res.test.rf )

plot.test.res = rbind(res.rd.te,res.en.te,res.la.te,res.rf.te) 
plot.train.res = rbind(res.rd.tr,res.en.tr,res.la.tr,res.rf.tr)

res.test <-ggplot(plot.test.res, aes(x=Model, y=value, fill=Model)) +  
  geom_boxplot()+   
  labs(title = "Boxplots of Models",
       subtitle = "based on Test Residuals")

res.train <-ggplot(plot.train.res, aes(x=Model, y=value, fill=Model)) +  
  geom_boxplot() + labs(title = "Boxplots of Models",
                        subtitle = "based on Train Residuals")
grid.arrange(res.train, res.test, nrow = 1) # Arrange Boxplots 

#### Bootstrap Barplots ####
bootstrapSamples  <-   100
beta.rid.bs       <-   matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.en.bs        <-   matrix(0, nrow = p, ncol = bootstrapSamples) 
beta.las.bs       <-   matrix(0, nrow = p, ncol = bootstrapSamples)
beta.rf.bs        <-   matrix(0, nrow = p, ncol = bootstrapSamples)

start.boot = Sys.time()
for (m in 1:bootstrapSamples){
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  # Ridge Bootstrap
  a                =     0 
  rcv.fit          =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, alpha = a, lambda = rcv.fit$lambda.min)  
  beta.rid.bs[,m]  =     as.vector(fit$beta)

  # E-Net Boostrap
  a                =     0.5 # elastic-net
  elcv.fit         =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, alpha = a, lambda = elcv.fit$lambda.min)  
  beta.en.bs[,m]   =     as.vector(fit$beta)
  
  # Lasso Bootstrap
  a                =     1 # lasso
  lacv.fit         =     cv.glmnet(X.bs, y.bs, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, alpha = a, lambda = lacv.fit$lambda.min)  
  beta.las.bs[,m]  =     as.vector(fit$beta)
  
  # Random Forrest Bootstrap
  rf               =     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE)
  beta.rf.bs[,m]   =     as.vector(rf$importance[,1])
  cat(sprintf("Bootstrap Sample %3.f \n", m))
}
end.boot = Sys.time()
Total.boot <-start.boot - end.boot

# Calculating Bootstrap Standard Error
rid.bs.sd <- apply(beta.rid.bs, 1, "sd")
en.bs.sd  <- apply(beta.en.bs, 1, "sd")
las.bs.sd <- apply(beta.las.bs, 1, "sd")
rf.bs.sd  <- apply(beta.rf.bs, 1, "sd")

# Fitting Ridge to entire data
a=0 
rcv.fit  <- cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.rid  <- glmnet(X, y, alpha = a, lambda = rcv.fit$lambda.min)

# Fitting Elastic-Net to entire data
a=0.5
elascv.fit  <-  cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.en      <-  glmnet(X, y, alpha = a, lambda = elascv.fit$lambda.min)

# Fitting Lasso to entire data
a=1 
lcv.fit <- cv.glmnet(X, y, alpha = a, nfolds = 10)
fit.las <- glmnet(X, y, alpha = a, lambda = lcv.fit$lambda.min)

# Fitting Random Forest to entire data
rf <- randomForest(X, y, mtry = sqrt(p), importance = TRUE)

# Storing respective Betas into dataframes
betaS.rid             <- data.frame(c(1:p), as.vector(fit.rid$beta), 2*rid.bs.sd)
colnames(betaS.rid)   <-     c( "feature", "value", "err")

betaS.en              <-     data.frame(c(1:p), as.vector(fit.en$beta), 2*en.bs.sd)
colnames(betaS.en)    <-     c( "feature", "value", "err")

betaS.las             <-    data.frame(c(1:p), as.vector(fit.las$beta), 2*las.bs.sd)
colnames(betaS.las)   <-     c( "feature", "value", "err")

betaS.rf               <-     data.frame(c(1:p), as.vector(rf$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf)     <-    c( "feature", "value", "err")

# Modifying the order of Betas
betaS.rid$feature    =  factor(betaS.rid$feature,
                              levels = betaS.rid$feature[order(betaS.rid$value, decreasing = TRUE)])
betaS.en$feature     =  factor(betaS.en$feature,
                               levels = betaS.en$feature[order(betaS.en$value, decreasing = TRUE)])
betaS.las$feature    =  factor(betaS.las$feature,
                              levels = betaS.las$feature[order(betaS.las$value, decreasing = TRUE)])
betaS.rf$feature     =  factor(betaS.rf$feature,
                               levels = betaS.rf$feature[order(betaS.rf$value, decreasing = TRUE)])
# Plotting Barplots with Error bars
RidPlot =  ggplot(betaS.rid, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + 
  labs(title = "Ridge Beta Barplots")+  theme_dark()+
  theme(axis.text.x = element_text(size = 15,angle=90, vjust = 0.5,face='bold'),
        plot.title = element_text(size=13))
EnPlot =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + 
  labs(title = "Elastic-Net Beta Barplots")+  theme_dark()+
  theme(axis.text.x = element_text(size = 15,angle=90, vjust = 0.5,face='bold'),
        plot.title = element_text(size=13))
LasPlot =  ggplot(betaS.las, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) + 
  labs(title = "Lasso Beta Barplots")+  theme_dark()+
  theme(axis.text.x = element_text(size = 15,angle=90, vjust = 0.5,face='bold'),
        plot.title = element_text(size=13))
RfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  labs(title = "Random Forest Barplots")+ theme_dark()+
  theme(axis.text.x = element_text(size = 15,angle=90, vjust = 0.5,face='bold'),
        plot.title = element_text(size=13))
grid.arrange(RidPlot,EnPlot, nrow = 2)
grid.arrange(LasPlot,RfPlot, nrow = 2)

end.entire = Sys.time()
total.time <-abs(start.entire - end.entire)

# Calculating time stats
total.time # total time for enitre run
total.models # total time for models

min(Total.ridtime) 
min(Total.entime)  
min(Total.lastime) 
min(Total.rftime)

mean(Total.ridtime) 
mean(Total.entime)  
mean(Total.lastime) 
mean(Total.rftime)

max(Total.ridtime) 
max(Total.entime)  
max(Total.lastime) 
max(Total.rftime)

