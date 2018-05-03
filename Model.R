
#**********************************************************************************************************
#**********************************************************************************************************
#**********************************************************************************************************
#Save and Load worksapce
save.image(file='D:/HP Backup 02Oct2017/177565(35gb)/UChicago/Data Mining Principles/Project/Analysis/Saved_Workspace.RData')
 load(file='D:/HP Backup 02Oct2017/177565(35gb)/UChicago/Data Mining Principles/Project/Analysis/Saved_Workspace.RData')

#**********************************************************************************************************
#**********************************************************************************************************
#**********************************************************************************************************
#libraries
library(caret)
library(fitdistrplus)
library(rpart)
library(partykit)
#library(RWeka)
library(ipred)
library(gbm)
library(mice)
library(VIM)
library(missMDA)
library(missForest)
library(randomForest)
library(Metrics)
library(hydroGOF)
library(neuralnet)
 
#**********************************************************************************************************
#**********************************************************************************************************
#**********************************************************************************************************
#Read data

train <- read.csv("D:/HP Backup 02Oct2017/177565(35gb)/UChicago/Data Mining Principles/Project/train.csv",header=TRUE)
test <- read.csv("D:/HP Backup 02Oct2017/177565(35gb)/UChicago/Data Mining Principles/Project/test.csv",header=TRUE)

#**********************************************************************************************************
#**********************************************************************************************************
#**********************************************************************************************************
#Remove near zero variance variables

nearZeroVar(train)
train.nearZeroVar <- train[,c(6,  9, 10, 12, 15, 23, 32, 36, 37, 40, 46, 53, 56, 64, 65, 69, 70, 71, 72, 75, 76)]
head(train.nearZeroVar)
write.csv(train.nearZeroVar,'D:/HP Backup 02Oct2017/177565(35gb)/UChicago/Data Mining Principles/Project/Analysis/train.nearZeroVar.csv')
names(train.nearZeroVar)

# "Street"        "LandContour"   "Utilities"     "LandSlope"     "Condition2"    "RoofMatl"      "BsmtCond"     
# "BsmtFinType2"  "BsmtFinSF2"    "Heating"       "LowQualFinSF"  "KitchenAbvGr"  "Functional"    "GarageQual"   
# "GarageCond"    "EnclosedPorch" "X3SsnPorch"    "ScreenPorch"   "PoolArea"      "MiscFeature"   "MiscVal" 

with(train.nearZeroVar,table(Street))
with(train.nearZeroVar,table(KitchenAbvGr))
#0    1    2    3 
#1 1392   65    2 

with(train,hist(SalePrice))
f1n <- fitdistr(train$SalePrice,"normal")
summary(f1n)
f2n <- fitdist(train$SalePrice,"norm")
plot(f2n)
ks.test(train$SalePrice, "pnorm", 180921.20, 79415.29)
#data:  train$SalePrice
#D = 0.12367, p-value < 2.2e-16
#alternative hypothesis: two-sided

f3n <- fitdist(log(train$SalePrice),"norm")
plot(f3n)
ks.test(log(train$SalePrice), "pnorm", 12.024051, 0.399315)

#data:  log(train$SalePrice)
#D = 0.040868, p-value = 0.01524
#alternative hypothesis: two-sided
bc <- BoxCoxTrans(train$SalePrice)
bc1 <- predict(bc, train$SalePrice)
f4n <- fitdist(bc1,"norm")
plot(f4n)
ks.test(bc1, "pnorm", 12.024051, 0.399315)

train.bc <- as.data.frame(cbind(SalePrice.bc=log(train$SalePrice),train.nearZeroVar))
head(train.bc)
anova(lm(SalePrice.bc ~ Street,data=train.bc))

train.bc.anovaotpt <- vector(mode = "list", length = 21)

for (i in 1:21){
train.bc.anovaotpt[[i]] <- summary(lm(SalePrice.bc ~ train.bc[,i+1],data=train.bc))
}

write.csv(train.bc.anovaotpt,'D:/HP Backup 02Oct2017/177565(35gb)/UChicago/Data Mining Principles/Project/Analysis/train.bc.anovaotpt.csv')

sink("D:/HP Backup 02Oct2017/177565(35gb)/UChicago/Data Mining Principles/Project/Analysis/train.bc.anovaotpt.txt")
print(train.bc.anovaotpt)
sink()

names(train.nearZeroVar)

#The below low variance variables will not be dropped since ANOVA test on log of dep var reveals significant 
#differences between individual categories and R-Sq of lm is above 1%

#LandContour Condition2 BsmtCond BsmtFinSF2  KitchenAbvGr Functional GarageQual EnclosedPorch ScreenPorch MiscFeature

train1 <- train[,-which(names(train) %in% c("Street",   "Utilities",     "LandSlope",
                    "RoofMatl",     "BsmtFinType2",    "Heating",       
                    "LowQualFinSF",  "Functional",    "GarageCond", "X3SsnPorch",   "PoolArea",
                    "MiscVal" ))]

#**********************************************************************************************************
#**********************************************************************************************************
#**********************************************************************************************************
#Missing value imputation

sapply(train1, function(x) sum(is.na(x)))
length(train1)

#ANOVA on box cox transformed dependent variable for normal dist, and find whether high missing value 
#predictors have categories that are significantly different
train1.m <- train1[,which(names(train1) %in% c("SalePrice","Alley",
                                                "FireplaceQu",
                                                "PoolQC",
                                                "Fence",
                                                "MiscFeature"))]
head(train1.m,10)
summary(train1.m)

re_levels <- 
  function(col) {
    if (is.factor(col))  levels(col) <- c(levels(col), "M")
    col
  }

df <- sapply(train1.m[,c(1,2,3,4,5)],re_levels)
df[is.na(df)] <-  "M"
head(df,10)
train1.m <-as.data.frame(cbind(SalePrice=as.numeric(train1.m$SalePrice),df))
head(train1.m,10)
train1.m$SalePrice <- as.numeric(as.character(train1.m$SalePrice))
summary(train1.m)

bc.m <- BoxCoxTrans(train1.m$SalePrice)
bc1.m <- predict(bc.m, train1.m$SalePrice)
f5n <- fitdist(bc1.m,"norm")
plot(f5n)
ks.test(bc1.m, "pnorm", 12.024051, 0.399315)

train.bc.m <- as.data.frame(cbind(SalePrice.bc.m=log(train1.m$SalePrice),train1.m))
head(train.bc.m)
summary(train.bc.m)

train.m.anovaotpt <- vector(mode = "list", length = 5)

for (i in 1:5){
  train.m.anovaotpt[[i]] <- summary(lm(SalePrice.bc.m ~ train.bc.m[,i+2],data=train.bc.m))
}

summary(lm(SalePrice.bc.m ~ train.bc.m[,7],data=train.bc.m))

sink("D:/HP Backup 02Oct2017/177565(35gb)/UChicago/Data Mining Principles/Project/Analysis/train.m.anovaopt.txt")
print(train.m.anovaotpt)
sink()

names(train.bc.m)
#Though 5 variables have missing values in the range of 50 to 99%, 3 are retained as anova tests show 
#singificant difference in categories with greater than 1% R-Squared

train2 <- train1[,-which(names(train1) %in% c("PoolQC","MiscFeature" ))]
names(train2)

#Converting all missing factor variable categories to "Missing"

re_levels <- 
  function(col) {
    if (is.factor(col))  levels(col) <- c(levels(col), "Missing")
    col
  }


df.all <- sapply(train2[,which(names(train2) %in% c("Alley","FireplaceQu","Fence" ))],re_levels)
df.all[is.na(df.all)] <-  "Missing"
head(df.all,10)
train2 <-as.data.frame(cbind(train2[,-which(names(train2) %in% c("Alley","FireplaceQu","Fence" ))],df.all))
head(train2,10)
summary(train2)


#train2$Alley <- addNA(train2$Alley)
#train2$FireplaceQu <- addNA(train2$FireplaceQu)
#train2$Fence <- addNA(train2$Fence)

#**********************************************************************************************************
#**********************************************************************************************************
#**********************************************************************************************************
#Build Single Tree - Recursive Partitioning and Conditional Inference
datarpart <- train2[,-which(names(train2) %in% c('Id'))]
train.tree <- rpart(formula=SalePrice~.,data=datarpart,
        control=rpart.control(method="anova",cp=0,minsplit=37,xval=10,maxsurrogate=3))

predrpart <- predict(train.tree)
cor(datarpart[,66],predrpart)^2

#RSq 0.7948943
names(train.tree)

#Plot tree
par(mai=c(0.01,0.01,0.01,0.01))
plot(train.tree,col=5, compress=TRUE, branch=0.9,uniform=TRUE)
text(train.tree,cex=0.55,col=10,use.n=TRUE)


#Print cost-complexity parameters of the 10-fold cross-validated models
#along with cross-validation errors

printcp(train.tree)

#Plot cost-complexity parameters of the 10-fold cross-validated models
#along with cross-validation errors

plotcp(train.tree,minline=TRUE,col=4)

#Build pruned  tree
train.tree.pr = rpart(formula=SalePrice~.,data=datarpart,
                      control=rpart.control(method="anova",cp=0.020665,minsplit=37,xval=10,maxsurrogate=3))

predrpart.pr <- predict(train.tree.pr)
cor(datarpart[,63],predrpart.pr)^2

#Rsq 0.670893

rmsle(datarpart[,63],predrpart.pr)

#Plot pruned  tree
par(mai=c(0.01,0.01,0.01,0.01))
plot(train.tree.pr,col=5, compress=TRUE, branch=0.9,uniform=TRUE)
text(train.tree.pr,cex=0.7,col=10,use.n=TRUE)

#Predict test and submit rpart

head(test)

predrpart.test <- predict(train.tree.pr,test)
head(predrpart.test)
length(predrpart.test)
dim(test)
names(predrpart.test)
submission.rpart <- as.data.frame(cbind(Id=test$Id,SalePrice=predrpart.test))

write.csv(submission.rpart,"D:/HP Backup 02Oct2017/177565(35gb)/UChicago/Data Mining Principles/Project/Submissions/submission.rpart.csv",row.names = F)

#**********************************************************************************************************
#**********************************************************************************************************
#**********************************************************************************************************
#Bagging

baggedTree <- bagging(SalePrice ~ ., data = datarpart, coob=TRUE, nbagg=25,
                      control=rpart.control(method="anova",cp=0,minsplit=18,xval=10,maxsurrogate=3))
predbagged <- predict(baggedTree,newdata=datarpart)
cor(datarpart[,63],predbagged)^2

#RSq 0.9254848

names(baggedTree)
summary(baggedTree)
baggedTree$err
#33494.67
predbagging.test <- predict(baggedTree,newdata=test)
head(predbagging.test)
length(predbagging.test)
dim(test)
submission.bagging <- as.data.frame(cbind(Id=test$Id,SalePrice=predbagging.test))

write.csv(submission.bagging,"D:/HP Backup 02Oct2017/177565(35gb)/UChicago/Data Mining Principles/Project/Submissions/submission.bagging.csv",row.names = F)

#**********************************************************************************************************
#**********************************************************************************************************
#**********************************************************************************************************

#Conditional Random Forests - cforest and randomForest cant handle missing values
crfTree <- cforest(formula=SalePrice ~ ., data = datarpart.missForest, ntree=100, mtry=33, trace=TRUE)
predcrf <- predict(crfTree,newdata=datarpart)
cor(datarpart[,63],predcrf)^2

#RSq 0.9254848

names(crfTree)
summary(crfTree)
predcrf.test <- predict(crfTree,newdata=test)
head(predcrf.test)
length(predcrf.test)
dim(test)
submission.crf <- as.data.frame(cbind(Id=test$Id,SalePrice=predcrf.test))

write.csv(submission.crf,"D:/HP Backup 02Oct2017/177565(35gb)/UChicago/Data Mining Principles/Project/Submissions/submission.crf.csv",row.names = F)
?randomForest
#**********************************************************************************************************
#**********************************************************************************************************
#**********************************************************************************************************

#Random Forests

mtryGrid <- expand.grid(mtry = seq(15,45,by=3)) 
rfTune<- train(x=datarpart.missForest[,-which(colnames(datarpart.missForest)=="SalePrice")],
               y=datarpart.missForest$SalePrice,
               method = "rf",
               trControl = trainControl(method="oob"),
               metric = "RMSE",
               ntree = c(500),
               tuneGrid = mtryGrid,
               sampsize = c(1200),
               nodesize = c(5),
               allowParallel=TRUE)
rfTune$bestTune
rfTune$results
#ntree = 500, sampsize = 400, nodesize = 10
#30071.70 0.8566136   36
#ntree = 500, sampsize = 600, nodesize = 10
#29160.68 0.8651698   33
#ntree = 500, sampsize = 800, nodesize = 10
#28573.29 0.8705469   33
#ntree = 500, sampsize = 1000, nodesize = 10
#28077.11 0.8750039   21
#ntree = 500, sampsize = 1200, nodesize = 10
#27843.17 0.8770781   24

#ntree = 500, sampsize = 1200, nodesize = 20
#28466.83 0.8715098   21
#ntree = 500, sampsize = 1200, nodesize = 30
#29549.47 0.8615506   24

#ntree = 500, sampsize = 1200, nodesize = 5
#27659.24 0.8786969   21

mtryGrid <- expand.grid(mtry = seq(15,45,by=3)) 
rfTune<- train(x=datarpart.missForest[,-which(colnames(datarpart.missForest)=="SalePrice")],
               y=datarpart.missForest$SalePrice,
               method = "rf",
               trControl = trainControl(method="oob"),
               metric = "RMSE",
               ntree = c(600),
               tuneGrid = mtryGrid,
               sampsize = c(1200),
               nodesize = c(5),
               allowParallel=TRUE)
rfTune$bestTune
rfTune$results
#ntree = 600, sampsize = 400, nodesize = 10
#30137.25 0.8559879   27
#ntree = 600, sampsize = 600, nodesize = 10
#28987.81 0.8667637   18
#ntree = 600, sampsize = 800, nodesize = 10
#28520.27 0.8710269   33
#ntree = 600, sampsize = 1000, nodesize = 10
#28039.63 0.8753374   15
#ntree = 600, sampsize = 1200, nodesize = 10
#27972.42 0.8759343   21

#ntree = 600, sampsize = 1200, nodesize = 20
#28656.49 0.8697920   21
#ntree = 600, sampsize = 1200, nodesize = 20
#29503.68 0.8619793   30

#ntree = 600, sampsize = 1200, nodesize = 5
#27538.95 0.8797496   18

mtryGrid <- expand.grid(mtry = seq(15,45,by=3)) 
rfTune<- train(x=datarpart.missForest[,-which(colnames(datarpart.missForest)=="SalePrice")],
               y=datarpart.missForest$SalePrice,
               method = "rf",
               trControl = trainControl(method="oob"),
               metric = "RMSE",
               ntree = c(700),
               tuneGrid = mtryGrid,
               sampsize = c(600),
               nodesize = c(5),
               allowParallel=TRUE)
rfTune$bestTune
rfTune$results
#ntree = 700, sampsize = 400, nodesize = 10
#30017.51 0.8571300   39
#ntree = 700, sampsize = 600, nodesize = 10
#29158.03 0.8651943   33

#ntree = 700, sampsize = 1200, nodesize = 5
#28828.83 0.8682211   24

mtryGrid <- expand.grid(mtry = seq(15,45,by=3)) 
rfTune<- train(x=datarpart.missForest[,-which(colnames(datarpart.missForest)=="SalePrice")],
               y=datarpart.missForest$SalePrice,
               method = "rf",
               trControl = trainControl(method="oob"),
               metric = "RMSE",
               ntree = c(800),
               tuneGrid = mtryGrid,
               sampsize = c(1000),
               nodesize = c(10),
               allowParallel=TRUE)
rfTune$bestTune
rfTune$results
#ntree = 800, sampsize = 400, nodesize = 10
#29983.80 0.8574507   24
#ntree = 800, sampsize = 600, nodesize = 10
#29174.94 0.8650379   27
#ntree = 800, sampsize = 800, nodesize = 10
#28538.57 0.8708614   27
#ntree = 800, sampsize = 1000, nodesize = 10
#28163.03 0.8742377   21

mtryGrid <- expand.grid(mtry = seq(15,45,by=3)) 
rfTune<- train(x=datarpart.missForest[,-which(colnames(datarpart.missForest)=="SalePrice")],
               y=datarpart.missForest$SalePrice,
               method = "rf",
               trControl = trainControl(method="oob"),
               metric = "RMSE",
               ntree = c(900),
               tuneGrid = mtryGrid,
               sampsize = c(600),
               nodesize = c(10),
               allowParallel=TRUE)
rfTune$bestTune
rfTune$results
#ntree = 900, sampsize = 400, nodesize = 10
#30053.79 0.8567844   21
#ntree = 900, sampsize = 600, nodesize = 10
#29137.93 0.8653802   30

mtryGrid <- expand.grid(mtry = seq(15,45,by=3)) 
rfTune<- train(x=datarpart.missForest[,-which(colnames(datarpart.missForest)=="SalePrice")],
               y=datarpart.missForest$SalePrice,
               method = "rf",
               trControl = trainControl(method="oob"),
               metric = "RMSE",
               ntree = c(1000),
               tuneGrid = mtryGrid,
               sampsize = c(1000),
               nodesize = c(10),
               allowParallel=TRUE)
rfTune$bestTune
rfTune$results
#ntree = 1000, sampsize = 400, nodesize = 10
#30085.99 0.8564774   33
#ntree = 1000, sampsize = 600, nodesize = 10
#29127.55 0.8654760   21
#ntree = 1000, sampsize = 800, nodesize = 10
#28551.05 0.8707484   27
#ntree = 1000, sampsize = 1000, nodesize = 10
#28150.91 0.8743459   27

rftrain <- datarpart.missForest[,-which(colnames(datarpart.missForest) %in% c("GarageQual",
                                                                              "Electrical",
                                                                              "Exterior1st",
                                                                              "Exterior2nd",
                                                                              "HouseStyle",
                                                                              "Condition2"))]

rftest <- datarpart.missForest.test[,-which(colnames(datarpart.missForest) %in% c("GarageQual",
                                                                                  "Electrical",
                                                                                  "Exterior1st",
                                                                                  "Exterior2nd",
                                                                                  "HouseStyle",
                                                                                  "Condition2"))]

str(rftrain)
str(rftest)

rfModel <- randomForest(x=rftrain[,-c(1)],
                        y=rftrain$SalePrice,
                        importance=TRUE,
                        ntree=600,
                        nodesize=5,
                        sampsize=1200,
                        mtry=18,
                        metric="RMSE",
                        allowParallel=TRUE)
rfModel$importance

rftree <- predict(rfModel,newdata=rftrain,type='response')
cor(rftrain[,1],rftree)^2
# 0.97762
rmse(rftrain[,1],rftree)
#13140.53
rmsle(rftrain[,1],rftree)

predrf.test <- predict(rfModel,newdata=rftest,type='response')
head(predrf.test)
length(predrf.test)
dim(test)
submission.rf <- as.data.frame(cbind(Id=datarpart.missForest.test$Id,SalePrice=predrf.test))
head(submission.rf)
write.csv(submission.rf,"D:/HP Backup 02Oct2017/177565(35gb)/UChicago/Data Mining Principles/Project/Submissions/submission.rf.csv",row.names = F)

#**********************************************************************************************************
#**********************************************************************************************************
#**********************************************************************************************************

#Boosting
gbmModel <- gbm.fit(solTrainXtrans, solTrainY, distribution = "gaussian")
## or
gbmModel <- gbm(SalePrice ~ ., data = datarpart, n.trees=5000,cv.folds=10,n.minobsinnode=18,
                shrinkage=0.1,bag.fraction=0.7)
gbmtree <- predict(gbmModel,newdata=datarpart)
cor(train[,81],gbmtree)^2
names(train)

#first parms
trainControl <- trainControl(method="cv", number=10)

gbmGrid <- expand.grid(.interaction.depth = seq(3,7,by=2),
                          .n.trees = seq(600,1000,by=200),
                          .shrinkage = seq(0.01,0.10,by=0.03),
                          .n.minobsinnode = seq(10,30,by=10))

nrow(gbmGrid)
set.seed(100)
head(datarpart.missForest)
gbmTune <- train(x=datarpart.missForest[,-which(colnames(datarpart.missForest)=="SalePrice")],
                 y=datarpart.missForest$SalePrice,
                 method = "gbm",
                 distribution="gaussian",
                 trControl=trainControl,
                 tuneGrid = gbmGrid,
                 bag.fraction=0.7,
                 metric="RMSE",
                 verbose = FALSE)
names(gbmTune)
gbmTune$bestTune
gbmTune$finalModel
#n.trees interaction.depth shrinkage n.minobsinnode
#  1000                 5      0.07             30

gbmModel <- gbm(SalePrice ~ ., data = datarpart.missForest, 
                n.trees=1000,
                interaction.depth=5,
                distribution="gaussian",
                n.minobsinnode=30,
                shrinkage=0.07,
                cv.folds=10,
                bag.fraction=0.7)

gbm.perf(gbmModel, method="cv")
gbmtree <- predict.gbm(gbmModel,newdata=datarpart.missForest,928)
cor(datarpart.missForest[,1],gbmtree)^2
# 0.9902853
rmse(datarpart.missForest[,1],gbmtree)
#7869.629

#second parms
gbmGrid1 <- expand.grid(.interaction.depth = seq(3,7,by=2),
                       .n.trees = seq(800,1200,by=200),
                       .shrinkage = seq(0.04,0.08,by=0.01),
                       .n.minobsinnode = seq(20,40,by=10))

nrow(gbmGrid1)

set.seed(100)
gbmTune1 <- train(x=datarpart.missForest[,-which(colnames(datarpart.missForest)=="SalePrice")],
                 y=datarpart.missForest$SalePrice,
                 method = "gbm",
                 distribution="gaussian",
                 trControl=trainControl,
                 tuneGrid = gbmGrid1,
                 bag.fraction=0.7,
                 metric="RMSE",
                 verbose = FALSE)

names(gbmTune1)
gbmTune1$finalModel
gbmTune1$bestTune

#n.trees interaction.depth shrinkage n.minobsinnode
#  1000                 5      0.08             30

gbmModel <- gbm(SalePrice ~ ., data = datarpart.missForest, 
                n.trees=1000,
                interaction.depth=5,
                distribution="gaussian",
                n.minobsinnode=30,
                shrinkage=0.08,
                cv.folds=10,
                bag.fraction=0.7)

gbm.perf(gbmModel, method="cv")
gbmtree <- predict.gbm(gbmModel,newdata=datarpart.missForest,928)
cor(datarpart.missForest[,1],gbmtree)^2
# 0.9917373
rmse(datarpart.missForest[,1],gbmtree)
#7258.247
rmsle(datarpart.missForest[,1],gbmtree)

#Used the second model
predgbm.test <- predict.gbm(gbmModel,newdata=datarpart.missForest.test,928)
head(predgbm.test)
length(predgbm.test)
dim(test)
submission.gbm <- as.data.frame(cbind(Id=datarpart.missForest.test$Id,SalePrice=predgbm.test))
head(submission.gbm)
write.csv(submission.gbm,"D:/HP Backup 02Oct2017/177565(35gb)/UChicago/Data Mining Principles/Project/Submissions/submission.gbm.csv",row.names = F)

names(gbmModel)
summary(gbmModel)
#**********************************************************************************************************
#**********************************************************************************************************
#**********************************************************************************************************

#Imputing missing data
md.pattern(datarpart)
aggr_plot <- aggr(datarpart, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE,
                  labels=names(data), cex.axis=.7, gap=3, 
                  ylab=c("Histogram of missing data","Pattern"))


xmis<-datarpart[,-which(names(datarpart) %in% c('SalePrice'))]
xmis_complement<-datarpart[,which(names(datarpart) %in% c('SalePrice'))]
summary(xmis)
ImputeForest <- missForest(xmis, ntree=100, variablewise=TRUE,decreasing=TRUE,
                           mtry = floor(sqrt(ncol(xmis))),replace=TRUE,verbose=TRUE)
                          
names(xmis)
ImputeForest$OOBerror
names(ImputeForest$ximp)
datarpart.missForest <- cbind(SalePrice=xmis_complement,ImputeForest$ximp)
summary(datarpart.missForest)
sapply(datarpart.missForest, function(x) sum(is.na(x)))

#Impute on test

xmis<-test[,-which(names(test) %in% c("Street",   "Utilities",     "LandSlope",
                    "RoofMatl",     "BsmtFinType2",    "Heating",       
                    "LowQualFinSF",  "Functional",    "GarageCond", "X3SsnPorch",   "PoolArea",
                    "MiscVal",
                    "PoolQC","MiscFeature",
                    "Id"))]

xmis_complement<-test[,which(names(test) %in% c('Id'))]
summary(xmis)

#Converting all missing factor variable categories to "Missing"
re_levels <- 
  function(col) {
    if (is.factor(col))  levels(col) <- c(levels(col), "Missing")
    col
  }


df.all <- sapply(xmis[,which(names(xmis) %in% c("Alley","FireplaceQu","Fence" ))],re_levels)
df.all[is.na(df.all)] <-  "Missing"
head(df.all,10)
xmis <-as.data.frame(cbind(xmis[,-which(names(xmis) %in% c("Alley","FireplaceQu","Fence" ))],df.all))
head(xmis,10)
summary(xmis)

ImputeForest.test <- missForest(xmis, ntree=100, variablewise=TRUE,decreasing=TRUE,
                           mtry = floor(sqrt(ncol(xmis))),replace=TRUE,verbose=TRUE
)

ImputeForest.test$OOBerror
names(ImputeForest.test$ximp)
datarpart.missForest.test <- cbind(Id=xmis_complement,ImputeForest.test$ximp)
str(datarpart.missForest.test)
class(datarpart.missForest.test)
sapply(datarpart.missForest.test, function(x) sum(is.na(x)))
head(datarpart.missForest.test)

#**********************************************************************************************************
#**********************************************************************************************************
#**********************************************************************************************************

#Neural Network
train.num <- datarpart.missForest[,sapply(datarpart.missForest, is.numeric)]
test.num <- datarpart.missForest.test[,sapply(datarpart.missForest.test, is.numeric)]
train.num <- train.num[,-c(1)]
test.num <- test.num[,-c(1)]

str(train.num)
str(test.num)

maxs <- apply(train.num, 2, max) 
mins <- apply(train.num, 2, min)

scaled.train <- as.data.frame(scale(train.num, center = mins, scale = maxs - mins))
scaled.test <- as.data.frame(scale(test.num, center = mins, scale = maxs - mins))

str(scaled.train)
str(scaled.test)

nntrain <- as.data.frame(cbind(SalePrice=datarpart.missForest[,c(1)],datarpart.missForest[,sapply(datarpart.missForest, is.numeric)=="FALSE"],scaled.train))
nntest <- as.data.frame(cbind(Id=datarpart.missForest.test[,c(1)],datarpart.missForest.test[,sapply(datarpart.missForest.test, is.numeric)=="FALSE"],scaled.test))

str(nntrain)
names(nntrain)
str(nntest)

nntrain$SalePrice1 <-   (nntrain$SalePrice - min(nntrain$SalePrice))/(max(nntrain$SalePrice) - min(nntrain$SalePrice)) 

#GarageQual
#Electrical
#Exterior1st
#Exterior2nd
#HouseStyle
#Condition2

nntrain.format <- as.data.frame(model.matrix( SalePrice1 ~    MSZoning  +    LotShape +     LandContour +  LotConfig  +   Neighborhood +  Condition1  +   
                     BldgType      + RoofStyle      + MasVnrType  +  ExterQual  +   ExterCond    +
                     Foundation   + BsmtQual    +  BsmtCond     + BsmtExposure +  BsmtFinType1 + HeatingQC  +   CentralAir    +
                     KitchenQual  + GarageType  +  GarageFinish   +  PavedDrive   + SaleType    +  SaleCondition + Alley        +
                     FireplaceQu  + Fence       +  MSSubClass   + LotFrontage +  LotArea    +   OverallQual +  OverallCond  + YearBuilt    +
                     YearRemodAdd + MasVnrArea  +  BsmtFinSF1   + BsmtFinSF2  +  BsmtUnfSF  +   TotalBsmtSF +  X1stFlrSF  +   X2ndFlrSF    +
                     GrLivArea    + BsmtFullBath + BsmtHalfBath + FullBath    +  HalfBath   +   BedroomAbvGr +  KitchenAbvGr  + TotRmsAbvGrd +
                     Fireplaces   + GarageYrBlt +  GarageCars   + GarageArea  +  WoodDeckSF +   OpenPorchSF +  EnclosedPorch + ScreenPorch  +
                     MoSold       + YrSold   ,nntrain))
head(nntrain.format)
nntrain.format.final <- cbind(SalePrice1=nntrain$SalePrice1,nntrain.format[,-c(1)])
head(nntrain.format.final)
n <- names(nntrain.format.final)
f <- as.formula(paste("SalePrice1 ~", paste(n[!n %in% "SalePrice1"], collapse = " + ")))
nnModel <- neuralnet(f,data=nntrain.format.final,hidden=c(2),linear.output=FALSE,stepmax=100000)
plot(nnModel)
nnPred <- compute(nnModel,nntrain.format.final[,-c(1)])
nnPred.Orig <- nnPred$net.result*(max(nntrain$SalePrice)-min(nntrain$SalePrice))+min(nntrain$SalePrice)
head(nnPred.Orig)

cor(datarpart.missForest[,1],nnPred.Orig)^2
#2 HN 0.9332394304
#5 HN 0.9816666158
#10 HN 0.9878189852
#15 HN 0.995258419

rmse(datarpart.missForest[,1],as.numeric(nnPred.Orig))
#2 HN 20520.59925
#5 HN 10770.62165
#10 HN 8784.457097
#15 HN 5472.449522

rmsle(datarpart.missForest[,1],as.numeric(nnPred.Orig))

nntest$SalePrice <- 0
nntest.format <- as.data.frame(model.matrix( SalePrice ~    MSZoning  +    LotShape +     LandContour +  LotConfig  +   Neighborhood +  Condition1  +   
                                                BldgType      + RoofStyle      + MasVnrType  +  ExterQual  +   ExterCond    +
                                                Foundation   + BsmtQual    +  BsmtCond     + BsmtExposure +  BsmtFinType1 + HeatingQC  +   CentralAir    +
                                                KitchenQual  + GarageType  +  GarageFinish   +  PavedDrive   + SaleType    +  SaleCondition + Alley        +
                                                FireplaceQu  + Fence       +  MSSubClass   + LotFrontage +  LotArea    +   OverallQual +  OverallCond  + YearBuilt    +
                                                YearRemodAdd + MasVnrArea  +  BsmtFinSF1   + BsmtFinSF2  +  BsmtUnfSF  +   TotalBsmtSF +  X1stFlrSF  +   X2ndFlrSF    +
                                                GrLivArea    + BsmtFullBath + BsmtHalfBath + FullBath    +  HalfBath   +   BedroomAbvGr +  KitchenAbvGr  + TotRmsAbvGrd +
                                                Fireplaces   + GarageYrBlt +  GarageCars   + GarageArea  +  WoodDeckSF +   OpenPorchSF +  EnclosedPorch + ScreenPorch  +
                                                MoSold       + YrSold   ,nntest))

nnPred.test <- compute(nnModel,nntest.format[,-c(1)])
nnPred.test.Orig <- nnPred.test$net.result*(max(nntrain$SalePrice)-min(nntrain$SalePrice))+min(nntrain$SalePrice)
head(nnPred.test.Orig)
length(nnPred.test.Orig)
dim(test)
submission.nn <- as.data.frame(cbind(Id=datarpart.missForest.test$Id,SalePrice=nnPred.test.Orig))
head(submission.nn)
write.csv(submission.nn,"D:/HP Backup 02Oct2017/177565(35gb)/UChicago/Data Mining Principles/Project/Submissions/submission.nn.csv",row.names = F)

#**********************************************************************************************************
#**********************************************************************************************************
#**********************************************************************************************************

#Ensemble

#rmse - rf - 13140.53
#rmse - gbm - 7258.247
#rmse - nn - 10770.62165


wtgbm <- 1/7258.247

wtrf <- 1/13140.53

wtnn <- 1/20520.59925


submission.ensemble <- as.data.frame(cbind(Id=datarpart.missForest.test$Id,
                                      SalePrice=(wtgbm*predgbm.test + wtrf*predrf.test + wtnn*nnPred.test.Orig)/(wtgbm + wtrf + wtnn)))
colnames(submission.ensemble) <- c('Id','SalePrice')
head(submission.ensemble)
write.csv(submission.ensemble,"D:/HP Backup 02Oct2017/177565(35gb)/UChicago/Data Mining Principles/Project/Submissions/submission.ensemble.csv",row.names = F)

