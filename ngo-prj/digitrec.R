# Creates a simple random forest benchmark

library(randomForest)
library(readr)

trainData <- read_csv("/Users/owner/Documents/CMPT404/ngo-prj/Digits/train.csv")
# test <- read_csv("/Users/owner/Documents/CMPT404/ngo-prj/Digits/test.csv")

rFfun <- function(numTrain = 10000){
  numTrees <- 25
  train <- trainData

  labels <- as.factor(train$label)
  
  trainRows <- sample(1:nrow(train), numTrain)
  test <- train[-trainRows,]
  train <- train[trainRows,]
  trainLabels <- as.factor(train$label)
  train <- train[,-1]
  testLabels <- as.factor(test$label)
  test <- test[,-1]
  
  rf <- randomForest(train, trainLabels, xtest=test, ntree=numTrees)
  predictions <- data.frame(ImageId=1:nrow(test), Label=levels(labels)[rf$test$predicted])
  
  # write_csv(predictions, "rf_benchmark.csv") 
  return(mean(predictions$Label == testLabels))
}

accuracy <- matrix(nrow = 5, ncol=12)
accuracy <- as.data.frame(accuracy)
colnames(accuracy) <- seq(.25,.8,.05)

for (i in seq(.25,.8,.05)) {
  for (j in 1:5) {
    numTrain <- as.integer(nrow(trainData)*i)
    print(numTrain)
    accuracy[j,as.character(i)] <- rFfun(numTrain)
    print(accuracy[j,as.character(i)])
  }
}

mean(predictions$Label[which(testLabels == 9)] == 4)

accuracy[6,] <- apply(accuracy,2,mean)
rownames(accuracy)[6] <- "avg"

plot(seq(.25,.8,.05),accuracy["avg",])

accuracy2 <- unlist(accuracy[1:5,])
accuracy2 <- data.frame(cbind(sort(rep(seq(.25,.8,.05),5))*nrow(trainData),accuracy2))
colnames(accuracy2) <- c("numTrain","accuracy")

plot(accuracy2$numTrain,accuracy2$accuracy)
lo <- loess(accuracy2$accuracy ~ accuracy2$numTrain)
lines(accuracy2$numTrain,predict(lo), col='red', lwd=2)

rf <- randomForest(train, trainLabels, xtest=test, ntree=numTrees, importance = TRUE)
predictions <- data.frame(ImageId=1:nrow(test), Label=levels(labels)[rf$test$predicted])
mean(predictions$Label == testLabels)

View(rf$importance)

varImpPlot(rf,n.var = 100)
varImpPlot(rf,n.var = 25)

as.integer(ncol(train)*.25)

top25 <- names(sort(rf$importance[,"MeanDecreaseAccuracy"], decreasing = TRUE)[1:as.integer(ncol(train)*.25)])

train <- train[,sapply(top25, function(x) which(x == colnames(train)))]

test <- test[,sapply(top25, function(x) which(x == colnames(test)))]

rfTop25 <- randomForest(train, trainLabels, xtest=test, ntree=numTrees, importance = TRUE)
predictionsTop25 <- data.frame(ImageId=1:nrow(test), Label=levels(labels)[rfTop25$test$predicted])
mean(predictionsTop25$Label == testLabels)

strictlyPositive <- names(rf$importance[,"MeanDecreaseAccuracy"] > 0)

set.seed(0)
train <- trainData

labels <- as.factor(train$label)

trainRows <- sample(1:nrow(train), numTrain)
test <- train[-trainRows,]
train <- train[trainRows,]
trainLabels <- as.factor(train$label)
train <- train[,-1]
testLabels <- as.factor(test$label)
test <- test[,-1]

train <- train[,sapply(strictlyPositive, function(x) which(x == colnames(train)))]

test <- test[,sapply(strictlyPositive, function(x) which(x == colnames(test)))]

rfSP <- randomForest(train, trainLabels, xtest=test, ntree=numTrees, importance = TRUE)
predictionsSP <- data.frame(ImageId=1:nrow(test), Label=levels(labels)[rfSP$test$predicted])
mean(predictionsSP$Label == testLabels)

#
accuracy3 <- matrix(nrow = 10, ncol= 10)
accuracy3 <- as.data.frame(accuracy3)
rownames(accuracy3) <- as.character(0:9)
colnames(accuracy3) <- as.character(0:9)

for (i in 0:9) {
  for (j in 0:9) {
    accuracy3[i+1,j+1] <- mean(predictions$Label[which(testLabels == i)] == j)
  }
}

apply(accuracy3, 1, sum)
# Therefore each row represents the classification spread of each number

apply(accuracy3,1, function(x) sort(x,decreasing = TRUE)[2])

## nine and four
train <- train[which(trainLabels == 4|trainLabels == 9),]
test <- test[which(testLabels == 4|testLabels == 9),]
trainLabels <- droplevels(trainLabels[which(trainLabels == 4|trainLabels == 9)])
testLabels <- droplevels(testLabels[which(testLabels == 4|testLabels == 9)])

rf.4.9 <- randomForest(train, trainLabels, xtest=test, ntree=numTrees, importance = TRUE)
predictions.4.9 <- data.frame(ImageId=1:nrow(test), Label=levels(trainLabels)[rf.4.9$test$predicted])
mean(predictions.4.9$Label == testLabels)

View(rf.4.9$importance)
set.seed(0)

