
# Install all required packages.
install.packages(c("ggplot2", "e1071", "caret", "quanteda", 
                   "irlba", "randomForest"))

library(ggplot2)
library(e1071)
library(caret)
library(quanteda)
library(irlba)
# Load up the .CSV data and explore in RStudio.
data.raw <- read.csv("airline.csv", stringsAsFactors = FALSE)
View(data.raw)



# Renaming the columns
data.raw <- data.raw[, 1:2]
names(data.raw) <- c( "Label","Text")
View(data.raw)


#Check data to see if there are missing values.
length(which(!complete.cases(data.raw)))


# Convert our class label into a factor.
data.raw$Label <- as.factor(data.raw$Label)



# The first step, as always, is to explore the data.
# First, let's take a look at distibution of the class labels (i.e., ham vs. spam).
prop.table(table(data.raw$Label))



# Next up, let's get a feel for the distribution of text lengths of the SMS 
# messages by adding a new feature for the length of each message.
data.raw$TextLength <- nchar(data.raw$Text)
summary(data.raw$TextLength)

# Use caret to create a 70%/30% stratified split. Set the random
# seed for reproducibility.
set.seed(32984)
indexes <- createDataPartition(data.raw$Label, times = 1,
                               p = 0.7, list = FALSE)

train <- data.raw[indexes,]
test <- data.raw[-indexes,]


# Verify proportions.
prop.table(table(train$Label))
prop.table(table(test$Label))


#Unigram

# Tokenize SMS text messages.
train.tokens <- tokens(train$Text, what = "word", 
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, remove_hyphens = TRUE)

# Using quanteda's built-in stopword list for English.

train.tokens <- tokens_select(train.tokens, stopwords(), 
                              selection = "remove")
train.tokens[[20]]


# Perform stemming on the tokens.
train.tokens <- tokens_wordstem(train.tokens, language = "english")
train.tokens[[20]]


# Creating our first bag-of-words model.

train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)


# Transform to a matrix and inspecting.
train.tokens.matrix <- as.matrix(train.tokens.dfm)
View(train.tokens.matrix[1:20, 1:100])
dim(train.tokens.matrix)


# Investigate the effects of stemming.
colnames(train.tokens.matrix)[1:50]

# Setup a the feature data frame with labels.
train.tokens.df <- cbind(Label = train$Label, data.frame(train.tokens.dfm))
View(train.tokens.df)
dim(train.tokens.df)


# Cleanup column names.
names(train.tokens.df) <- make.names(names(train.tokens.df))

# Add bigrams to our feature matrix.
# train.tokens <- tokens_ngrams(train.tokens, n = 1:2)

# Our function for calculating relative term frequency (TF)
term.frequency <- function(row) {
  row / sum(row)
}

# Our function for calculating inverse document frequency (IDF)
inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  
  log10(corpus.size / doc.count)
}

# Our function for calculating TF-IDF.
tf.idf <- function(x, idf) {
  x * idf
}
View(train.tokens.df)

# First step, normalize all documents via TF.
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)
dim(train.tokens.df)
View(train.tokens.df[1:20, 1:100])


# Second step, calculate the IDF vector that we will use - both
# for training data and for test data!
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)
str(train.tokens.idf)


# Lastly, calculate TF-IDF for our training corpus.
train.tokens.tfidf <-  apply(train.tokens.df, 2, tf.idf, idf = train.tokens.idf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25, 1:25])


# Transpose the matrix
train.tokens.tfidf <- t(train.tokens.tfidf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf[1:25, 1:25])


# Check for incopmlete cases.
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
train$Text[incomplete.cases]


# Fix incomplete cases
train.tokens.tfidf[incomplete.cases,] <- rep(0.0, ncol(train.tokens.tfidf))
dim(train.tokens.tfidf)
sum(which(!complete.cases(train.tokens.tfidf)))


# Make a clean data frame using the same process as before.
train.tokens.tfidf.df <- cbind(Label = train$Label, data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))

library(irlba)

# Time the code execution
start.time <- Sys.time()

# Perform SVD. Specifically, reduce dimensionality down to different number of columns
# for our latent semantic analysis (LSA).

#Trying with different predictors
train.irlba <- irlba(t(train.tokens.tfidf), nv = 10, maxit = 600)
#train.irlba <- irlba(t(train.tokens.tfidf), nv = 20, maxit = 600)
#train.irlba <- irlba(t(train.tokens.tfidf), nv = 15, maxit = 600)



# Look at the first 10 components of projected document and the corresponding
# row in our document semantic space (i.e., the V matrix)

train.irlba$v[1, 1:10]

library(doSNOW)


train.svd <- data.frame(Label = train$Label, train.irlba$v)


# Create a cluster to work on 10 logical cores.
cl <- makeCluster(10, type = "SOCK")
registerDoSNOW(cl)

# Time the code execution
start.time <- Sys.time()
library(caret)

cv.folds <- createMultiFolds(train$Label, k = 10, times = 5)

cv.cntrl <- trainControl(method = "repeatedcv", number = 10,
                         repeats = 3, index = cv.folds)




rpart_uni.cv.4_10 <- train(Label ~ ., data = train.svd, method = "rpart", 
                           trControl = cv.cntrl, tuneLength = 7)

#rpart_uni.cv.4_20 <- train(Label ~ ., data = train.svd, method = "rpart", 
                          # trControl = cv.cntrl, tuneLength = 7)

#rpart_uni.cv.4_15 <- train(Label ~ ., data = train.svd, method = "rpart", 
                          # trControl = cv.cntrl, tuneLength = 7)
# Processing is done, stop cluster.
stopCluster(cl)

# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time

# Check out our results.

rpart_uni.cv.4_10
rpart_uni.cv.4_20
rpart_uni.cv.4_15

##########################Using Random Forest##########################

# Create a cluster to work on 10 logical cores.
cl <- makeCluster(10, type = "SOCK")
registerDoSNOW(cl)

# Time the code execution
start.time <- Sys.time()



rf_uni.cv.1_10 <- train(Label ~ ., data = train.svd, method = "rf", 
                        trControl = cv.cntrl, tuneLength = 4)

rf_uni.cv.1_20 <- train(Label ~ ., data = train.svd, method = "rf", 
                        trControl = cv.cntrl, tuneLength = 4)

rf_uni.cv.1_15 <- train(Label ~ ., data = train.svd, method = "rf", 
                        trControl = cv.cntrl, tuneLength = 4)
library(randomForest)
# Processing is done, stop cluster.
stopCluster(cl)

# Total time of execution on workstation was 
total.time <- Sys.time() - start.time
total.time




# Check out our results.
rf_uni.cv.1_10
grf_uni.cv.1_20
rf_uni.cv.1_15

# Let's drill-down on the results.
confusionMatrix(train.svd$Label, rf_uni.cv.10$finalModel$predicted)
confusionMatrix(train.svd$Label, rf_uni.cv.1_20$finalModel$predicted)
confusionMatrix(train.svd$Label, rf_uni.cv.1_15$finalModel$predicted)



#SVM
svm_1_uni <- train(Label ~ ., data = train.svd, method = 'svmLinear3',
                   trControl = cv.cntrl, tuneLength = 3,
                   importance = TRUE)
confusionMatrix(svm_1_uni)



##############################Test data
# Tokenization.
test.tokens <- tokens(test$Text, what = "word", 
                      remove_numbers = TRUE, remove_punct = TRUE,
                      remove_symbols = TRUE, remove_hyphens = TRUE)

# Lower case the tokens.
test.tokens <- tokens_tolower(test.tokens)

# Stopword removal.
test.tokens <- tokens_select(test.tokens, stopwords(), 
                             selection = "remove")

# Stemming.
test.tokens <- tokens_wordstem(test.tokens, language = "english")

# # Add bigrams.
#test.tokens <- tokens_ngrams(test.tokens, n = 1)

# Convert n-grams to quanteda document-term frequency matrix.
test.tokens.dfm <- dfm(test.tokens, tolower = FALSE)

# Explore the train and test quanteda dfm objects.
train.tokens.dfm
test.tokens.dfm

# Ensure the test dfm has the same n-grams as the training dfm.
#
# NOTE - In production we should expect that new text messages will 
#        contain n-grams that did not exist in the original training
#        data. As such, we need to strip those n-grams out.
#
test.tokens.dfm <- dfm_select(test.tokens.dfm, features = train.tokens.dfm)
test.tokens.matrix <- as.matrix(test.tokens.dfm)
test.tokens.dfm




# With the raw test features in place next up is the projecting the term
# counts for the unigrams into the same TF-IDF vector space as our training
# data. The high level process is as follows:
#      1 - Normalize each document (i.e, each row)
#      2 - Perform IDF multiplication using training IDF values

# Normalize all documents via TF.
test.tokens.df <- apply(test.tokens.matrix, 1, term.frequency)
str(test.tokens.df)

# Lastly, calculate TF-IDF for our training corpus.
test.tokens.tfidf <-  apply(test.tokens.df, 2, tf.idf, idf = train.tokens.idf)
dim(test.tokens.tfidf)
View(test.tokens.tfidf[1:25, 1:25])

# Transpose the matrix
test.tokens.tfidf <- t(test.tokens.tfidf)

# Fix incomplete cases
summary(test.tokens.tfidf[1,])
test.tokens.tfidf[is.na(test.tokens.tfidf)] <- 0.0
summary(test.tokens.tfidf[1,])




# With the test data projected into the TF-IDF vector space of the training
# data we can now to the final projection into the training LSA semantic
# space (i.e. the SVD matrix factorization).
test.svd.raw <- t(sigma.inverse * u.transpose %*% t(test.tokens.tfidf))
dim(test.svd.raw)



# Lastly, we can now build the test data frame to feed into our trained
# machine learning model for predictions. First up, add Label and TextLength.
test.svd <- data.frame(Label = test$Label, test.svd.raw)




# Now we can make predictions on the test data set using our trained mighty 
# Decision tREE

preds <- predict(rpart_uni.cv.4_10, test.svd)

preds <- predict(rpart_uni.cv.4_20, test.svd)
preds <- predict(rpart_uni.cv.4_15, test.svd)

# Drill-in on results
confusionMatrix(preds, test.svd$Label)

#rANDOM_FOREST

preds <- predict(rf_uni.cv.1_10, test.svd)
preds <- predict(rf_uni.cv.1_20, test.svd)

preds <- predict(rf_uni.cv.1_15, test.svd)

# Drill-in on results
confusionMatrix(preds, test.svd$Label)


pred_svm1_uni <- predict(svm_1_uni, test.svd)


# Drill-in on results
confusionMatrix(pred_svm1_uni, test.svd$Label)







