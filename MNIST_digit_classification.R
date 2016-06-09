library(nnet)
library(ggplot2)
#https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r
#library(caret)
#================================#
#Read data
#================================#
##Read data
setwd("/")
train <- read.table("train.csv", sep = ",", header = TRUE)
test <- read.table("test.csv", sep = ",", header = TRUE)

folds <- createFolds(as.factor(train$label), k = 5)

train_px <- as.matrix(train[, -1]) / 255
test_px <- as.matrix(test) / 255


##Column/vector to pixel matrix
to_matrix <- function(x) {
  matrix(as.numeric(x), nrow = 28, byrow = TRUE)
}
##Pixel matrix to vector
to_vector <- function(x) {
  as.numeric(t(x))
}
##Display digit
display_digit <- function(x) {
  img <- expand.grid(y = nrow(x):1, x = 1:ncol(x))
  img$intensity <- as.numeric(x)
  ggp <- ggplot(img, aes(x = x, y = y, fill = intensity)) + geom_raster()
  ggp <- ggp + 
    theme(line = element_blank(),
          text = element_blank(),
          line = element_blank(),
          title = element_blank(), 
          legend.position = "None", 
          panel.background = element_rect(fill = "transparent"), 
          plot.background = element_rect(fill = "transparent"), 
          strip.background = element_rect(fill = "transparent")) + 
    scale_fill_distiller(palette = 6, direction = 1)
  return(ggp)
  #image(t(x)[, ncol(t(x)):1], col = grey.colors(255))
}

##Pixel shifting
random_shift <- function(x, ...) {
  px <- x
  whiterow <- apply(px, 1, function(row) prod(row ==  0))
  px <- px[!whiterow, ]
  whitecol <- apply(px, 2, function(col) prod(col ==  0))
  px <- px[, !whitecol]
  px <- cbind(0, rbind(0, px, 0), 0)
  
  if (length(px) < length(x)) {
    px_shifted <- matrix(0, nrow = nrow(x), ncol = ncol(x))
    x0 <- sample(1:(nrow(x) - nrow(px)), 1)
    y0 <- sample(1:(ncol(x) - ncol(px)), 1)
    px_shifted[x0:(x0 + nrow(px) - 1), y0:(y0 + ncol(px) - 1)] <- px
  } else {
    return(x)
  }
  return(px_shifted)
}


to_matrix(train[1990, -1])
display_digit(to_matrix(train[1990, -1]))
display_digit(random_shift(to_matrix(train[1990, -1])))

##Augment data
augment <- function(df, multiflier = 2, keep_old = TRUE) {
  mtx <- as.matrix(df[, -1])
  aug_df <- lapply(1:(multiflier - keep_old), function(i) {
    new_df <- apply(mtx, 1, function(row) {
      to_vector(random_shift(to_matrix(row)))
    })
    new_df <- t(new_df)
    new_df <- cbind(df$label, as.data.frame(new_df))
    names(new_df) <- names(df)
    new_df
  })
  if (keep_old) {
    aug_df <- c(list(df), aug_df)
  }
  do.call("rbind", aug_df)
}


train_aug4 <- augment(train, 4)

library(grid)
library(gridExtra)
grid.arrange(
  display_digit(to_matrix(train[1, -1])),
  display_digit(to_matrix(train[2, -1])),
  display_digit(to_matrix(train[3, -1])),
  display_digit(to_matrix(train[4, -1])),
  display_digit(to_matrix(train[5, -1])),
  display_digit(to_matrix(train[6, -1])),
  display_digit(to_matrix(train[7, -1])),
  display_digit(to_matrix(train[8, -1])),
  display_digit(to_matrix(train[9, -1])),
  display_digit(to_matrix(train[10, -1])),
  display_digit(to_matrix(train[11, -1])),
  display_digit(to_matrix(train[12, -1])),
  display_digit(to_matrix(train[13, -1])),
  display_digit(to_matrix(train[14, -1])),
  display_digit(to_matrix(train[15, -1])),
  display_digit(to_matrix(train[16, -1]))
)


#================================#
#Compress to 14*14 (using neighbour) then NN ****Still too large****
#================================#
# compress <- function(img, w = 14, h = 14) {
#   w0 <- nrow(img)
#   h0 <- ncol(img)
#   w_ratio <- w0 / w
#   h_ratio <- h0 / h
#   compressed <- matrix(NA, nrow = w, ncol = h)
#   for (i in 1:w) {
#     for (j in 1:h) {
#       compressed[i, j] <- mean(img[(1:w0) %in% (floor(i * w_ratio):ceiling((i + 1) * w_ratio)), (1:h0) %in% (floor(j * h_ratio):ceiling((j + 1) * h_ratio))])
#     }
#   }
#   return(compressed)
# }
# ##Resize from 28*28 to 14*14
# train_px <- as.matrix(train[, -1]) / 255
# test_px <- as.matrix(test) / 255
# train_r <- cbind(label = train$label, as.data.frame(t(apply(train_px, 1, function(row) {compress(matrix(row, nrow = 28, ncol = 28))}))))
# test_r <- as.data.frame(t(apply(test_px, 1, function(row) {compress(matrix(row, nrow = 28, ncol = 28))})))
# 
# ##Fitting
# set.seed(1234)
# nn20_r <- nnet(as.factor(label) ~ ., data = train_r, size = 20, MaxNWts = 100000, maxit = 20000)
# save(nn20_r, file = "nn20.Rdata")
# 
# ##Prediction
# table(True = train_r$label, Predict = predict(nn20_r, type = "class"))
# prop.table(table(predict(nn20_r, type = "class") ==  as.character(train_r$label)))
# 
# ##Testing data
# test_nn20_r <- data.frame(ImageId = 1:28000,	Label = as.numeric(predict(nn20_r, newdata = test_r, type = "class")))
# write.csv(test_nn20_r, "sample_submission.csv", row.names = FALSE)








#================================#
#PCA then NN
#================================#
train_px <- as.matrix(train[, -1]) / 255
S <- cov(train_px)
pca <- prcomp(S)

var_explained <- pca$sdev^2 / sum(pca$sdev^2)
plot(cumsum(var_explained)[1:50], type = "l")
abline(h = 0.99, col = "red")

min(which(cumsum(var_explained) > =  0.99))
loadings <- pca$x[, 1:min(which(cumsum(var_explained) > =  0.99))]
train_44pc <- cbind(label = train$label, as.data.frame(train_px %*% loadings))

##Fit a Neural Network, size = 100
set.seed(100)
nn100_44pc <- nnet(as.factor(label) ~ ., data = train_44pc, size = 100, MaxNWts = 1E6, maxit = 3000)
save(train_44pc, test_44pc, nn100_44pc, file = "nn100_44pc.Rdata")

##Prediction
table(True = train_44pc$label, Predict = predict(nn100_44pc, newdata = train_44pc[, -1], type = "class"))
prop.table(table(as.character(train$label) ==  predict(nn100_44pc, newdata = train_44pc[, -1], type = "class")))

##Testing data
test_px <- as.matrix(test) / 255
test_44pc <- as.data.frame(test_px %*% loadings)
test_100_44pc <- data.frame(ImageId = 1:28000,	Label = as.numeric(predict(nn100_44pc, newdata = test_44pc, type = "class")))
write.csv(test_100_44pc, "sample_submission.csv", row.names = FALSE)




#================================#
#Data augmentation then PCA then NN 150
#================================#
train_px <- as.matrix(train_aug4[, -1]) / 255
S <- cov(train_px)
pca <- prcomp(S)

var_explained <- pca$sdev^2 / sum(pca$sdev^2)
plot(cumsum(var_explained)[1:50], type = "l")
abline(h = 0.99, col = "red")

min(which(cumsum(var_explained) > =  0.99))
loadings <- pca$x[, 1:min(which(cumsum(var_explained) > =  0.99))]
train_pc_aug4 <- cbind(label = train$label, as.data.frame(train_px %*% loadings))

##Fit a Neural Network, size = 150
set.seed(150)
nn150_pc_aug4 <- nnet(as.factor(label) ~ ., data = train_pc_aug4, size = 150, MaxNWts = 1E6, maxit = 4500)
save(train_pc_aug4, nn150_pc_aug4, file = "nn150_pc_aug4.Rdata")

##Performance in training set
table(True = train_pc_aug4$label, Predict = predict(nn150_pc_aug4, newdata = train_pc_aug4[, -1], type = "class"))
prop.table(table(as.character(train$label) ==  predict(nn150_pc_aug4, newdata = train_pc_aug4[, -1], type = "class")))

##Performance in validation set (shifted images)
validate <- augment(train, multiflier = 1, keep_old = FALSE)
validate_px <- as.matrix(validate[, -1]) / 255
validateset <- cbind(label = validate$label, as.data.frame(validate_px %*% loadings))
table(True = validset$label, Predict = predict(nn150_pc_aug4, newdata = validateset[, -1], type = "class"))
prop.table(table(as.character(validateset$label) ==  predict(nn150_pc_aug4, newdata = validateset[, -1], type = "class")))

##Submission of testing data
test_px <- as.matrix(test) / 255
testset <- as.data.frame(test_px %*% loadings)
test_submit <- data.frame(ImageId = 1:28000,	Label = as.numeric(predict(nn150_pc_aug4, newdata = testset, type = "class")))
write.csv(test_submit, "sample_submission.csv", row.names = FALSE)




#================================#
#Data augmentation then PCA then NN 200
#================================#
train_aug4 <- augment(train, 4)
train_px <- as.matrix(train_aug4[, -1]) / 255
S <- cov(train_px)
pca <- prcomp(S)

var_explained <- pca$sdev^2 / sum(pca$sdev^2)
plot(cumsum(var_explained)[1:50], type = "l")
abline(h = 0.99, col = "red")

min(which(cumsum(var_explained) > =  0.99))
loadings <- pca$x[, 1:min(which(cumsum(var_explained) > =  0.99))]
train_pc_aug4 <- cbind(label = train$label, as.data.frame(train_px %*% loadings))

##Fit a Neural Network, size = 200
set.seed(200)
nn200_pc_aug4 <- nnet(as.factor(label) ~ ., data = train_pc_aug4, size = 200, MaxNWts = 1E6, maxit = 6000)
save(train_pc_aug4, nn200_pc_aug4, file = "nn200_pc_aug4.Rdata")

##Performance in training set
table(True = train_pc_aug4$label, Predict = predict(nn200_pc_aug4, newdata = train_pc_aug4[, -1], type = "class"))
prop.table(table(as.character(train$label) ==  predict(nn200_pc_aug4, newdata = train_pc_aug4[, -1], type = "class")))

##Performance in validation set (shifted images)
validate <- augment(train, multiflier = 1, keep_old = FALSE)
validate_px <- as.matrix(validate[, -1]) / 255
validateset <- cbind(label = validate$label, as.data.frame(validate_px %*% loadings))
table(True = validset$label, Predict = predict(nn200_pc_aug4, newdata = validateset[, -1], type = "class"))
prop.table(table(as.character(validateset$label) ==  predict(nn200_pc_aug4, newdata = validateset[, -1], type = "class")))

##Submission of testing data
test_px <- as.matrix(test) / 255
testset <- as.data.frame(test_px %*% loadings)
test_submit <- data.frame(ImageId = 1:28000,	Label = as.numeric(predict(nn200_pc_aug4, newdata = testset, type = "class")))
write.csv(test_submit, "sample_submission.csv", row.names = FALSE)










