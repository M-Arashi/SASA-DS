# Keras is a high-level neural networks API, running on top of TensorFlow, CNTK, 
# or Theano. You need to install it first. 
#You can do this within your R session using reticulate:


# remotes::install_github('rstudio/tensorflow', force = TRUE)
# reticulate::install_python()
library(tensorflow)
# install_tensorflow(envname = "r-tensorflow")
# install.packages("keras3")
library(keras3) # For R Version 4.4.0 and 4.4.1
# install.packages("tidyverse")
library(tidyverse)

mnist <- dataset_mnist()


x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# The images are encoded as as 3D arrays, and the labels are a 1D array 
# of digits, ranging from 0 to 9. 
# There is a one-to-one correspondence between the images and the labels.
train_images<-x_train;        str(train_images)
train_labels<-y_train;        str(train_labels)
test_images<-x_test;          str(test_images)
test_labels<-y_test;          str(test_labels)
dim(train_images)
typeof(train_images)


# So what we have here is an array of 60,000 matrices of 28 * 28 integers. 
# Each such matrix is a grayscale image, with coefficients between 0 and 255. 
# Let us plot the 10th digit in this 3D tensor:
digit <- train_images[10,,]
plot(as.raster(digit, max = 255))

#---------------------------------------------------------------------------
# Preparing the data
#---------------------------------------------------------------------------
# The x data is a 3D array (images,width,height) of grayscale values. 
# To prepare the data for training we convert the 3D arrays into matrices 
# by reshaping width and height into a single dimension 
# (28x28 images are flattened into length 784 vectors). 
# Then, we convert the grayscale values from integers ranging between 0 to 255 
# into floating point values ranging between 0 and 1:

# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

# The y data is an integer vector with values ranging from 0 to 9. 
#To prepare this data for training we encode the vectors into binary class 
# matrices using the Keras to_categorical() function:

# Convert class vectors to binary class matrices


y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)
# Example
to_categorical(c(5,0,4,1,9,2),10)


#---------------------------------------------------------------------------
# The network architecture
#---------------------------------------------------------------------------
# The core data structure of Keras is a model, a way to organize layers. 
# We will feed the neural network the training data. 
# The network will then learn to associate images and labels. 
# Finally, we will ask the network to produce predictions for test_images, 
# and we will verify whether these predictions match the labels from test_labels.

# The simplest type of model is the Sequential model, a linear stack of layers.
# We begin by creating a sequential model and then adding layers using 
# the pipe (%>%) operator: Actually you can translate 
# the pipe (\%>\%) as "then" 
# (e.g. start with a model, then add a layer, then add another layer, etc.)

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')
# The input_shape argument to the first layer specifies the shape of the input 
# data (a length 784 numeric vector representing a grayscale image). 
# The final layer outputs a length 10 numeric vector 
# (probabilities for each digit) using a softmax activation function.

# Use the summary() function to print the details of the model:
summary(model)


# Number of parameters in the model:
# (784+1)*256 + (256+1)*128 + (128+1)*10 =
#    200,960  +    32,896   +   1,290    =    235,146


# Step 3
# Next, compile the model with appropriate 
# loss function, optimizer, and metrics: 
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)


#---------------------------------------------------------------------------
# Training and Evaluation
#---------------------------------------------------------------------------
# Use the fit() function to train the model for 30 epochs using 
# batches of 128 images:
history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)
# The history object returned by fit() includes loss and accuracy metrics 
# which we can plot:
plot(history)

# Evaluate the model's performance on the test data:
model %>% evaluate(x_test, y_test)

# Generate predictions on new data:
model %>% predict(x_test)

