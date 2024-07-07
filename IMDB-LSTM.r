# Large Movie Review Dataset (IMDB)
# # Example of LSTM


# remotes::install_github('rstudio/tensorflow', force = TRUE)
# reticulate::install_python()
library(tensorflow)
# install_tensorflow(envname = "r-tensorflow")
# install.packages("keras3")
library(keras3) # For R Version 4.4.0 and 4.4.1
# install.packages("tidyverse")
library(tidyverse)


# Process the data
max_words <- 10000 
max_len <- 100 

# Load the IMDB data 
imdb <- dataset_imdb(num_words = max_words) 

# Split the data into training and test sets 
x_train <- imdb$train$x 
y_train <- imdb$train$y 
x_test <- imdb$test$x 
y_test <- imdb$test$y 

# Pad the sequences to have a fixed length 
x_train <- pad_sequences(x_train, maxlen = max_len) 
x_test <- pad_sequences(x_test, maxlen = max_len)

# Define the model
model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_words, 
                  output_dim = 32) %>% 
  layer_lstm(units = 32) %>% 
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model
model %>% compile( 
  optimizer = "adam", 
  loss = "binary_crossentropy", 
  metrics = c("accuracy") 
)

# Train the model
history <- model %>% fit( 
  x_train, y_train, 
  epochs = 10, 
  batch_size = 32, 
  validation_split = 0.2 
)

# Note that we are using a validation split of 0.2, 
# which means that 20% of the training data will be used as a validation set 
# to evaluate the model’s performance during training.
# Finally, we can use the trained model to evaluate its performance on the test set:
scores <- model %>% evaluate(x_test, y_test, 
                             verbose = 0) 
print(paste("Test accuracy:", scores[[2]]))
