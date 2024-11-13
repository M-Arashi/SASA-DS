# Step 1: Install and Load Required Packages

install.packages("igraph")
install.packages("torch")

library(igraph)
library(torch)

# Step 2: Create a Graph

# Create a graph with 5 nodes and some edges
g <- graph_from_literal(A--B, B--C, C--D, D--E, E--A, A--C)

# Plot the graph
plot(g)

# Step 3: Generate Random Node Features

# Generate random node features
node_features <- matrix(runif(5 * 3), nrow = 5, ncol = 3)
V(g)$features <- split(node_features, row(node_features))

#This creates a matrix of random numbers (between 0 and 1) and assigns them to the nodes as features.


# Step 4: Define the Weight Matrix
# The weight matrix 'W' is used to transform node features during 
# the GNN layer's message-passing process. 
# This matrix will be learned during training, and it is initialized randomly.
# Define the weight matrix W outside the function
W <- torch_tensor(matrix(runif(3 * 3), nrow = 3, ncol = 3), requires_grad = TRUE)
# Here, 'W' is a 3x3 matrix (assuming each node has 3 features) 
# initialized with random values and marked as a variable that requires 
# gradients (for backpropagation).

# Step 5: Define a Simple GNN Layer
# The GNN layer function performs message passing by multiplying 
# the adjacency matrix with the node features, followed by applying 
# the weight matrix and a non-linear activation function (ReLU).
# Simple GNN layer using message passing
gnn_layer <- function(node_features, adj_matrix, W) {
  X <- torch_tensor(node_features)
  A <- torch_tensor(adj_matrix, dtype = torch_float32())
  AX <- torch_mm(A, X)
  AXW <- torch_mm(AX, W)
  H <- torch_relu(AXW)
  return(H)
}

# Step 6: Compute the Adjacency Matrix and Apply the GNN Layer
# Get adjacency matrix
adj_matrix <- as_adjacency_matrix(g, sparse = FALSE)
adj_matrix 
# Apply GNN layer
output_features <- gnn_layer(node_features, adj_matrix, W)
print(output_features)
# This converts the graph into an adjacency matrix that shows which nodes 
# are connected.

# Step 7: Define a Simple Loss Function and Target


# Define the target 
target <- torch_tensor(matrix(runif(5 * 3), nrow = 5, ncol = 3)) 
# Same shape as output features  

# Define a simple loss function (mean squared error)
loss_fn <- function(pred, target) {
  torch_mean((pred - target)^2)
}

# Step 8: Training Loop
# Example training loop
for (epoch in 1:100) {
  pred <- gnn_layer(node_features, adj_matrix, W)
  loss <- loss_fn(pred, target)
  loss$backward()
  
  # Update weights (gradient descent step)
  with_no_grad({
    W$sub_(W$grad * 0.01)
    W$grad$zero_()
  })
  
  # Print loss every 10 epochs
  if (epoch %% 10 == 0) {
    cat("Epoch:", epoch, "Loss:", as.numeric(loss$item()), "\n")
  }
}






