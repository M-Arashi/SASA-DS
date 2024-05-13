W1 <- matrix(c(0.15, 0.2, 0.25, 0.3), 2, byrow = T)
X <- c(0.05 , 0.1)
b1 <- c(0.35, 0.35)
net1<- W1 %*% X + b1
W2 <- matrix(c(0.4, 0.45, 0.55, 0.6), 2, byrow = T)
b2 <- c(0.6, 0.6)
f <- function(x) {
  y <- NULL
  for (i in 1:2) {
    y[i] <- solve(1+exp(-x[i]))  
  }
  return(y)
  }
out1 <- f(net1)
Z = out1
net2 <- W2 %*% Z + b2
out2 <- f(net2)
