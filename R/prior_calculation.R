library(flowClust)

# Define the prior function
prior <- function(x, kappa, Nt = NULL, addCluster = NULL) {
  if (is.null(Nt)) {
    Nt <- nrow(x@z)
  }
  p <- ncol(x@mu)
  K <- x@K
  nu0 <- Ng <- x@w * Nt
  if (all((nu0 * kappa - p - 1) > 0)) {
    Lambda0 <- x@sigma
    for (i in 1:K) {
      Lambda0[i, , ] <- Lambda0[i, , ] * (kappa * nu0[i] - p - 1)
    }
  } else {
    stop("Can't proceed. Prior nu0 is negative for cluster(s) ", 
         paste(which((nu0 - p - 1) > 0), collapse = ","), 
         "\n(p-1) = ", p - 1, ": Try increasing kappa")
  }
  Omega0 <- array(0, c(K, p, p))
  for (i in 1:K) {
    Omega0[i, , ] <- diag(1, p)
    if (p == 1) {
      dS <- x@sigma[i, , ]
      dO <- Omega0[i, , ]
    } else {
      dS <- det(x@sigma[i, , ])
      dO <- det(Omega0[i, , ])
    }
    k <- (dO/dS)^(1/p)
    Omega0[i, , ] <- Omega0[i, , ] * k
    Omega0[i, , ] <- solve(Omega0[i, , ] * Ng[i] * kappa)
  }
  nu0 <- nu0 * kappa
  Mu0 <- x@mu
  lambda <- x@lambda
  w0 <- x@w * Nt
  if (!is.null(addCluster)) {
    for (i in (K + 1):(K + addCluster)) {
      S <- stats::cov(Mu0)
      Lam <- array(0, c(K + 1, p, p))
      om <- array(0, c(K + 1, p, p))
      Mu0 <- rbind(Mu0, colMeans(Mu0))
      for (i in 1:K) {
        om[i, , ] <- Omega0[i, , ]
        Lam[i, , ] <- Lambda0[i, , ]
      }
      om[K + 1, , ] <- diag(1, p)
      Lam[K + 1, , ] <- diag(1, p)
      diag(Lam[K + 1, , ]) <- diag(S)
      diag(om[K + 1, , ]) <- diag(S)
      if (p == 1) {
        dS <- Lam[K + 1, , ]
        dO <- om[K + 1, , ]
      } else {
        dS <- det(Lam[K + 1, , ])
        dO <- det(om[K + 1, , ])
      }
      k <- (dO/dS)^(1/p)
      om[K + 1, , ] <- om[K + 1, , ] * k
      Omega0 <- om
      Lambda0 <- Lam
      nu0 <- c(nu0, p + 2)
      w0 <- c(w0, 1)
      K <- K + 1
    }
  }
  prior <- list(Mu0 = Mu0, Lambda0 = Lambda0, Omega0 = Omega0, 
                w0 = w0, nu0 = nu0, nu = x@nu, lambda = lambda, K = K)
  class(prior) <- "flowClustPrior"
  attr(prior, "lambda") <- lambda
  
  return(prior)
}
