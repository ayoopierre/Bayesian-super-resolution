# Whitening transformation
The core idea of this transformation is to take a vector of random variables with known covariance matrix and perform linear transformation into new set of variables whose covariance matrix is a identity - meaning that we change them into uncorrelated set of variables. It called like that since it transforms it into a white noise vector.

This idea is useful when working with correlated data, since structure of the covariance matrix may be really complicated to work with numerically. Since we perform a linear bijective transformation we can easily work with more well behaved numerically data and after transformations go back to original form.

**Note** - this linear transformation does not change the probabilistic model but modifies coordinates we represent it with. We only modify the random variable but the underlying random properties are preserved.

The issues with keeping the covariance matrix are:
1. We need $O(n^2)$ memory to keep it and operations on it are relatively expensive as well
2. The structure where some directions have much larger variances then others may induce worse numerical stability

Suppose we have a vector of RVs:
$$
X \sim \mathcal N (\mu, K)
$$
We want such transformation that:
$$
\operatorname{Cov}(Z) = I_d
$$
For instance:
$$
\mathbb E[X] = \mu
$$
$$
Z = K^{-1/2}(X - \mu)
$$
$$
\operatorname{Cov}(X) = \mathbb E[(X - \mathbb E[X])(X - \mathbb E[X])^T]
$$
$$
\begin{align*}
\operatorname{Cov}(Z) &= \mathbb E[(Z - \mathbb E[Z])(Z - \mathbb E[Z])^T]
\\ &\stackrel{\mathbb E[Z] = 0}{=} \mathbb E[ZZ^T]
\\ &= \mathbb E[K^{-1/2}(X - \mu)(X - \mu)^T(K^{-1/2})^T]
\\ &= K^{-1/2}\mathbb E[(X - \mu)(X-\mu)^T](K^{-1/2})^T
\\ &= K^{-1/2}K(K^{-1/2})^T
\\ &\stackrel{K = K^T}{=} K^{-1/2}K^{1/2}(K^{1/2})^T(K^{-1/2})^T
\\ &= I_d
\end{align*}
$$

This is especially useful if initial vector is multivariate Gaussian since in this case zero covariance also is independence.

Now for the iverse transform we only need to:
$$
X = \mu + K^{1/2}Z
$$
Now suppose we have a $X \sim \mathcal(\mu, K)$:
$$
P(X|Y) \propto P(Y|X)P(X)
$$
$$
\begin{align*}
P(Z|Y) = P(Y|\mu + K^{1/2}Z)P(Z) && Z\sim\mathcal N(0, I_d)
\end{align*}
$$

**Note** - This is very useful however one part that still remains is finding $K^{1/2}$ and its inverse.

## Why does the covariance change?
One thing which may be unclear is that why, even though we did not change anything in the underlying randomness, the covariance changed. The reason that happens is that **covariance is not property of the underlying random event but the random variable**. The correlation and covariance at the end of the day requires definition of random variable, and only this definition exposes what we intuitvely understand as for instance "corelated".
