[MAP vs Bayes SR](https://proceedings.neurips.cc/paper_files/paper/2002/file/88bfcf02e7f554f9e9ea350b699bc6a7-Paper.pdf)
# Forward/generative model
**Initial assumption** -  we assume that there exists a ground-truth, high resolution image $\mathbf x$, which we can observe via image sensor, which has downsampling operator $\mathbf D$, blur $\mathbf H$ operator (related to any blur related to optical equipment) - both of those operators are shared between each photo taken - and lastly some kind off warp and rotation between each row-res photo $\mathbf W(\mathbf \theta_k)$. Additionally we apply some kind off noise $\mathbf n$ - normally assumed to be a Gaussian of form $\mathcal N(\mathbf 0, \sigma^2 \mathbf I)$. Therfore each photo $\mathbf y^{(k)}$ can be modeled as:
$$
\mathbf y^{(k)} = \mathbf D \mathbf H \mathbf W(\theta_k)\mathbf x+\mathbf n
$$
Given the noise model we know that (since the downsampling, blur and shift with rotation are all deterministic - however the blur might come from random process like motion blur comming from virbations - the only random part is noise, which is Gaussian):
$$
p(\mathbf y^{(k)}|\mathbf x, \theta_k, \sigma^2) \sim
\mathcal N(\mathbf y^{(k)};\mathbf D \mathbf H\mathbf W(\theta_k) \mathbf x, \sigma^2\mathbf I)
$$
*Note* - the notation $\mathcal N(x; \mu, \sigma)$ means that $x$ is distributed from $\mathcal N(\mu, \sigma)$. This is basically a value of probability density function for given point. For instance in multivariate case
$$
\mathcal N(\mathbf x, \mu, \Sigma)=\frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\left(
-\frac{1}{2}(\mathbf x-\mu)^T\Sigma^{-1}(\mathbf x - \mu)
\right)
$$

Now assuming independence of each frame:
$$
p(\mathbf Y|\mathbf x, \boldsymbol \theta, \sigma^2)=
\prod_{i=1}^K p(\mathbf y^{(k)}|\mathbf x, \theta_k,\sigma^2)
$$
**Posterior vs. data-driven**
At this point we have to posibilities to formulte complete equation. We can formulate a prior, which models the image but it is hard since we need to select such model that approprietlly relates to ground-truth high resolution image. On the other hand we can estimate the high resolution image, using kernel based method, yielding high resolution multi-variate Gaussian estimation of the image. This multi-variate Gaussian estimation leads us to a Gaussian process.

# Data driven Gaussian process
To sum up we take the sample data, and upscale it using for instance bilinear interpolation (the issue that remains is the allignment - formulate with uknown $\theta_k$ and try to optmize?). This approximation acts a the multi-variate Gaussian mean, and we use covariance matrix encoding with "radius of influence":
$$
\mathbf K_{ij} = \eta^2\exp(-\frac{|| \mathbf r_i \ -\mathbf r_j||^2}{2\ell^2})
$$
Where $\ell$ is some kind off influence distance, and $\eta$ is marginal prior variance. This basically yields a'priori $\sim \mathcal N(\boldsymbol\mu_0, \mathbf K)$, where $\boldsymbol\mu_0$ is computed via this bilinear approach. So using the Bayes rule, holding $\boldsymbol\theta$ and $\sigma^2$:
$$
p(\mathbf x|\mathbf Y,\boldsymbol\theta,\sigma^2) \propto 
p(\mathbf Y|\mathbf x,\boldsymbol\theta,\sigma^2)p(\mathbf x)
$$
Nowing this we can formulate actual ditribution of the high resolution image (let $A_k=DHW(\theta_k)$)
$$
\boldsymbol\Sigma^{-1}_x=\frac{1}{\sigma^2}\sum_kA_kA_k^T+\boldsymbol K^{-1}
$$
$$
\boldsymbol\mu_x=\boldsymbol\Sigma_x\left (
\frac{1}{\sigma^2}\sum_k \mathbf A_k^T\mathbf y^{(k)}+\mathbf K^{-1}\boldsymbol\mu_0
\right)
$$
This can be precisely derived by posing the problem as Gaussian process. See the previous equation:
$$
p(\mathbf x|\mathbf Y,\boldsymbol\theta,\sigma^2) \propto 
p(\mathbf Y|\mathbf x,\boldsymbol\theta,\sigma^2)p(\mathbf x)
$$
We basically need to find such $\hat{\mathbf x}$ which maximizes the probability given data. However the $\boldsymbol \theta$ and $\sigma$ remain unknown (they will be estimated elsewhere and assumed as constant and non-random here). We assume $\mathbf x \sim \mathcal N(\boldsymbol\mu_0, \mathbf K)$, where Gaussian parameters compute as stated above. Know we know from the initial image model (the first equation in this note):
$$
\mathbf y^{(k)} | \mathbf x\sim \mathcal N \left(
	\mathbf A_k(\theta_k)\mathbf x,
	\sigma^2 \mathbf I
\right)
$$
Stacking everything together, using the assumption of frame independance:
$$
\mathbf Y|\mathbf x \sim \mathcal N \left (
\mathbf A(\boldsymbol \theta) \mathbf x,
\sigma^2\mathbf I
\right)
$$
Therfore:
$$
p(\mathbf x|\mathbf Y) = p(\mathbf Y|\mathbf x)p(\mathbf x)
$$
$$
\log p(\mathbf x|\mathbf Y) = \log p(\mathbf Y | \mathbf x) + \log p(\mathbf x) 
$$
Since we know that both are Gaussians we essentially keep the exponating terms a and some constant:
$$
\log p(\mathbf x| \mathbf Y) =
-\frac{1}{2\sigma^2}(\mathbf Y - \mathbf A\mathbf x)^T(\mathbf Y - \mathbf A\mathbf x)
- \frac{1}{2}(\mathbf x - \boldsymbol\mu_0)^T\mathbf K^{-1}(\mathbf x - \boldsymbol \mu_0) + const
$$
**Note** - Here we take MLE like approach for $\boldsymbol \theta$ rather than modeling it as a random variable we assume this to be constant parameter. This allows us in return to model the conditional as a Gaussian. In the first part we have only variance propagated from noise factor from the model, and for the second part, which is prior estimate variance comes from kernel like estmation.

We now expand both multiplication terms:
$$
(\mathbf Y - \mathbf A\mathbf x)^T(\mathbf Y - \mathbf A\mathbf x) = 
\mathbf Y^T\mathbf Y
-\mathbf Y^T\mathbf A\mathbf x
-\mathbf x^T\mathbf A^T\mathbf Y
+\mathbf x^T\mathbf A^T\mathbf A\mathbf x
$$
One can notice however that product of this expression is a sclar. Therefore each part of the sum is a scalar, which means that $(\mathbf Y^T\mathbf A\mathbf x)^T = \mathbf x^T\mathbf A^T\mathbf Y$.
$$
(\mathbf Y - \mathbf A\mathbf x)^T(\mathbf Y - \mathbf A\mathbf x) = 
\mathbf Y^T\mathbf Y
-2\mathbf x^T\mathbf A^T\mathbf Y
+\mathbf x^T\mathbf A^T\mathbf A\mathbf x
$$
We do similar trick for the second term:
$$
(\mathbf x -\boldsymbol\mu_0)^T\mathbf K^{-1}(\mathbf x-\boldsymbol\mu_0) = 
\mathbf x^T\mathbf K^{-1}\mathbf x
-2\mathbf x^T\mathbf K^{-1}\boldsymbol \mu_0
+\boldsymbol \mu_0^T\mathbf K^{-1}
$$
So now we group for GP:
$$
\log(\mathbf x|\mathbf Y) =
-\frac{1}{2}\mathbf x \underbrace{
\left (
    \frac{1}{\sigma^2}\mathbf A^T\mathbf A+\mathbf K^{-1}
\right)
}_{\boldsymbol \Sigma_x^{-1}}\mathbf x^T 
+ \mathbf x^T \underbrace{
\left (
    \frac{1}{\sigma^2}\mathbf A^T\mathbf Y+\mathbf K^{-1}\boldsymbol \mu_0
\right)}_{\boldsymbol \Sigma_x^{-1}\boldsymbol \mu_x}
$$


Additionally lets unpack the stacking we did (the same algerba can be done with plain sum of log probability of non-stacked expressions per low res images - however this is abit more elegant). We could use pure sum as well:
$$
\log P(\mathbf x|(\mathbf y_1,...,\mathbf y_k);(\theta_1,...,\theta_k);\sigma^2) = 
\sum_{i = 1}^k \log P(\mathbf x|\mathbf y_i,\theta_k,\sigma^2)
$$

$$
\begin{align}
\mathbf A = \begin{bmatrix}
\mathbf A_1 \\
\mathbf A_2 \\
\vdots \\
\mathbf A_k \\
\end{bmatrix}
&&
\mathbf Y = \begin{bmatrix}
\mathbf y_1 \\
\mathbf y_2 \\
\vdots \\
\mathbf y_k \\
\end{bmatrix}
\end{align}
$$
$$
\begin{align}
\mathbf A^T\mathbf A = \sum_k \mathbf A_k^T\mathbf A_k 
&&
\mathbf A^T\mathbf Y = \sum_k\mathbf A_k^T\mathbf y_k
\end{align}
$$
And this yields original equations that were initially listed.
$$
\mathbf x | \mathbf Y \sim \mathcal N(\boldsymbol\mu_x, \boldsymbol \Sigma_x)
$$
And we have direct expressions we can use to compute both. To get to the $\boldsymbol \mu_x$ however we need to first compute $\boldsymbol \Sigma_x^{-1}$ invert is and multiply the $\boldsymbol\Sigma_x^{-1}\boldsymbol\mu_x$. The issue however is the fact that to do so we need to invert matrix of HR resolution by HR resolution which becoms intractable very quickly. This however still uses $\mathbf A$ in closed form therefore estimation of $\boldsymbol\theta$ and $\sigma^2$ is still required

**Bayesian approach for warp and noise variance estimation**
The easiest way to do this is to marginalize over all possible high-resolution generating image $\mathbf x$. Basically:
$$
p(\mathbf Y | \boldsymbol \theta, \sigma) = \int p(\mathbf Y|\mathbf x,\boldsymbol\theta, \sigma)p(\mathbf x)d\mathbf x
$$
This is essentially process of marginalization. This fact comes from below:
$$
p_X(x)=
\int p_{X|Y}(x|y)p_Y(y)dy=
\int\frac{p_{XY}(x,y)}{p_Y(y)}p_Y(y)dy=
\int p_{XY}(x, y)dy
$$


# $\boldsymbol\mu_x$ and $\boldsymbol\Sigma_x$ derivation from log probability
We already know that what we have there is a log probability of a Gaussian (comming from Gaussian noise assumption and the fact that we assume rest of the parametrs to be concrete constant parametrs - of course they are estimated by marginalizing $\mathbf x$  but this estimation makes the problem tractable). We only need to extract the mean and covariance matrix. Let:
$$
\mathbf P \stackrel{def}{=}\frac{1}{\sigma^2}\mathbf A^T\mathbf A + \mathbf K^{-1}
$$
$$
\mathbf b \stackrel{def}{=}\frac{1}{\sigma^2}\mathbf A^T\mathbf Y
+\mathbf K^{-1}\boldsymbol \mu_0
$$
than:
$$
\log(\mathbf x|\mathbf Y) = -\frac{1}{2}\mathbf x^T\mathbf P\mathbf x + 
\mathbf x^T\mathbf b+const.
$$
For a Gaussian log density looks like:
$$
\log\mathcal N(\mathbf x;\boldsymbol \mu, \boldsymbol\Sigma)=
-\frac{1}{2}(\mathbf x - \boldsymbol\mu)^T
\boldsymbol\Sigma^{-1}
(\mathbf x-\boldsymbol\mu) + const.
$$
So we basically need to translate that to a quadriatic form. Basically multiply stuff in brackets, keeping in mind that not only whole expression is a scalar, but also $\boldsymbol\Sigma^{-1}$ is a symmetric matrix (since it is the covariance matrix):
$$
-\frac{1}{2}\mathbf x^T\mathbf P\mathbf x + 
\mathbf x^T\mathbf b+const=
-\frac{1}{2}\mathbf x^T\boldsymbol \Sigma ^{-1}\mathbf x\mathbf + 
\mathbf x^T\boldsymbol\Sigma^{-1}\boldsymbol\mu
$$
From the equation we can easily conclude what $\boldsymbol\Sigma$ is given $\mathbf P$ and having that plus $\mathbf b$ we can find $\boldsymbol \mu$. Those observations complete everything we need to know for GP estimation.

# The shady part
The pipeline is as follows - from problem statement we know that many things are linear and Gaussian. However what we do is we first marginalize $\mathbf x$ to find $\boldsymbol\theta$ estimation and $\sigma$ estimation. This allows us for alligment and to understand the noise. After that we find $\mathbf x$ which is the best fitting estimation to **our estimation**. We actually estimate using a estimate - which is shady... 

If we were to estimate everything at the same time, meaning estimate alligment $\boldsymbol\theta$, noise $\sigma$ and the generating HR source image $\mathbf x$ the problem would be intractable. Doing this in two steps however allows for marginalization yielding Gaussian and estimation from conditional yielding again Gaussian. 

# Computational challenges
**Efficient estimator computation**
We have established that:
$$
\begin{align}
(\boldsymbol {\hat\theta}, \hat \sigma) =
\arg\max_{(\boldsymbol\theta, \sigma)} \int p(\mathbf x)p(
\mathbf Y|\mathbf x,\boldsymbol \theta, \sigma)d\mathbf x
\end{align}
$$
Which means we basically marginalize over $\mathbf x$. Therefore - speaking of the inregral inside this expression - we have a Gaussian integral of product of two Gaussians since we have $\mathbf x \sim \mathcal N(\boldsymbol \mu_0, \mathbf K$) as a a'priori distribution of $\mathbf x$ which we need for $p(\mathbf x)$ and the $p(\mathbf Y|\mathbf x, \boldsymbol \theta, \sigma)$ is also Gaussian from intial equation. This means that
$$
\mathbf Y|\boldsymbol \theta, \sigma \sim
\mathcal N(
	\mathbf A(\boldsymbol\theta)\boldsymbol\mu_0,
	\mathbf S(\boldsymbol \theta, \sigma ^2, \mathbf K))
$$
$$
\mathbf S(\boldsymbol \theta, \sigma, \mathbf K) = 
\sigma^2\mathbf I+\mathbf A(\boldsymbol \theta)\mathbf K\mathbf A(\boldsymbol \theta)
$$
Given that fact we are only left with finite-dimensional optimization problem over $\boldsymbol \theta$ and $\sigma$. Since we now that this is in fact a Gaussian we have a closed form solution for log-evidance over $\boldsymbol \theta$ and $\sigma$
$$
\mathcal L(\boldsymbol \theta, \sigma) = \log(\mathbf Y|\boldsymbol \theta, \sigma) = -\frac{1}{2}\mathbf r^T\mathbf S^{-1}\mathbf r-\frac{1}{2}\log|\mathbf S| + const. 
$$
Now we can hit this with plain SGD. Having that we are provided with complete parameter set.

**Efficient final estimation**
$$
\boldsymbol{\Sigma}_x^{-1}=\frac{1}{\sigma^2}\sum_k\mathbf A_k^T\mathbf A_k +\mathbf K^{-1}
$$
$$
\boldsymbol{\Sigma}_x^{-1}\boldsymbol{\mu}_x = 
\frac{1}{\sigma^2}\sum_k \mathbf A_k^T\mathbf y^k + 
\mathbf K^{-1}\boldsymbol{\mu}_0
$$
Since $\boldsymbol{\Sigma}_x^{-1}$ is a symmetric, positive deifinite we can use conjugate gradient. Which is cool, since we dont need to ever from a complete co-variance matrix, but rather keep current iteration value. This is already save on memory usage from $O(n^2)$ to $O(n)$ - note that if we have a HR image of 1000x1000 this matrix would already be $10^6$x$10^6$ which with floats would already be in range of TB.

# Computational pipeline
## Setup
**Warp - rotation and shift**
From computational perspective it is much easier to think of the image a two-dimensional signal. We basically have a image which is a function $I:\mathbb R^2 \rightarrow C$, which for each coordinate, which is a float will assign some color. In this case a digitalize image, would be kind off a step-function. This allows us to formulate more precisely what the $\mathbf A_k$ matrix really is. Lets assume that rotation is centered around point $\mathbf c$. For every position $\mathbf p$ we can compute at what pixel of the image we are looking at. Given $\theta_k = (\phi, \mathbf t)$:
$$
T_\theta(\mathbf p)=\mathbf R(\phi)(\mathbf p - \mathbf c) + \mathbf c + \mathbf t
$$
Where $\mathbf R$ is simply a rotation matrix. Let this rotatoin and shift be captured in matrix $\mathbf W(\theta)$. Now we sample the high res image $\mathbf x$ by bi-linear interpolation to achive pixel color at position $\mathbf p$:
$$
(\mathbf W(\theta)\mathbf x)(\mathbf p) = 
\sum_{\mathbf q} \beta(T_\theta(\mathbf p) - \mathbf q)\cdot\mathbf x(\mathbf q)
$$
This basically means that we take weighted sum neighbours being $4$ closest pixel in high res image.

**Blur - cameras point spread function**
$\mathbf G$ is a convolution with Gaussian kernel of standard deviation $s$, measured in HR pixels. This value is known and means how much do surroding pixels leak to this pixel. This has to be set prior, and can be measured for the optical system. One can assume that this is like $s = 0.4r$

**Decimation - downscaling operator**
Basically we sample every other pixel. In this case we sample every $r$-th pixel.
$$
\begin{align}
(\mathbf D\mathbf z)(i, j) = \mathbf z(ri + o, rj + o) &&
o = \lfloor{\frac{r}{2}}\rfloor
\end{align}
$$
**Joining all elements**
$$
\mathbf A_k = \mathbf D\mathbf G\mathbf W(\theta_k)
$$
$$
\mathbf A_k^T = \mathbf W(\theta_k)^T \mathbf G^T\mathbf D^T
$$
Also assume that:
$$
\mathbf A_1 = \mathbf D \mathbf G
$$
which means:
$$
\mathbf W(\theta_1) = \mathbf I_d 
$$
**Setting up the prior**
For $\boldsymbol \mu_0$ we can start with bi-linear upsampling. This is best guess before as the prior. Now for the covariance:
$$
\mathbf K_{ij} = \eta^2\exp(-\frac{||\mathbf r_i - \mathbf r_j||^2}{2\ell^2})
$$
The issue we have here is that $\mathbf K$ has $N^2$ elements which is huge. This matrix however has very regular structure. It basically can be described by two matricies $\mathbf K_u \in \mathbb R^{H\times H}$ and $\mathbf K_v \in\mathbb R^{W\times W}$ where:
$$
(\mathbf K_u)_{ij} = \exp\left(-\frac{(i - j)^2}{2\ell^2}\right)
$$
which basically defines correlation along the rows. The other matrix is essentially the same but defining correlation along columns. If we apply this matrix to image $\mathbf V \in \mathbb R^{H \times W}$:
$$
\mathbf K\mathbf V = \eta^2\mathbf K_u\mathbf V\mathbf K_v^T
$$
