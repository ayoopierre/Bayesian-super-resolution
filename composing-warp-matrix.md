# Composing the warp matrix
One important part of the algorithm is to apply the parametrized, rotation, shift, blur and decimation matrix. It might not be clear how it looks like. For a single point/pixel those operations are clear. Basically rotation and shift can be done by clasicall rotations and shift affine transformations. However pixels are discrete - in this case when we land in non-integer (for pixel position) places, we can bi-lerp between surrouding pixels. Now blur and decimation kind off do the same. This means that in complete transformation matrix, on whole image - represented as flattened image - can simply be a matrix that samples with diffrent weights pixels (entries) in the image vector.

From computational standpoint this matrix can be created, by calculating for a pixel where it would land and sample.

## Formalities
Let image $\mathbf I$ be a vector flattened from $\mathbb R^{H\times W}\rightarrow\mathbb R^{HW}$. We use ordering:
$$
k(x, y) = yW + x
$$
where $k(x, y)$ is a index in $\mathbf I$ that corresponds to appropriate $(x, y)$ pixel. For each pixel $(x, y)$ we can compute the source location $(u, v_s)$:
$$
\begin{bmatrix}
u \\
v_s \\
1
\end{bmatrix} = M^{-1}
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
$$
We use inverse to find pixels from original image that we have used. Now $M$ can be composed out of:
$$
\begin{align*}
R=\begin{bmatrix}
\cos \theta & -\sin \theta & 0 \\
\sin \theta & \cos\theta & 0 \\
0 & 0 & 1
\end{bmatrix}
&&
T=\begin{bmatrix}
1 & 0 & d_x \\
0 & 1 & d_y \\
0 & 0 & 1
\end{bmatrix}
\end{align*}
$$
Where:
$$
M = T(d_x, f_y)T(c_x, c_y)R(\theta)T(-c_x, -c_y)
$$
this is basically, center to the begining of the axis, rotate pixel, move back and apply shift. Now we know which pixel to access:
$$
\begin{align*}
i = \lfloor u\rfloor && j = \lfloor v_s \rfloor && \alpha = u - i && \beta = v_s - j
\end{align*}
$$
Now we want to access: $(i, j)$, $(i + 1, j)$, $(i, j + 1)$, $(i + 1, j + 1)$. We apply bi-lerp with weights:
$$
\begin{align*}
w_{00} &= (1 - \alpha)(1 - \beta) \\
w_{10} &= \alpha(1-\beta) \\
w_{01} &= (\alpha - 1)\beta \\
w_{11} &= \alpha\beta
\end{align*}
$$
Therefore we know what to put in the matrix:
$$
\begin{align*}
k_{00} &= jW + i \\
k_{10} &= jW + (i + 1) \\
k_{01} &= (j + 1)W + i \\
k_{11} & = (j + 1)W + (i + 1)
\end{align*}
$$
So now matrix is:
$$
\mathbf A_{r, k} = \begin{cases}
(1 - \alpha)(1 - \beta) & k=k_{00} \\
(1 - \alpha)\beta & k=k_{10} \\
\alpha(1 - \beta) & k=k_{01} \\
\alpha\beta & k=k_{11} \\
0 & k \\
\end{cases}
$$
**Note** - this means that theoretically we can create this matrix by iterating over all pixels and using the algorithm above.

In general:
$$
\begin{aligned} i&=\lfloor u_{xy}\rfloor,& \alpha&=u_{xy}-i,\\\ j&=\lfloor v_{xy}\rfloor,& \beta &= v_{xy}-j. \end{aligned} 
$$
$$
\begin{aligned} A_{yW+x,\\,pW+q} ={}&(1-\alpha)(1-\beta) \mathbf1[p=j,\ q=i]\\\ &+\alpha(1-\beta) \mathbf1[p=j,\ q=i+1]\\\ &+(1-\alpha)\beta \mathbf1[p=j+1,\ q=i]\\\ &+\alpha\beta \mathbf1[p=j+1,\ q=i+1], \end{aligned}
$$

## Handling out of bounds sampling
There are couple ways we could handle that part:
1. Zero out out of bounds pixels
2. Continue color we had
3. Use more advance algoithm for larger domain reconstruction
4. Use padding - calculate using central snippted image in the middle

## Practical aspects
Composing the complete transformation matrix may be unpractical. Basically this matrix is in $\mathbf A \in \mathbb R^{HW \times HW}$ which even for modest image sizes may be quite huge. Therefore we shall avoid building the complete matrix and rather perform pixel-by-pixel computation.