# Regression

## Linear Regression

### Step1: Model

$$
y = b + \sum w_i x_i
$$

$x_i:\ feature$

$w_i:\ weight$

$b:\ bias$

### Step2: Goodness of Function

**Loss function $L$:**

$$
L(f)=L(w,b)=\sum_n \Big(\hat{y}^n-(b+\sum w_i x_i^n)\Big)^2
$$

### Step3: Gradient Descent

## Another Model

$$
y = b + \sum w_i x_i + \sum w_i x_i^2
$$

**Overfitting?**

## Modification

### Back to Step1: Redesign the Model

$$
\begin{aligned}
y& = \delta(class=class\_1)(b_1 + w_1 x_i)\\
 & + \delta(class=class\_2)(b_2 + w_2 x_i)\\
 & +\dots +\\
 & + \delta(class=class\_n)(b_n + w_n x_i)
\end{aligned}
$$

### Back to Step1: Regularization
$Assume\ that$

$$
y = b + \sum w_i x_i
$$

$Let$

$$
L(f) = \sum_n \Big(\hat{y}^n-(b+\sum w_i x_i^n)\Big)^2 + \lambda \sum(w_i)^2
$$

The loss function make the result function smoother.

If some noises corrupt input $x_i$ when testing, a smoother function has less influence.

But don't be too smooth.

In addition, $b$ doesn't affect the smoothness.



# Classification: Probabilistic Generative Model

## Idea: Take Binary Classification as an example
### Function (Model)

$$
f(x)=
\begin{cases}
&g(x) > 0,\ Output = class\ 1,\\
&else,\ Output = class\ 2
\end{cases}
$$

### Loss Function
$$
L(f)= \sum \delta(f(x^n) \not= \hat{y}^n )
$$

### Find the best function
- Example: Perception, SVM

## Details

### Function Set (Model)

$Calculate$

$$
\mathbb{P}(C_1|x)=\cfrac{\mathbb{P}(x|C_1)\mathbb{P}(C_1)}{\mathbb{P}(x|C_1)\mathbb{P}(C_2)+\mathbb{P}(x|C_2)\mathbb{P}(C_1)}
$$

$If \ \mathbb{P}(C_1|x) > 0.5,\ Output:\ Class1;\ Otherwise,\ Output:\ Class2$.

### Goodness of a function
Take Gaussian distribution as an example:

- The mean $\mu_i$ and covariance $\Sigma_i$ $(i=1,2,\dots, n,\ where\ n\ is\ the\ number\ of$ $classes )$ that maximizing the likelihood (the probability of generating data)

- ($Modifying\ Model$) If the accuracy is not so hign, we can make the Gaussian of all class share th same covariance $\Sigma$. For example, we can choose $\Sigma=\frac{1}{n}(\Sigma_1+\Sigma_2+\dots+\Sigma_n)$. Then, find the value of $\mu_i$. If you do so, the boundary will be linear. (See in **Reflection**)

### Find the best function

## Remark
If you assume all the dimensions are independent, then you can use $\textbf{Naive Bayes Classifier}$. In this case, if 

$$
x=
\begin{bmatrix}
x_1\\
x_2\\
\vdots
\\
x_K
\end{bmatrix}
$$

then 

$$
\mathbb{P}(C_1|x)=\mathbb{P}(C_1|x_1)\mathbb{P}(C_1|x_2)\dots\mathbb{P}(C_1|x_K)
$$

## Reflection
$Consider\ Binary\ Classification:$

$$
\begin{aligned}
\mathbb{P}(C_1|x)&=\cfrac{\mathbb{P}(x|C_1)\mathbb{P}(C_1)}{\mathbb{P}(x|C_1)\mathbb{P}(C_2)+\mathbb{P}(x|C_2)\mathbb{P}(C_1)}\\
&=\cfrac{1}{1+\cfrac{\mathbb{P}(x|C_2)\mathbb{P}(C_2)}{\mathbb{P}(x|C_1)\mathbb{P}(C_1)}}=\cfrac{1}{1+\text{exp}(-z)}\\
&=\sigma(z) \quad (\textbf{Sigmoid Function})
\end{aligned}
$$

$then$

$$
z=\ln \cfrac{\mathbb{P}(x|C_1)\mathbb{P}(C_1)}{\mathbb{P}(x|C_2)\mathbb{P}(C_2)}
=\ln \cfrac{\mathbb{P}(C_1)}{\mathbb{P}(C_2)}+\ln \cfrac{\mathbb{P}(x|C_1)}{\mathbb{P}(x|C_2)}
$$

$$
\ln \cfrac{\mathbb{P}(C_1)}{\mathbb{P}(C_2)}= \ln \cfrac{N_1}{N_2}\\
{}\\
\ln \cfrac{\mathbb{P}(x|C_1)}{\mathbb{P}(x|C_2)}= \ln \cfrac{(\Sigma^2)^{1/2}}{(\Sigma^1)^{1/2}} - \cfrac{1}{2}[(x-\mu^1)^T(\Sigma^1)^{-1}(x-\mu^1)-(x-\mu^2)^T(\Sigma^2)^{-1}(x-\mu^2)]
$$

$If\ we\ choose\ \Sigma=(\Sigma_1+\Sigma_2)/2,$

$$
z = (\mu^1-\mu^2)^T\Sigma^(-1)x - \cfrac{1}{2}(\mu^1)^T\Sigma^{-1}\mu^1 + \cfrac{1}{2}(\mu^2)^T\Sigma^{-1}\mu^2 + \ln \cfrac{N_1}{N_2}
$$

$The\ rsult\ can\ be\ written\ as$

$$
z=wx+b
$$

Thus

$$
\mathbb{P}(C_1|x) = \sigma(wx+b)
$$

**How about directly find $w$ and $b$ ?**

# Logistic Regression

## Binary Blassfication

### Function Set

$$
f_{w,b}(x) = \mathbb{P}(C_1|x) =\sigma(wx+b) =\sigma (\sum_iw_ix_i+b)
$$

### Goodness of a Function
Suppose our Training Data is

$$
\begin{matrix}
x^1 \quad & x^2 \quad & \dots \quad & x^N \\
C_1 \quad & C_2 \quad & \dots \quad  & C_1 
\end{matrix}
$$

then the Likelyhood Function is

$$
\begin{aligned}
L(w,b) &= \mathbb{P}(C_1|x^1)\mathbb{P}(C_2|x^2)\cdots\mathbb{P}(C_1|x^N)\\
&= \mathbb{P}(C_1|x^1)(1-\mathbb{P}(C_1|x^2))\cdots\mathbb{P}(C_1|x^N))\\
&= f_{w,b}(x^1)(1-f_{w,b}(x^2))\cdots f_{w,b}(x^N)
\end{aligned}
$$

thus

$$
w^{\ast},\ b^{\ast} = \argmax_{w,\ b}L(w,b) = \argmin_{w,\ b} -\ln L(w,b)
$$

we can call $-\ln L(w,b)$ the Loss Function

$$
\begin{aligned}
-\ln L(w,b)
&=-\ln f_{w,b}(x^1) - \ln(1-f_{w,b}(x^2))- \cdots\\
&=-[\hat{y}^1\ln f_{w,b}(x^1) + (1-\hat{y}^1) \ln(1-f_{w,b}(x^1))]\\
&\quad \ \ -[\hat{y}^2\ln f_{w,b}(x^2) + (1-\hat{y}^2) \ln(1-f_{w,b}(x^2))]\\
&\quad \ \ -\cdots
\end{aligned}
$$

namely

$$
-\ln L(w,b) = \sum_n -[\hat{y}^n\ln f_{w,b}(x^n) + (1-\hat{y}^n) \ln(1-f_{w,b}(x^n))]
$$

we call the content in the bracket as $\textbf{Cross Entrophy}$ betweem two Bernoulli distribution.

In fact, for two bernoulli fistribution $p$ and $q$, their cross entrophy is 

$$
H(p,q)= -\sum_xp(x)\ln(q(x))
$$

**Why don't we simply use square error linear regression?**

### Find the best function

$$
\cfrac{\partial{(-\ln L(w,b))}}{\partial w_i}=-\sum_n \Big(\hat{y}^n- f_{w,b}(x^n) \Big)x_i^n
$$

We can see that its gradient descent is the same as that of linear regression regardless of coefficient.

## Why not "Logistic Regression + Square Error"？

If we do so, 

$$
    \cfrac{\partial{(f_{w,b}(x)-\hat{y})^2}}{\partial w_i}=2(f_{w,b}(x)-\hat{y})f_{w,b}(x)(1-f_{w,b}(x))x_i
$$

thus, if $\hat{y}^n=1$, whether $f_{w,b}(x)=1$ or $f_{w,b}(x)=0$, $\partial L/\partial w_i = 0$, which means the result is far from the target or is close to the target, it won't update.

## Discriminative v.s. Generative

In this section, we directly find $w$ and $b$, the result is not equal to that obtained using the method in the last section, since the latter assume that the data is subject to normal distribution.

**Bebefit of generative model**
- With the assumption of probbility distribution, less training data is needed.
- With the assumption of probbility distribution, more robust to the noise.
- Priors and class-dependent probabilities can be estimated from different sources. 

## Multi-class Classification

$$
C_1: w^1,b^1 \quad z_1 = w^1 x + b_1\\
C_2: w^2,b^2 \quad z_2 = w^2 x + b_2\\
C_1: w^3,b^3 \quad z_3 = w^3 x + b_3\\
(w^i,x\ can\ be\ vetors)
$$

$\textbf{Softmax}:$

$$
y_1=\text{e}^{z_1}\Big/ \sum^3_{j=1} \text{e}^{z_j}\\
y_2=\text{e}^{z_2}\Big/ \sum^3_{j=1} \text{e}^{z_j}\\
y_3=\text{e}^{z_3}\Big/ \sum^3_{j=1} \text{e}^{z_j}
$$

Here $y_i$ can be consedered as the estimation of $\mathbb{P}(C_i|x)$.

So we get

$$
y^i=
\begin{bmatrix}
y_1^i\\
y_2^i\\
y_3^i
\end{bmatrix}
$$

Now consider the target:

$If x \in Class\ 1$

$$
\hat{y}=
\begin{bmatrix}
1\\
0\\
0
\end{bmatrix}
$$

$If x \in Class\ 2$

$$
\hat{y}=
\begin{bmatrix}
0\\
1\\
0
\end{bmatrix}
$$

$If x \in Class\ 3$

$$
\hat{y}=
\begin{bmatrix}
0\\
0\\
1
\end{bmatrix}
$$

Loss Function:

$$
L(f)= - \sum_i(\hat{y}^i)^T\ln y^i
$$

## Limitation

The boundary is linear.

**Solution**

- Feature Transformation

- Cascading logistic regression models (Make thr machine deal with Feature Transformation itself)

![Figure 1](./graphs/figure1.jpg)
![Figure 2](./graphs/figure2.jpg)

#  Adaptive Learning Rate

## Fundamental Optimizers

**Training Stuck $\not=$ Small Gradient**

In fact, it may also be caused by learning rate.

**However, learning rate can't be one-size-fits-all.**

- Off-line: pour all $(x_t,\hat{y}_t)$ into the model at every time step
 
The precedent method (SGD or SGDM):

$$
\theta_i^{t+1} \leftarrow \theta_i^t - \eta g_i^t\\
g_i^t = \cfrac{\partial L}{\partial \theta_i} \Big|_{\theta=\theta^t}
$$

Modified method:

$$
\theta_i^{t+1} \leftarrow \theta_i^t - \cfrac{\eta}{\sigma_i^t} g_i^t
$$

Momentum: weighter sum of the previous gradients.

$$
\theta_i^{t+1} \leftarrow \theta_i^t - \cfrac{\eta^t}{\sigma_i^t} m_i^t
$$

## How to choose $\sigma$ ?

### Adagrad (for different parameters)

$$
\begin{aligned}
&\theta_i^1 \leftarrow \theta_i^0 - \cfrac{\eta}{\sigma_i^0} g_i^0    &\sigma_i^0=\sqrt{(g_i^0)^2}=|g_i^0|   \quad \ \ \ \\
&\theta_i^2 \leftarrow \theta_i^1 - \cfrac{\eta}{\sigma_i^1} g_i^1    &\sigma_i^1=\sqrt{\cfrac{1}{2}[(g_i^0)^2+(g_i^1)^2]} \\
&\ \ \vdots   &\vdots \qquad \qquad \qquad\qquad\quad\\
&\theta_i^{t+1} \leftarrow \theta_i^t - \cfrac{\eta}{\sigma_i^t} g_i^t   &\sigma_i^0=\sqrt{\cfrac{1}{t+1}\sum_{i=0}^{t}(g_i^t)^2} \ \ \,
\end{aligned}
$$

### RMSProp (learning rate adapts dynamically)

$$
\begin{aligned}
&\theta_i^1 \leftarrow \theta_i^0 - \cfrac{\eta}{\sigma_i^0} g_i^0    &\sigma_i^0=\sqrt{(g_i^0)^2}=|g_i^0|   \qquad\qquad\quad\,\\
&\theta_i^2 \leftarrow \theta_i^1 - \cfrac{\eta}{\sigma_i^1} g_i^1    &\sigma_i^1=\sqrt{\alpha(\sigma_i^0)^2+(1-\alpha)(g_i^1)^2} \ \ \ \\
&\ \ \vdots   &\vdots \qquad \qquad \qquad\qquad\qquad\quad\quad\\
&\theta_i^{t+1} \leftarrow \theta_i^t - \cfrac{\eta}{\sigma_i^t} g_i^t   &\quad\sigma_i^t=\sqrt{\alpha(\sigma_i^{t-1})^2+(1-\alpha)(g_i^t)^2} \,
\end{aligned}
\\
where\ \alpha \in (0,1)
$$

### Adam

### SWATS

Begin with Adam (fast), end with SGDM (stable

## Improvement

### Learning Rate Scheduling (SGD/SGDM)

$$
\theta_i^{t+1} \leftarrow \theta_i^t - \cfrac{\eta^t}{\sigma_i^t} g_i^t
$$

**Learning Rate Decay**

As the training goes, we are close to the destiantion, so we reduce the learning rate.

**Warm Up**

Increase at first and then decrease, since at the beginning, the estimate of $\sigma_i^t$ has large variance.

**Cyclical LR**

**One-cycle LR**

**SGDR**

### AMSGrad (Adam)


# Selection of dataset

$h$ is a unkown parameter of our function $f$, then we call it $f_h$
Let $\mathcal{H}$ be the set that contains all possible value of $h$.

## What does "$ \mathcal{D}_{train} $ is good" mean?

Usually, 

$$
L(h^{train},\mathcal{D}_{all}) \geq L(h^{all},\mathcal{D}_{all})\\
L(h^{train},\mathcal{D}_{train}) \leq L(h^{all},\mathcal{D}_{train})
$$

We hope

$$
L(h^{train},\mathcal{D}_{all}) -L(h^{all},\mathcal{D}_{all}) \leq \delta
$$

A sufficient condition of the inequality is

$$
\forall h \in \mathcal{H},|L(h,\mathcal{D}_{train}) -L(h,\mathcal{D}_{all}) \leq \delta| \leq \delta/2=\varepsilon
$$

## What is the probability of sampling bad $\mathcal{D}_{train}$ ?

$\textbf{Hoeffding's Inequality}$

$$
\mathbb{P}(\mathcal{D}_{train}\ is\ bad\ due\ to\ h) \leq 2 \text{exp}(-2N\varepsilon^2),\\
where\ N\ is\ the\ number\ of\ examples\ in\ \mathcal{D}_{train}
$$

then

$$
\begin{aligned}
\mathbb{P}(\mathcal{D}_{train}\ is\ bad)&=\mathbb{P}(\{\bigcup_{h \in \mathcal{H}}\mathcal{D}_{train}\ is\ bad\ due\ to\ h\})\\
&\leq \sum_{h \in \mathcal{H}} \mathbb{P}(\mathcal{D}_{train}\ is\ bad\ due\ to\ h)\\
&\leq \sum_{h \in \mathcal{H}} 2\text{exp}(-2N\varepsilon^2)\\
&=|\mathcal{H}| \cdot 2\text{exp}(-2N\varepsilon^2)
\end{aligned}
$$

**VC-dimension**

## Tradeoff of Model Complexity

Smaller $\mathbb{P}(\mathcal{D}_{train}\ is\ bad) \Longrightarrow$ Smaller $\delta \Longrightarrow$ Larger $N$ and smaller $|\mathcal{H}|$ $\Longrightarrow$ Larger $L(h^{all},\mathcal{D}_{all})$
