# Example: Neoclassical growth model, stochastic version


## Model setup

$$
\begin{aligned}
& v(k;z) = \max_{c} \frac{c^{1-\gamma}}{1-\gamma} + \beta \mathbb{E}\left\{ v(k';z') | z \right\}  \\
\text{s.t. }& k' = e^z k^\alpha + (1-\delta) k - c & \text{(endo state equation)} \\
& z' \sim \text{AR}(1)(\rho,\sigma,\overline{z}) & \text{(exog state process)}  \\
& 0 \leq c < \infty   & \text{(box constraints of control)} \\
& 0 \leq k' < \infty
\end{aligned}
$$

where the AR(1) process has formula $z' = \rho z + (1-\rho) \overline{z} + \sigma \varepsilon$ where $\varepsilon \sim N(0,1)$.

The model, when solved numerically, often does the following parameterization:

- Approximating $z$'s process with a finite-state Markov chain $(\mathbf{Z},P_z)$ where $\mathbf{Z}$ is the grid space of $z$ and $P_z$ is transition matrix. This approximation can be done by Tauchen's or other methods.
- Bounding the computation domain of $k$ to a finite grid $\mathbf{K}$ in which the smallest $k$ state is small but still positive. The box constraint of $k'$ is updated to $k_{\min} \leq k' \leq k_{\max}$.
- Transforming $c$'s lower and upper bounds to $\max\{y-k_{\max},0\} \leq c \leq \min\{y-k_{\min},k_{\max}\}$, which are derived from the new box constraint of $k'$ respectively. The notation $y:=e^z k^\alpha + (1-\delta) k$ is the disposable income in this period. 
- A small (typically machine-precision float) positive number $\epsilon$ is added to $c$ in the CRRA utility to avoid undefined behavior at the boundary.

Then, the practical approximated model to compute is:

$$
\begin{aligned}
& v(k;z) = \max_{c} \frac{(c+\epsilon)^{1-\gamma}}{1-\gamma} + \beta \mathbb{E}\left\{ v(k';z') | z \right\}  \\
\text{s.t. }& k' = e^z k^\alpha + (1-\delta) k - c & \text{(endo state equation)} \\
& z' \sim \text{MarkovChain}(\mathbf{Z},P_z) & \text{(exog state process)}  \\
& \max\{y-k_{\max},0\} \leq c < \min\{y-k_{\min},k_{\max}\}   & \text{(box constraints of control)} \\
\end{aligned}
$$

Meanwhile, suppose we are also interested in the following extra moments/statistics of the solved/equilibrium model and want to collect them together with solivng the problem:
- Disposable income $y$
- Saving rate $1-c/y$


## Fitting into the framework

Let's fit the approximated model into the programmatic framework of `LifetimeProblems.jl`. The following table collects all elements that are required to define an `

|Item|Specification|
|----|----------|
|Programming horizon | infinite, transversality condition satisfied |
|Time homogeneity  | time-homogeneous |
|Endo state(s) and grid| $\mathbf{z}:=(k,)$, dimensionality $D_x=1$; grid space $\mathbf{X}$ |
|Exog state(s) and process | $\mathbf{z}:=(z,)$, $D_z=1$, but have $N_z>1$; follows an AR(1) approximated by MarkovChain$(\mathbf{Z},P_z)$  |
|Control(s)   | $\mathbf{c}:=(c,)$, only $D_c=1$, all controls are continuous   |
|Flow utility | $u(\mathbf{x},\mathbf{z},\mathbf{c}) := \frac{(c+\epsilon)^{1-\gamma}}{1-\gamma}$, CRRA |
|Endo state equation | $f(\mathbf{x},\mathbf{z},\mathbf{c}):=y -c$  |
|Lower bound of $\mathbf{c}$ | $\text{lb}_c(\mathbf{x},\mathbf{z}) := \max\{y-k_{\max},0\}$, state-dependent  |
|Upper bound of $\mathbf{c}$ | $\text{lb}_c(\mathbf{x},\mathbf{z}) := \max\{y-k_{\max},0\}$, state-dependent  |
|Generic equation of $\mathbf{c}$ | $g(\mathbf{x},\mathbf{z}) := []$, nothing, $D_g=0$  |
|Extra statistics | $s(\mathbf{x},\mathbf{z},\mathbf{c}) := [y,1-c/y]^T$, $D_s=2$  |
|Discounting | $\beta < 1$ |
|Maximization solver| Brent's method, because of $D_c=1$ and only box constrained |


Check `main.jl` for the programs and visualization.
