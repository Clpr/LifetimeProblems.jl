# Example: Price setting problem with Rotemberg price rigidity


## Model setup

Let's consider a price setting problem with Rotemberg-style nominal price rigidity. Such problems are standard in NK literature. To partial such a problem out a larger model, ad hoc, I assume the marignal cost $m$ and demand $y$ follow a VAR(1) process with **correlations**.


```math
\begin{aligned}
v(p_{-1};m,y) =& \max_{\pi} \left[ \left( 1 - \frac{\theta}{2}\left(\pi - \overline{\pi} \right)^2  \right) p  - m \right] \cdot y + \beta \mathbb{E}\left\{ v(p;m',y') |m,y  \right\}  \\
\text{s.t. }&p = \pi p_{-1}   \\
& (m',y') \sim \text{VAR}(1)(\rho \in \mathbb{R}^{2\times 2},\Sigma \in\mathbb{R}^{2\times 2}, [\overline{m},\overline{y}]) \\
& p > 0
\end{aligned}
```

where

```math
\begin{bmatrix}
m' \\ y'
\end{bmatrix} = \rho \cdot \begin{bmatrix}
m \\ y
\end{bmatrix} + (1-\rho) \begin{bmatrix}
\overline m \\ \overline y
\end{bmatrix} + \Sigma \cdot \vec\varepsilon, \vec\varepsilon \sim N(0,I)
```


**Parameterization and transformation**

- Discretize the VAR(1) process of $(m,y)$ to a Markov chain $(\mathbf{Z}_{m,y},P_{m,y})$ by frequency estimatation of a simulated path (API available in `MultivariateMarkovChains.jl`). Grid space of the two variables are joinned by Cartesian product.
- Bounds $p$ to an interval $[p_{\min},p_{\max}]$.
- Directly choose $p$ rather than $\pi$. The inflation $\pi$ is now a statistics.

The approximate problem is then:


```math
\begin{aligned}
v(p_{-1};m,y) =& \max_{p} \left[ \left( 1 - \frac{\theta}{2}\left(\frac{p}{p_{-1}} - \overline{\pi} \right)^2  \right) p  - m \right] \cdot y + \beta \mathbb{E}\left\{ v(p;m',y') |m,y  \right\}  \\
\text{s.t. }& (m',y') \sim  \text{MarkovChain}(\mathbf{Z}_{m,y},P_{m,y}) \\
& p_{\min} \leq p \leq p_{\max}
\end{aligned}
```

Meanwhile, we are also interested in the following statistics:

1. premium $q := p - m$
2. adjustment cost $\varphi := \frac{\theta}{2}\left(\pi - \overline{\pi} \right)^2 p$
3. profit amount $w := (q - \varphi) y$
4. inflation $\pi := p/p_{-1}$






## Fitting into the framework

Let's fit the approximated model into the programmatic framework of `LifetimeProblems.jl`. The following table collects all elements that are required to define an `


|Item|Specification|
|----|----------|
|Programming horizon | infinite, transversality condition satisfied |
|Time homogeneity  | time-homogeneous |
|Endo state(s) and grid| $\mathbf{x}:=(p_{-1},)$, dimensionality $D_x=1$; grid space $\mathbf{X}$ |
|Exog state(s) and process | $\mathbf{z}:=(m,y)$, $D_z=2$; follows a VAR(1) approximated by MarkovChain$(\mathbf{Z}_{m,y},P_{m,y})$  |
|Control(s)   | $\mathbf{c}:=(p,)$, $D_c=1$, all controls are continuous   |
|Flow utility | $u(\mathbf{x},\mathbf{z},\mathbf{c}) := \left[ \left( 1 - \frac{\theta}{2}\left(\pi - \overline{\pi} \right)^2  \right) p  - m \right] \cdot y$ |
|Endo state equation | $f(\mathbf{x},\mathbf{z},\mathbf{c}):= p\pi$   |
|Lower bound of $\mathbf{c}$ | $`\text{lb}_c(\mathbf{x},\mathbf{z}) := [p_{\min},]^T`$, state-dependent  |
|Upper bound of $\mathbf{c}$ | $`\text{ub}_c(\mathbf{x},\mathbf{z}) := [p_{\max},]^T`$, state-dependent  |
|Generic equation of $\mathbf{c}$ | $g(\mathbf{x},\mathbf{z},\mathbf{c}) := []$, $D_g=0$  |
|Extra statistics | $s(\mathbf{x},\mathbf{z},\mathbf{c}) := [q,\varphi,w]^T$, $D_s=3$  |
|Discounting | $\beta < 1$ |
|Maximization solver| Brent's method, because: $D_c>1$; there is no generic constraint $D_g=0$ |


Check `main.jl` for the programs and visualization.



