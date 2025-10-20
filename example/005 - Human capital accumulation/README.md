# Example: Lucas (1988) human capital accumulation model with productivity shock

## Reference

Robert E. Lucas Jr., “*On the Mechanics of Economic Development*,” **Journal of Monetary Economics**, 1988. 


## Model setup

Let's consider a representative agent that owns physical capital $k_t$ and human capital $h_t$.


```math
\begin{aligned}
& v(k,h;z) = \max_{c,n} \frac{c^{1-\gamma}}{1-\gamma} - \theta \frac{n^{1+\nu}}{1+\nu} + \beta \mathbb{E}\left\{ v(k',h';z') | z \right\} \\
\text{s.t. }& k' = e^z k^\alpha (n h)^{1-\alpha} + (1-\delta_k) k - c \\
& h' = (1-\delta_h) h + \phi(1-n)h \\
& z' \sim \text{AR}(1)(\rho,\sigma,\overline{z}) \\
& c \geq 0, n \in [0,1] \\
& k \geq 0, h \geq 0
\end{aligned}
```

where $n$ is the share of time endowment invested into work; $\phi$ is the education technology parameter.


**Parameterization and transformation**

- Discretize $z$'s process to a Markov chain $(\mathbf{Z},P_z)$.
- Add a small positive value $\epsilon$ to the flow utility for numerical stability.
- Bounds the endogenous state's space to $k_{\min}\leq k' \leq k_{\max}$ and $h_{\min}\leq h' \leq h_{\max}$. 
- Change the control variables from $(c,n)$ to $(k',n)$ to remove the unbounded box constraint of $c$ say $c\in[0,\infty)$.


Based on the new box constraints, we derive the following new box constraints for $(k',n)$. Notice that the box constraint of $h'$ implies simple box constraint on $n$:

```math
\begin{aligned}
& h' \geq h_{\min} \\
\implies& (1-\delta_h)h + \phi(1-n)h \geq h_{\min} \\
\implies& n \leq 1- \frac{1}{\phi}\left[ \frac{h_{\min}}{h} - (1-\delta_h) \right] \\
\overset{n\leq 1}{\implies}& n \leq \min\left\{ 1,  1- \frac{1}{\phi}\left[ \frac{h_{\min}}{h} - (1-\delta_h) \right]  \right\}
\end{aligned}
```

```math
\begin{aligned}
& h' \leq h_{\max} \\
\implies& (1-\delta_h)h + \phi(1-n)h \geq h_{\max} \\
\implies& n \geq 1- \frac{1}{\phi}\left[ \frac{h_{\max}}{h} - (1-\delta_h) \right] \\
\overset{n\geq 0}{\implies}& n \geq \max\left\{ 0,  1- \frac{1}{\phi}\left[ \frac{h_{\max}}{h} - (1-\delta_h) \right]  \right\}
\end{aligned}
```

Meanwhile, we notice that, even though the control variable change from $c$ to $k$ trivially applies the box constraints $k'\in[k_{\min},k_{\max}]$, the non-negative consumption constraint implies a new generic constraint that simultaneously depend on $(k',n)$ and cannot be formulated as box constraints:


```math
\begin{aligned}
& c \geq 0 \\
\implies & e^z k^\alpha (n h)^{1-\alpha} + (1-\delta_k) k - k' \geq 0 \\
\implies & k' - \underbrace{ \left\{ e^z k^\alpha (n h)^{1-\alpha} + (1-\delta_k) k  \right\} }_{=:y} \leq 0
\end{aligned}
```

Then, the approximate problem is:


```math
\begin{aligned}
& v(k,h;z) = \max_{k',n} \frac{(c+\epsilon)^{1-\gamma}}{1-\gamma} - \theta \frac{n^{1+\nu}}{1+\nu} + \beta \mathbb{E}\left\{ v(k',h';z') | z \right\} \\
\text{s.t. }& h' = (1-\delta_h) h + \phi(1-n)h \\
& z' \sim \text{MarkovChain}(\mathbf{Z},P_z) \\
& k_{\min} \leq k' \leq k_{\max} \\
& \max\left\{ 0,  1- \frac{1}{\phi}\left[ \frac{h_{\max}}{h} - (1-\delta_h) \right]  \right\} \leq n \leq \min\left\{ 1,  1- \frac{1}{\phi}\left[ \frac{h_{\min}}{h} - (1-\delta_h) \right]  \right\}  \\
& k' - \underbrace{ \left\{ e^z k^\alpha (n h)^{1-\alpha} + (1-\delta_k) k  \right\} }_{=:y} \leq 0
\end{aligned}
```

Meanwhile, we are interested in the following extra statistics:

1. Disposable income $y := e^z k^\alpha (n h)^{1-\alpha} + (1-\delta_k) k$
2. Consumption $c$




## Fitting into the framework

Let's fit the approximated model into the programmatic framework of `LifetimeProblems.jl`. The following table collects all elements that are required to define an `


|Item|Specification|
|----|----------|
|Programming horizon | infinite, transversality condition satisfied |
|Time homogeneity  | time-homogeneous |
|Endo state(s) and grid| $\mathbf{x}:=(k,h)$, dimensionality $D_x=2$; grid space $\mathbf{X}$ |
|Exog state(s) and process | $\mathbf{z}:=(z,)$, $D_z=1$; follows an AR(1) approximated by MarkovChain$(\mathbf{Z},P_{z})$  |
|Control(s)   | $\mathbf{c}:=(k',n)$, $D_c=2$, all controls are continuous   |
|Flow utility | $u(\mathbf{x},\mathbf{z},\mathbf{c}) := \frac{(c+\epsilon)^{1-\gamma}}{1-\gamma} - \theta \frac{n^{1+\nu}}{1+\nu}$ |
|Endo state equation | $f(\mathbf{x},\mathbf{z},\mathbf{c}):= [ k', (1-\delta_h) h + \phi(1-n)h  ]^T$   |
|Lower bound of $\mathbf{c}$ | $`\text{lb}_c(\mathbf{x},\mathbf{z}) := [\dots,\dots]^T`$ (too long, ignored), state-dependent  |
|Upper bound of $\mathbf{c}$ | $`\text{ub}_c(\mathbf{x},\mathbf{z}) := [\dots,\dots]^T`$ (too long, ignored), state-dependent  |
|Generic equation of $\mathbf{c}$ | $`g(\mathbf{x},\mathbf{z},\mathbf{c}) := k' - \underbrace{ \left\{ e^z k^\alpha (n h)^{1-\alpha} + (1-\delta_k) k  \right\} }_{=:y}`$, $D_g=1$  |
|Extra statistics | $s(\mathbf{x},\mathbf{z},\mathbf{c}) := [q,\varphi,w]^T$, $D_s=3$  |
|Discounting | $\beta < 1$ |
|Maximization solver| Brent's method, because: $D_c>1$; there is no generic constraint $D_g=0$ |


Check `main.jl` for the programs and visualization.