# LifetimeProblems.jl
Modeling and solving lifetime dynamic problems in quantitative economics.

<font color=red>THIS REPO IS IN DEV, NOT READY FOR USE YET</font>


## Math formulation


### Infinite horizon problem

For inifnite horizon problems, this package particularly focus on _time-homogenous_ problems:

$$
\begin{aligned}
& v(x,z) = \max_{c} u(x,z,c) + \beta \mathbb{E}\{ v(x',z') | z \} \\
\text{s.t. } & x' = f(x,z,c) \\
    & z' \sim \text{MarkovChain}(\mathbf{Z},\mathbf{P}_z) \\
    & \text{lb}_c (x,z) \leq c \leq \text{ub}_c (x,z) \\
    & g(x,z,c) \leq 0 
\end{aligned}
$$

where endogenous state $x$ is supposed to be all continuous; upper letters in bold fonts (e.g. $\mathbf{X}$) denote grid spaces:

$$
\mathbf{X} := [x^1, x^2, \dots]
$$

Marginal grids are restricted to be joinned with tensor/Cartesian product. To correctly specify a lifetime problem, I list all dimensionlaities below:

$$
\begin{aligned}
& x \in \mathbf{X} \subset \mathbb{R}^{D_x}, z \in \mathbf{Z} \subset \mathbb{R}^{D_z}, c \in \mathbb{R}^{D_c} \\
& \text{lb}_c\mapsto\mathbb{R}^{D_c}, \text{ub}_c\mapsto\mathbb{R}^{D_c}\\
& f\mapsto \mathbb{R}^{D_x}, g\mapsto\mathbb{R}^{D_g} \\
& \mathbf{Z} \in \mathbb{R}^{N_z}, \mathbf{P}_z \in \mathbb{R}^{N_z \times N_z}  \\
& \mathbf{X} \in \mathbb{R}^{N_x \times D_x}
\end{aligned}
$$

Meanwhile, there might be some statistics $s = s(x,z,c) \in\mathbb{R}^{D_p}$ of economist's interests such as some model moments (e.g. saving rate).
The problem can be summarized by the following data package:

$$
\text{DP}^{\infty}_{x,z,c} := \{ \mathbf{X}; \text{MarkovChain}(\mathbf{Z},\mathbf{P}_z);  (u,f,\text{lb}_c,\text{ub}_c,g,s) ; \beta \}
$$

Depending on restrictions on $c$, some extra elements are also required (e.g. flags indicating if a specific dimenison of $c$ is discrete choice). Particularly, for control variables of discrete chocie problems, the set of choice should be provided as well.


> **Note**: the lower and upper bounds of $c$ are allowed to depend on $x$ and $z$, which gives necessary flexibilities in managing optimizations.










## Some technical notes

- Value function interpolation is restricted to (multi-)linear interpolation. There are three reasons: first, linear interpolation does not overshoot which preserves essential concavity in some applications; second, linear interpolation is cheap to construct and evaluate given the fact that an interpolant would be called for billions of times; third, the package uses a trick that interpolating the expected value function once then evaluating it everywhere, in which linearity is required to ensure equivalence between interpolating the expected value function and taking expectation over the interpolated value function at different points. Please check [my post](https://clpr.github.io/posts/005_averagelinearinterp/) for proof.

- The package intends to disable equality constraints because: first, it is almost always possible to substitute an equality constraint out by defining an extra auxiliary variable; second, many numerical solvers do not work with equality constraints very well as an equality constraint basically is a "line" in a state space.