# LifetimeProblems.jl
Modeling and solving lifetime dynamic problems in quantitative economics. This package aims to provide a structural framework for solving lifetime problems in dynamic economic models, in which the lifetime problems are part of the whole system but usually takes most of time to program.

THIS REPO IS IN DEV, NOT READY FOR USE YET


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



### Finite horizon problem

TODO




---

## Optimization stage: available algorithms

### All-continuous control variables

TODO


### All-discrete control variables


TODO


### Mixed continuity control variables

When both continuous and discrete control variables are present, the optimization problem becomes a **nonlinear mixed-integer program (MINLP)**, as selecting from an ordered discrete set is equivalent to choosing its integer index. Such problems are well known to be NP-hard, and they become especially challenging when economists seek a global optimum. In economic modeling, mixed-integer structures commonly appear in contexts such as portfolio selection with discrete housing choices, option exercise decisions, and matching problems.

Solving such an MINLP in this package follows the idea of **complete (outer) enumeration** of the discrete control set.  All control variables are divided into two categories: discrete and continuous. Users must provide a tensor grid for the discrete control dimensions.  The solver then solves a series of conditional optimization problems (“child problems”) at grid points of discrete controls, each treating all the left controls as continuous. This approach converts the NP-hard problem into multiple standard optimization problems. Finally, the solver selects the best child problem as the overall solution to the MINLP.







---

## Usage & examples


TODO




---

## Some technical notes

- **Value function interpolation** is limited to (multi-)linear interpolation for three reasons:  
  1. Linear interpolation avoids overshooting, preserving essential concavity in some applications.  
  2. It is computationally efficient to build and evaluate, which matters when the interpolant is called billions of times.  
  3. The package uses a numerical trick that interpolates the *expected* value function once and evaluates it everywhere. Linearity is required to ensure equivalence between interpolating the expected value function and taking expectations over the interpolated value function. See [this post](https://clpr.github.io/posts/005_averagelinearinterp/) for a detailed explanation.

- **Equality constraints** are intentionally disabled because:  
  1. They can almost always be removed by space transformation or introducing auxiliary variables.  
  2. Many numerical solvers handle them poorly since an equality constraint effectively defines a “line” in the state space.

- **Multiple value functions** (e.g., regime-switching or multi-stage decision models) will not be supported by this package. Such models are typically ad hoc and rely on specific structural assumptions that fall outside the intended design of this package.

- **Non-tensor grids** for discrete controls are under consideration and may be supported in future updates. This setup can significantly improve efficiency by skipping inadmissible points. Such cases arise, for example, when discrete controls are subject to constraints like a household portfolio leverage ratio. Economists can pre-filter some portfolios that are obviously inadmissible.

- For optimization stages involving only discrete controls, the package plans to add support for a branch-and-bound (B&B) method in future releases, complementing the current naive grid search solver.