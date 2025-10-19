# Example: Endogenous labor supply problem

## Model setup

Let's consider a standard household's problem with endogenous labor supply:


```math
\begin{aligned}
& v(a;z) = \max_{c,n,a'} \frac{c^{1-\gamma}}{1-\gamma} - \alpha \frac{n^{1+\nu}}{1+\nu} + \beta \mathbb{E}\{ v(a';z') | z \} \\
\text{s.t. }& a' = (1+r)a + e^z n  - c \\
& z' \sim \text{AR}(1)(\rho,\sigma,\overline{z}) \\
& 0 \leq c < \infty \\
& 0 \leq n \leq 1 \\
& 0 \leq a' < \infty
\end{aligned}
```

**Parameterization and transformation**:

- Discretize $z$'s process with a finite-state Markov chain $(\mathbf{Z},P_z)$
- Add a small positive $\epsilon$ to the consumption bundle in flow utility
- Use a finite discrete grid of $a$ to bound $a_{\min}\leq a\leq a_{\max}$
- Replace control $c$ with $a'$ to obtain box constraints, then, further shrink the admissible space of $a'$ by substituting the non-negative consumption constraint into it:

```math
\begin{aligned}
& c = (1+r)a + e^z n - a' \geq 0 \\
\implies& a' \leq (1+r) a + e^z n \\
\overset{0\leq n\leq 1}{\implies}& a' \leq \min\{a_{\max},(1+r)a + e^{z} \}
\end{aligned}
```

However, one should notice that $`\min\{a_{\max},(1+r)a + e^{z} \}`$ is a supremum of the admissible set. The true admissible upper bound of $a'$ is $`\min\{a_{\max},(1+r)a + e^{z} n\}`$ which depends on $n$ simultaneously. Thus, we introduce the following generic constraint:

```math
g(\mathbf{x},\mathbf{z},\mathbf{c}) := a' - [(1+r) a + e^z n] \leq 0
```

> **Remark**:
> In the above transformation, we plugged the box constraint of consumption $c$ to the box constraint of bond holding $a'$. Why bothering to do this?
> 
> **Answer**: We want to remove the equality constraint (budget constraint) by removing one control variable (consumption). This is required by the package, but also beneficial as less control variables would improve numerical stability of algorithms. When we do this, we should avoid just dropping related constraints but should "move" its information to box constraints if possible. Box constraints can effectively shrink the feassible inadmissible space. It is fine, in some cases, to keep the box constraint of $a'$ as $[a_{\min},a_{\max}]$ without modification/shrinkage as above. However, such box constraints varies by the speicfied computation domain, which is not plausible and easy to be numerical instable: an "inelastic" box constraint introduces large feasible but inadmissible regions of searching, which often causes numerical issues.


Then, the apprixmated problem is:

```math
\begin{aligned}
& v(a;z) = \max_{a',n} \frac{(c+\epsilon)^{1-\gamma}}{1-\gamma} - \alpha \frac{n^{1+\nu}}{1+\nu} + \beta \mathbb{E}\{ v(a';z') | z \} \\
\text{s.t. }& a' = (1+r)a + e^z n  - c \\
& z' \sim \text{MarkovChain}(\mathbf{Z}, P_z) \\
& a_{\min} \leq a' \leq \min\{a_{\max},(1+r)a + e^{z} \} \\
& 0 \leq n \leq 1 
\end{aligned}
```

Meanwhile, we are interested in the following statistics:
- disposable income $y:=(1+r)a + e^z n$
- consumption $c$ (it becomes a statistics as we substituted it out)
- saving rate $1-c/y$



## Fitting into the framework

Let's fit the approximated model into the programmatic framework of `LifetimeProblems.jl`. The following table collects all elements that are required to define an `


|Item|Specification|
|----|----------|
|Programming horizon | infinite, transversality condition satisfied |
|Time homogeneity  | time-homogeneous |
|Endo state(s) and grid| $\mathbf{x}:=(a,)$, dimensionality $D_x=1$; grid space $\mathbf{X}$ |
|Exog state(s) and process | $\mathbf{z}:=(z,)$, $D_z=1$, but have $N_z>1$; follows an AR(1) approximated by MarkovChain$(\mathbf{Z},P_z)$  |
|Control(s)   | $\mathbf{c}:=(c,n)$, $D_c=2$, all controls are continuous   |
|Flow utility | $u(\mathbf{x},\mathbf{z},\mathbf{c}) := \frac{(c+\epsilon)^{1-\gamma}}{1-\gamma}- \alpha \frac{n^{1+\nu}}{1+\nu}$ |
|Endo state equation | $f(\mathbf{x},\mathbf{z},\mathbf{c}):= a'$, direct choosing   |
|Lower bound of $\mathbf{c}$ | $`\text{lb}_c(\mathbf{x},\mathbf{z}) := [a_{\min},0]^T`$, state-dependent  |
|Upper bound of $\mathbf{c}$ | $`\text{ub}_c(\mathbf{x},\mathbf{z}) := [ \min\{a_{\max},(1+r)a + e^{z} \} ,1]^T`$, state-dependent  |
|Generic equation of $\mathbf{c}$ | $g(\mathbf{x},\mathbf{z},\mathbf{c}) := a' - [(1+r) a + e^z n] \leq 0$, $D_g=1$  |
|Extra statistics | $s(\mathbf{x},\mathbf{z},\mathbf{c}) := [y,c,1-c/y]^T$, $D_s=3$  |
|Discounting | $\beta < 1$ |
|Maximization solver| Constrianed Nelder-Mead (simplex) method or Interior point Newton, or SQP, because: $D_c>1$; there is generic constraint $D_g>0$; kinks expected to exist because of $n\in[0,1]$ |


Check `main.jl` for the programs and visualization.






















