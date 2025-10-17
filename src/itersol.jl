#===============================================================================
INTERFACE: ITERATION SOLVERS
===============================================================================#
export iterate!

export AbstractIterationSolver
export IterOptions
export RelaxationIteration
export NewtonIteration


abstract type AbstractIterationSolver <: Any end

# ------------------------------------------------------------------------------
# GENERIC ITERATION OPTIONS & LOGGER
# ------------------------------------------------------------------------------
"""
    IterOptions

Options for iterative solvers.

# Fields

- `maxiter::Int=100`: Maximum number of iterations allowed.
- `tol::Float64=1E-3`: Convergence tolerance for the solver.
- `verbose::Bool=true`: If `true`, print progress information.
- `showevery::Int=1`: Frequency (in iterations) to show progress if `verbose`
  is enabled.
- `parallel::Int=true`: Enable parallel computation (set to `true` or number of
  threads).
- `optimization::Bool=true`: if doing optimization step during VFI.
- `bottomvalue::Float64=-6.66E66`: default value function for infeasible states
   which is used to improve numerical stability.
- `interpmethod::Symbol=:linear`: type of interpolations to use; now supports
  `linear` or `cubic`.
- `pnorm::Real=Inf`: the order of lp norm to aggregate errors; Inf for abs-max
  error, and 2 for least square error.
- `use_current_value_guess::Bool=false`: if to use the value function stackings 
  that are currently stored in a given DP result object. if false, then use
  all-zero initialization
- `progressbar::Bool=true`: if to display a progress bar to `stdout`. may help
  if the optimization takes long time.
"""
Base.@kwdef mutable struct IterOptions

    # ------------------------------
    # generic parameters of value/policy iteration

    maxiter::Int = 100

    tol::Float64 = 1E-3

    verbose  ::Bool = true
    
    showevery::Int  = 1

    parallel ::Bool  = true

    optimization::Bool = true

    bottomvalue::Float64 = -6.66E66

    pnorm::Real = Inf

    use_current_value_guess::Bool = false

    progressbar::Bool = true

    allow_optim_diverge::Bool = true  # if to allow no-convergence on some grid points in the optimization stage


    # ------------------------------
    # generic optimization parameters

    optim_algorithm::Symbol  = :constrainedsimplex # what optimization algorithm to use for the optimization stage
    optim_xtol     ::Float64 = 1E-8  # converge criteria wrt control change (abs tol)
    optim_ftol     ::Float64 = 1E-8  # converge criteria wrt objective change (abs)
    optim_maxiter  ::Int     = 1000  # maximum iterations of optimization algorithms

    # ------------------------------
    # optional parameters for Adaptive Particle Swarm (APS) solver
    aps_nparticle::Int = 3  # number of particles, at least 3

    # ------------------------------
    # optional parameters for constrained Nelder-Mead 
    # `ConstrainedSimplexSearch.jl`
    css_radius ::Float64 = 0.5   # radius (relative distance to boundaries) to initialize a simplex
    css_δ      ::Float64 = 1E-4  # tol for the equality constraint violation
    css_R      ::Float64 = 1.0   # penalty factor for the eq constraint violation
    css_α      ::Float64 = 1.0   # reflection factor, (0,∞)
    css_γ      ::Float64 = 2.0   # expansion factor, (1,∞)
    css_ρout   ::Float64 = 0.5   # outside contraction factor, (0,0.5]
    css_ρin    ::Float64 = 0.5   # inside contraction factor, (0,0.5]
    css_σ      ::Float64 = 0.5   # shrink factor, (0,1)
    css_ftol   ::Float64 = 1E-5  # tol for the func value change at centroids
    css_xtol   ::Float64 = 1E-5  # tol for the max simplex edge length/size


end




# ------------------------------------------------------------------------------
# RELAXATION ITERATIONS: x = F(x)
# ------------------------------------------------------------------------------
"""
    RelaxationIteration(relaxation::Float64 = 1.0)

Representation for iterating over the following non-linear operator with 
relaxation:

`x = F(x)`

where `x` is a real-valued vector and `F` is an operator mapping to its own
space. The relaxation factor controls how radical to update the current guess
using the following updating formula:

`xNew = xOld + relaxation * (xUpdated - xOld)`

or equivalently,

`xNew = (1 - relaxation) * xOld + relaxation * xUpdated`

where `xUpdated` is the implied new values of x's guess. When relaxation is 1,
then the algorithm is the textbook version. The relaxation factor must be >0.

## Hints
- `relaxation = 1`: standard fixed point iteration
- `1 < relaxation < 2`: over relaxation
- `0 < relaxation < 1`: under relaxation
- Gauss-Seidel can be applied in defining `F`, which works with `>1` relaxation
typically
- Try `relaxation < 1` when the system is ill-conditioned and slow convergent.

## Example

```julia
import LifetimeProblems as ltp

F(x) = [.5 .2; .1 .4] * x .+ [1,0]

its = ltp.RelaxationIteration(relaxation = 1.05)

# usage: returns a copy
x0 = rand(2)
for i in 1:100
    x0 = iterate(F, x0, its)
    println(i, ": ", x0)
end

# usage: inplace updates
x0 = rand(2)
for i in 1:100
    ltp.iterate!(F, x0, its)
    println(i, ": ", x0)
end
```
"""
struct RelaxationIteration <: AbstractIterationSolver
    relaxation::Float64
    function RelaxationIteration(;relaxation::Real = 1.0)
        @assert relaxation > 0 "relaxation > 0 required but got $relaxation"
        if relaxation >= 2.0
            @warn "relaxation = $relaxation > 2, too radical; be careful"
        end
        new(Float64(relaxation))
    end
end
function Base.show(io::IO, its::RelaxationIteration)
    @printf(
        io, 
        "Relaxation Iteration (factor = %.3f), %s\n",
        its.relaxation,
        if its.relaxation == 1
            "Fixed point iteration"
        elseif its.relaxation > 1
            "Over relaxation"
        else
            "Under relaxation"
        end
    )
    return nothing
end
# ------------------------------------------------------------------------------
"""
    Base.iterate(
        F  ::Function, 
        x  ::AbstractVector{<:Real},
        its::RelaxationIteration
    )

Iterate system `x = F(x)` once, where `F(x)` returns a vector of the same length
as `x`. Returns a new guess vector of `x`.
"""
function Base.iterate(
    F  ::Function, 
    x  ::AbstractVector{<:Real},
    its::RelaxationIteration
)
    return x .+ its.relaxation .* (F(x) .- x)
end
function iterate!(
    F  ::Function, 
    x  ::AbstractVector{<:Real},
    its::RelaxationIteration
)
    x .+= its.relaxation .* (F(x) .- x)
    return nothing
end






# ------------------------------------------------------------------------------
# NEWTON ITERATOR: F(x) = 0
# ------------------------------------------------------------------------------
"""
    NewtonIteration

Representation for iterating over the following non-linear operator:

`0 = F(x)`

where `x` is a real-valued vector and `F` is an operator mapping to a vector
space with the same length as `x` ("residuals"). The updating formula is:

`xNew = xOld - [∇F(xOld)]^{-1} * F(xOld)`

where `∇` is Jacobian/gradient matrix. This iteration representation is textbook
and no preconditioning applied.

## Hints
- Analytical Jacobian are preferred where `J[i,j]` represents `∂F[i]/∂x[j]`.
- The convergence is quadratically local.

## Example

```julia
import LifetimeProblems as ltp

F(x) = [-.5 .2; .1 -.4] * x .+ [1,0]
J(x) = [-.5 .2; .1 -.4]

its = ltp.NewtonIteration()

# usage: returns a copy
x0 = rand(2)
for i in 1:100
    x0 = iterate(F, J, x0, its)
    println(i, ": ", x0)
end

# usage: inplace updates
x0 = rand(2)
for i in 1:100
    ltp.iterate!(F, J, x0, its)
    println(i, ": ", x0)
end
```
"""
struct NewtonIteration <: AbstractIterationSolver
end
function Base.show(io::IO, its::NewtonIteration)
    println(io, "Newton Iteration")
    return nothing
end
# ------------------------------------------------------------------------------
"""
    Base.iterate(
        F  ::Function, 
        J  ::Function,
        x  ::AbstractVector{<:Real},
        its::NewtonIteration
    )

Iterate the guess of system `0 = F(x)` once and returns a new guess vector using
Newton's method, where `F(x)` returns a vector of the same length as `x`, and
`J(x)` returns a squared matrix of Jacobian matrix.
"""
function Base.iterate(
    F  ::Function,
    J  ::Function,
    x  ::AbstractVector{<:Real},
    its::NewtonIteration
)
    return x .- J(x) \ F(x)
end
function iterate!(
    F  ::Function, 
    J  ::Function,
    x  ::AbstractVector{<:Real},
    its::NewtonIteration
)
    x .-= J(x) \ F(x)
    return nothing
end




