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
- `errortrace::Vector{Float64}=Float64[]`: Stores the error at each iteration.
"""
Base.@kwdef mutable struct IterOptions

    maxiter::Int = 100

    tol::Float64 = 1E-3

    verbose  ::Bool = true
    showevery::Int  = 1

    parallel ::Int  = true

    errortrace::Vector{Float64} = Float64[]

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




