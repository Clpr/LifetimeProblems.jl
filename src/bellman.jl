#===============================================================================
BELLMAN EQUATION FORMULATION

--------------------
## Generic model (discrete time):

v(x,z) = max_(c ∈ C) u(x,z,c) + β * 𝔼{v(x',z') | z}
s.t.
x' = f(x,z,c)
z' ~ MarkovChain
g(x,z,c) ≤ 0
lbc <= c <= ubc

where x ∈ X, z ∈ Z, c ∈ C; (X,Z,C) are all box/rectangular spaces. All equality
constraints are supposed to be substituted out or cancelled out by defining 
extra control variables.

Variants:
- Horizon    : finite or infinite?
- Uncertainty: deterministic or stochastic?
- Continuity : continuous controls, discrete choice, or mixed?

--------------------
## Notes
- Horizon deserves a single struct as the data structures (scalars vs series)
  differs much.
- Discrete time model for now; for continuous time model, will only consider
  discretized approximation (discount over Δt > 0 time period)
- Only consider all-continuous control variables for now. discrete choice model
  will be added later
===============================================================================#
export InfiniteHorizonDP


abstract type DynamicProgramming{DX,DZ,DC} <: Any end



# ------------------------------------------------------------------------------
# Interface
# ------------------------------------------------------------------------------
# TODO





# ------------------------------------------------------------------------------
# Infinite horizon DP
# ------------------------------------------------------------------------------
"""
    InfiniteHorizonDP(
        xgrid::Union{bdm.TensorDomain,bdm.CustomTensorDomain},
        ccont::AbstractVector{Bool},
        cbox ::bdm.AbstractBoxDomain{DC},
        u    ::Function,
        f    ::Function,
        g    ::Function ;

        zproc  ::Union{Nothing,mmc.MultivariateMarkovChain} = nothing,
        β      ::Float64 = 0.95,
    )

Formulation of infinite horizon dynamic programming, fromulated as a Bellman 
equation. Type parameters: `DX` is dimensionality of endogenous states, `DZ` is
dimensionality of exogenous uncertainties/shocks, and `DC` is dimensionality of
control variables.


## Arguments

- `xgrid`: tensor/Cartesian-joinned grid space for endogenous states `x`
- `ccont`: a vector indicating if a control variable is continuous (or discrete)
- `cbox`: a (grid or non-grid) space for control variables `c`. Only lower bound
and upper bound matter for continuous control variables; for the case in which
there is any `false` in `ccont` (discrete control), a gridded space object, e.g.
`TensorDomain` or `CustomTensorDomain` is required, while the number of grid
points specified in the gridded space object is ignored for those controls that
are claimed to be continuous.
- `u`: function of flow utility, should have a method that can take only three
positional arguments `(x,z,c)` where each of the three arguments is an static
vector of `Float64`. The `SVector{D,Float64}` type is from `StaticArrays.jl`.
And, empty static vectors should be passed if there is no uncertainty (no `z`).
- `f`: function of endogenous state transition `x_{t+1} = f(x_t,z_t,c_t)`. Has
the same requirement as `u`. If a `(x,z,c)` is infeasible or inadmissible, then
just return something (like NaN or Inf) but don't throw an error which will kill
the solver of `ConstrainedSimplexSearch.jl`. These returned values would never
be really used. The same rule applies to `g` and `u`
- `g`: function of inequality constraints `g(x,z,c)<=0` where `g` should return
a real vector of the left hand side of the inequality. The number of inequality
constraints can be zero or any positive number. 
- `zproc`: if this is a stochastic dynamic programming. pass `nothing` if this
is a deterministic dynamic programming. Otherwise, pass a multi-dmiensional
Markov chain defined by `MultivariateMarkovChains.jl`.
- `β`: discounting factor over time. Typically between 0 and 1 to avoid Ponzi's
scheme (divergent equilibrium). Can be greater than or equal to 1 if you really
know what you are doing.


## Math

```
v(x,z) = max_(c ∈ C) u(x,z,c) + β * 𝔼{v(x',z') | z}
s.t.
x' = f(x,z,c)
z' ~ MarkovChain
g(x,z,c) ≤ 0
```

where x ∈ X, z ∈ Z, c ∈ C; (X,Z,C) are all box/rectangular spaces. All equality
constraints are supposed to be substituted out or cancelled out by defining 
extra control variables. The problem is supposed to be **time-homogeneous**.


## Usage
- The functions `u`, `f` and `g` should all take 3 `(x,z,c)` as positional args.
The three functions should be able to take each of `(x,z,c)` as a
`SVector{D,Float64}` provided by `StaticArrays.jl`, where `D` is vector length. 
If any of the 3 vectors is unused, then the routines, internally, will pass an 
empty `SVector{0,Float64}()` to `u`, `f` and/or `g` as the corresponding arg. 
Users should take care of this default situation and make the user-supplied 
funcs compatible.
- A hint: even though static vectors are used, `u`, `f`, and `g` can still be
defined in a more broad way that takes e.g. `AbstractVector{Float64}`.
- If deterministic, then `u`, `f` and `g` should still take `z` as a positional
argument. The functions should be able to handle an empty `z` vector as input.



## Example

```julia
import LifetimeProblems as ltp

# define: inf-horizon time-homogeneous deterministic problem
#         all controls are continuous
dp = ltp.InfiniteHorizonDP(
    ltp.bdm.TensorDomain([2,3,4]),  # 3 endogenous state variables
                 
    [true, true],                   # 2 continuous controls
                 
    ltp.bdm.BoxDomain(              # box constraints for the control variables
        [-1.0, -1.0],               # lb; required for all controls
        [ 1.0,  1.0],               # ub; required for all controls
    ),                              # only lb,ub matters for continuous variable
                                    # so BoxDomain is enough
                
    (x,z,c) -> log.(c) |> sum,      # additive log utility
                 
    (x,z,c) -> x,                   # statioanry
                 
    (x,z,c) -> .-c,                 # non-negativity constraints
                 
    zproc   = nothing               # deterministic, no uncertainty
) ::ltp.InfiniteHorizonDP{3,0,2}


# define: inf-horizon time-homogeneous stochastic (Markovian) problem
#         all controls are continuous
dp = ltp.InfiniteHorizonDP(
    ltp.bdm.TensorDomain([2,3,4]),        # 3 endogenous state variables
    
    [true, true],                         # 1 continuous, 1 discrete control
    
    ltp.bdm.TensorDomain(                 # box constraints for the controls
        [-1.0, -1.0],                     # lb; required for all controls
        [ 1.0,  1.0],                     # ub; required for all controls
        [   1,    5]                      # #grid points; because of continuous
    ),                                    # controls, so arbitrary #points work
    
    (x,z,c) -> log.(c) |> sum,            # additive log utility
    
    (x,z,c) -> x,                         # statioanry
    
    (x,z,c) -> .-c,                       # non-negativity constraints
    
    zproc   = ltp.mmc.MultivariateMarkovChain(
        [
            [0.8, 0.5, -1.0,  9.0],
            [1.0, 1.0,  0.0, 10.0],
            [1.2, 1.5,  1.0, 11.0],
            [1.4, 2.0,  2.0, 13.0],
            [1.6, 2.5,  3.0, 14.0],
        ],               # states
        rand(5,5),       # transition probability matrix
        normalize = true # force normalizing to row-sum = 1
    ) # 4 shock variables (exogenous states), swtiching between 5 (multi-)states
) ::ltp.InfiniteHorizonDP{3,4,2}


# define: inf-horizon time-homogeneous stochastic (Markovian) problem
#         some controls are discrete (mixed)
dp = ltp.InfiniteHorizonDP(
    ltp.bdm.TensorDomain([2,3,4]),        # 3 endogenous state variables
    
    [true, false],                        # 1 continuous, 1 discrete control
    
    ltp.bdm.TensorDomain(                 # box constraints for the controls
        [-1.0, -1.0],                     # lb; required for all controls
        [ 1.0,  1.0],                     # ub; required for all controls
        [   1,    5]                      # #grid points; only discrete controls
    ),                                    # really need it
    
    (x,z,c) -> log.(c) |> sum,            # additive log utility
    
    (x,z,c) -> x,                         # statioanry
    
    (x,z,c) -> .-c,                       # non-negativity constraints
    
    zproc   = ltp.mmc.MultivariateMarkovChain(
        [
            [0.8, 0.5, -1.0,  9.0],
            [1.0, 1.0,  0.0, 10.0],
            [1.2, 1.5,  1.0, 11.0],
            [1.4, 2.0,  2.0, 13.0],
            [1.6, 2.5,  3.0, 14.0],
        ],               # states
        rand(5,5),       # transition probability matrix
        normalize = true # force normalizing to row-sum = 1
    ) # 4 shock variables (exogenous states), swtiching between 5 (multi-)states
) ::ltp.InfiniteHorizonDP{3,4,2}

```
"""
mutable struct InfiniteHorizonDP{DX,DZ,DC} <: DynamicProgramming{DX,DZ,DC}

    xgrid::Union{bdm.TensorDomain{DX},bdm.CustomTensorDomain{DX}}

    zproc::Union{Nothing,mmc.MultivariateMarkovChain{DZ}}

    ccont::sa.SVector{DC,Bool}

    cbox ::bdm.AbstractBoxDomain{DC}

    u::Function # u(x,z,c): R^{DX*DZ*DC} -> R

    f::Function # x' = f(x,z,c): R^{DX*DZ*DC} -> R^{DX}

    g::Function # g(x,z,c) ≤ 0 : R^{DX*DZ*DC} -> R^{M}

    β::Float64  # utility discounting factor

    function InfiniteHorizonDP(
        xgrid::Union{bdm.TensorDomain,bdm.CustomTensorDomain},
        ccont::AbstractVector{Bool},
        cbox ::bdm.AbstractBoxDomain,
        u    ::Function,
        f    ::Function,
        g    ::Function ;

        zproc  ::Union{Nothing,mmc.MultivariateMarkovChain} = nothing,
        β      ::Float64 = 0.95,
    )
        dx = ndims(xgrid)
        dz = isnothing(zproc) ? 0 : ndims(zproc)
        dc = length(ccont)

        (dx > 0) || @warn("zero endogenous state variables x defined")
        (dc > 0) || @warn("zero control variables c defined")
        @assert β >= 0 "negative discounting factor found: $β"
        (β > 1)  && @warn("β > 1, the problem may be diverging")
        @assert ndims(cbox) == length(ccont) "cbox & ccont are inconsistent"
        if (!all(ccont)) & isa(cbox,bdm.BoxDomain)
            error("discrete control claimed but a non-grid BoxDomain provided")
        end

        new{dx,dz,dc}(
            xgrid,
            zproc,
            sa.SVector{dc,Bool}(ccont),
            cbox,
            u, f, g,
            β
        )
    end # constructor
end # InfiniteHorizonDP{D}
function Base.show(io::IO, dp::InfiniteHorizonDP{DX,DZ,DC}) where {DX,DZ,DC}
    nx = dp.xgrid |> size |> prod
    nz = isnothing(dp.zproc) ? 0 : length(dp.zproc)

    println(
        io, 
        "Infinite-horizon Dynamic Programming (#x = $DX, #z = $DZ, #c = $DC)"
    )
    println(io, "- uncertainty   : ", typeof(dp.zproc))
    println(io, "- size(x nodes) : ", size(dp.xgrid), ", total = ", nx)
    println(io, "- size(z states): ", nz)
    println(io, "- discounting   : ", dp.β)
    return nothing
end # show




# ------------------------------------------------------------------------------
# Finite horizon DP
# ------------------------------------------------------------------------------
# TODO


