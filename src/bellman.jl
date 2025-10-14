#===============================================================================
BELLMAN EQUATION FORMULATION

--------------------
## Generic model (discrete time):

v(x,z) = max_(c ∈ C) u(x,z,c) + β * 𝔼{v(x',z') | z}
s.t.
x' = f(x,z,c)
z' ~ MarkovChain
g(x,z,c) ≤ 0

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






# ------------------------------------------------------------------------------
# Infinite horizon DP
# ------------------------------------------------------------------------------
"""
    InfiniteHorizonDP(
        xgrid::Union{bdm.TensorDomain,bdm.CustomTensorDomain},
        ccont::AbstractVector,
        u    ::Function,
        f    ::Function,
        g    ::Function ;

        horizon::Horizon = InfiniteHorizon(),
        zproc  ::Union{Nothing,mmc.MultivariateMarkovChain} = nothing
    )

Formulation of infinite horizon dynamic programming, fromulated as a Bellman 
equation. Type parameters: `DX` is dimensionality of endogenous states, `DZ` is
dimensionality of exogenous uncertainties/shocks, and `DC` is dimensionality of
control variables.

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


## Example

```julia
import LifetimeProblems as ltp

# define: inf-horizon time-homogeneous deterministic problem
dp = ltp.InfiniteHorizonDP(
    ltp.bdm.TensorDomain([2,3,4]),        # 3 endogenous state variables
    [ltp.Continuous(), ltp.Continuous()], # 2 continuous controls
    (x,z,c) -> log.(c) |> sum,            # additive log utility
    (x,z,c) -> x,                         # statioanry
    (x,z,c) -> .-c,                       # non-negativity constraints
    zproc   = nothing                     # deterministic, no uncertainty
) ::ltp.InfiniteHorizonDP{3,0,2}

# define: inf-horizon time-homogeneous stochastic (Markovian) problem
dp = ltp.InfiniteHorizonDP(
    ltp.bdm.TensorDomain([2,3,4]),        # 3 endogenous state variables
    [ltp.Continuous(), ltp.Continuous()], # 2 continuous controls
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

    ccont::sa.SVector{DC,Continuity}

    u::Function # u(x,z,c): R^{DX*DZ*DC} -> R

    f::Function # x' = f(x,z,c): R^{DX*DZ*DC} -> R^{DX}

    g::Function # g(x,z,c) ≤ 0 : R^{DX*DZ*DC} -> R^{M}

    function InfiniteHorizonDP(
        xgrid::Union{bdm.TensorDomain,bdm.CustomTensorDomain},
        ccont::AbstractVector,
        u    ::Function,
        f    ::Function,
        g    ::Function ;

        zproc  ::Union{Nothing,mmc.MultivariateMarkovChain} = nothing
    )
        dx = ndims(xgrid)
        dz = isnothing(zproc) ? 0 : ndims(zproc)
        dc = length(ccont)

        (dx > 0) || @warn("zero endogenous state variables x defined")
        (dc > 0) || @warn("zero control variables c defined")

        new{dx,dz,dc}(
            xgrid,
            zproc,
            sa.SVector{dc,Continuity}(ccont),
            u, f, g
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
    return nothing
end # show




# ------------------------------------------------------------------------------
# Finite horizon DP
# ------------------------------------------------------------------------------
"""


"""
# TODO



