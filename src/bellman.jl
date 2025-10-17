#===============================================================================
BELLMAN EQUATION FORMULATION
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
    InfiniteHorizonDP

Formulation of a time-homogenous infinite-horizon dynamic programming as Bellman
equation.
"""
mutable struct InfiniteHorizonDP{DX,DZ,DC,DG,DS} <: DynamicProgramming{DX,DZ,DC}

    xgrid::Union{bdm.TensorDomain{DX},bdm.CustomTensorDomain{DX}}

    zproc::Union{Nothing,mmc.MultivariateMarkovChain{DZ}}

    u  ::Function # u(x,z,c): R^{DX*DZ*DC} -> R
    f  ::Function # x' = f(x,z,c): R^{DX*DZ*DC} -> R^{DX}
    lbc::Function # lbc(x,z), lower bound of control variables, state-depend
    ubc::Function # ubc(x,z), upper bound of control variables, state-depend
    g  ::Function # g(x,z,c) ≤ 0 : R^{DX*DZ*DC} -> R^{DG}
    s  ::Function # s(x,z,c), extra statisticss

    β::Float64  # utility discounting factor

    ccont::sa.SVector{DC,Bool}
    cdisc::Union{Nothing,bdm.TensorDomain{DC},bdm.CustomTensorDomain{DC}}

end # InfiniteHorizonDP{D}
# ------------------------------------------------------------------------------
function Base.show(
    io::IO, 
    dp::InfiniteHorizonDP{DX,DZ,DC,DG,DS}
) where {DX,DZ,DC,DG,DS}
    nx = dp.xgrid |> size |> prod
    nz = isnothing(dp.zproc) ? 0 : length(dp.zproc)

    println(
        io, 
        "Time-homogenous infinite-horizon lifetime problem"
    )
    println(io, "- #endo states  : ", DX)
    println(io, "- #exog states  : ", DZ)
    println(io, "- #controls     : ", DC)
    println(io, "- #constraints  : ", DG)
    println(io, "- #statistics   : ", DS)
    println(io, "- uncertainty   : ", typeof(dp.zproc))
    println(io, "- size(x nodes) : ", size(dp.xgrid), ", total = ", nx)
    println(io, "- size(z states): ", nz)
    println(io, "- discounting   : ", dp.β)
    return nothing
end # show
# ------------------------------------------------------------------------------
"""
    InfiniteHorizonDP{DC,DG,DS}(
        xgrid::Union{bdm.TensorDomain,bdm.CustomTensorDomain},
        zproc::Union{Nothing,mmc.MultivariateMarkovChain},

        u  ::Function,
        f  ::Function,
        lbc::Function,
        ubc::Function,
        g  ::Function,
        s  ::Function;

        β    ::Float64 = 0.95,
        ccont::sa.SVector{DC,Bool} = sa.SVector{DC,Bool}(fill(true,dc)),
        cdisc::Union{Nothing,bdm.TensorDomain{DC},bdm.CustomTensorDomain{DC}} = nothing
    ) where {DC,DG,DS}

Define an infinite-horizon life time problem.

## Example: neoclassical growth model
```julia

# ------------------------------------------------------------------------------
# v(k,z) = max_c c^(1-μ)/(1-μ) + βE{v(kp,z')|z}
# kp = f(k,z,c) := exp(z) * k^α + (1-δ)*k - c
# z' ~ AR(1), i.e. z' = ρ * z + (1-ρ) * zss + σ * ϵ, ϵ ~ N(0,1)
# kmin <= kp <= kmax, c >= 0
# ------------------------------------------------------------------------------
# dimensionalities:
# - DX = 1, (capital)
# - DZ = 1, (technology shock)
# - DC = 1, (consumption)
# - DG = 1, (non-negative consumption constraint)
# - DS = 2, (income y, and saving rate 1-c/y)
# ------------------------------------------------------------------------------

# parameters
par = Dict(
    :α   => 0.33,  # capital share
    :β   => 0.99,  # discounting
    :γ   => 2.0,   # risk aversion
    :δ   => 0.025, # depreciation

    :ρz  => 0.95,  # AR1 coefficient of z
    :σz  => 0.007, # volatility of z
    :zss => 1.0,   # long-term mean of z
)

# ------------------------------------------------------------------------------
# deterministic steady state of capital
kss = let
    _nomi = par[:α] * par[:zss]
    _deno = 1/par[:β] - (1 - par[:δ])
    (_nomi / _deno)^(1/(1-par[:α]))
end

# ------------------------------------------------------------------------------
# discretize z's process to a Markov chain, using Tachen
mc_z = ltp.mmc.tauchen(
    ltp.mmc.AR1(ρ = par[:ρz], σ = par[:σz], xavg = par[:zss] ),
    5,      # number of states
    nσ = 2, # μ±2σ
)

# ------------------------------------------------------------------------------
# computation domains & grid
xgrid = ltp.bdm.TensorDomain(
    [kss * 0.01,],
    [kss * 2.00,],
    [50,],
)

# ------------------------------------------------------------------------------
# helper: disposable income
function get_y(x,z,c,par)::Real
    return exp(z[1]) * x[1] ^ par[:α] + (1 - par[:δ]) * x[1]
end


# ------------------------------------------------------------------------------
# box constraints of the control (kp, the new capital level)
# hint: bounds consumption using the maximum possible net income, which improves
#       the efficiency of optimization greatly.
# DC = 1
lbc(x,z,c ; par = par, kmax = xgrid[1][end]) = begin

    y = get_y(x,z,c,par)

    # derived from: kp = y - c <= kmax --> c >= max{y - kmax, 0}

    return [
        max(y - kmax, 0.0)
    ]
end
ubc(x,z,c ; par = par, kmin = xgrid[1][1]) = begin

    y::Real  = exp(z[1]) * x[1]^par[:α] + (1 - par[:δ]) * x[1]
    
    # derived from: kp = y - c >= kmin --> c <= y - kmin
    @assert (y - kmin) >= 0 "infeasible income for consumption"

    return [
        y - kmin
    ]
end

# ------------------------------------------------------------------------------
# define: flow utility, CRRA
u(x,z,c ; par = par) = begin
    y = get_y(x,z,c,par)

    return if c[1] < 0
        # if infeasible, return something without throwing an error
        -114514.0
    else
        # avoid numerical error at exact 0
        (c[1] + eps())^(1-γ)/(1-γ)
    end
end

# ------------------------------------------------------------------------------
# define: state equation of x, the budget constraint
# DX = 1
f(x,z,c ; par = par) = begin
    y  = get_y(x,z,c,par)
    kp = y - c[1]
    return [
        kp
    ]
end

# ------------------------------------------------------------------------------
# define: control's generic constraints, the non-negativity constraint, c >= 0
# hint: neutralized in this example by specifying `lbc` function
# DG = 1
g(x,z,c) = begin
    return [
        -c[1]
    ]
end

# ------------------------------------------------------------------------------
# define: extra statistics, disposable income y and saving rate 1-c/y
# DS = 2
s(x,z,c; par = par) = begin
    y = get_y(x,z,c,par)
    srate = 1 - c[1]/y
    return [
        y,
        srate,
    ]
end


# ------------------------------------------------------------------------------
# define: dynamic programming problem
dp = ltp.InfiniteHorizonDP{1,1,1}(
    xgrid,
    mc_z,

    u, f, lbc, ubc, g, s,
    
    β = par[:β]
)
```
"""
function InfiniteHorizonDP{DC,DG,DS}(
    xgrid::Union{bdm.TensorDomain,bdm.CustomTensorDomain},
    zproc::Union{Nothing,mmc.MultivariateMarkovChain},

    u  ::Function,
    f  ::Function,
    lbc::Function,
    ubc::Function,
    g  ::Function,
    s  ::Function;

    β    ::Float64 = 0.95,
    ccont::sa.SVector{DC,Bool} = sa.SVector{DC,Bool}(fill(true,DC)),
    cdisc::Union{Nothing,bdm.TensorDomain{DC},bdm.CustomTensorDomain{DC}} = nothing
) where {DC,DG,DS}

    DX = ndims(xgrid)
    DZ = isnothing(zproc) ? 0 : ndims(zproc)

    (DX > 0) || @warn("zero endogenous state variables x defined")
    @assert DC > 0 "zero control variables c defined"
    @assert β >= 0 "negative discounting factor found: $β"
    (β > 1)  && @warn("β > 1, the problem may be diverging")
    
    if !all(ccont)
        if isnothing(cdisc)
            error("discrete control claimed but no grid space provided")
        end
    end

    InfiniteHorizonDP{DX,DZ,DC,DG,DS}(
        xgrid,

        zproc,
        
        u,f,lbc,ubc,g,s,

        β,
        ccont,
        cdisc,
    )
end # constructor










# ------------------------------------------------------------------------------
# Finite horizon DP
# ------------------------------------------------------------------------------
# TODO


