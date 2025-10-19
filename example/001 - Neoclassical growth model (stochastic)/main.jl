#===============================================================================
Example: Neoclassical growth model, stochastic version
--------------------------------------------------------------------------------
v(k,z) = max_c c^(1-μ)/(1-μ) + βE{v(kp,z')|z}
kp = f(k,z,c) := exp(z) * k^α + (1-δ)*k - c
z' ~ AR(1), i.e. z' = ρ * z + (1-ρ) * zss + σ * ϵ, ϵ ~ N(0,1)
kmin <= kp <= kmax, c >= 0
--------------------------------------------------------------------------------
dimensionalities:
- DX = 1, (capital)
- DZ = 1, (technology shock)
- DC = 1, (consumption)
- DG = 0, (non-negative consumption constraint canceled out by box constraints)
- DS = 2, (income y, and saving rate 1-c/y)
--------------------------------------------------------------------------------
Abbreviations:
- `viz`: visualization
- `par`: parameter
- `ss`: (deterministic) steady state
===============================================================================#
# import LifetimeProblems as ltp

# define: parameters
par = Dict(
    :α   => 0.33,  # capital share
    :β   => 0.90,  # discounting
    :γ   => 2.00,   # risk aversion
    :δ   => 0.025, # depreciation

    :ρz  => 0.95,  # AR1 coefficient of z
    :σz  => 0.075, # volatility of z
    :zss => 0.0,   # long-term mean of z
)

# specify: grid space of endo states
xgrid = begin
    
    # deterministic steady state capital level, used to bound the state space
    kss = let
        _nomi = par[:α] * exp(par[:zss])
        _deno = 1/par[:β] - (1 - par[:δ])
        (_nomi / _deno)^(1/(1-par[:α]))
    end

    # even-sapce grid, sufficient number of points for quantitative analysis
    ltp.bdm.TensorDomain(
        [kss * 0.01,],
        [kss * 2.00,],
        [200,],
    )
end

# specify: Markov chain of exog states
zproc = begin
    # discretize z's AR(1) to a Markov chain, using Tachen 1986
    # 5 states, spanning 2σ around the long-term mean
    ltp.mmc.tauchen(
        ltp.mmc.AR1(ρ = par[:ρz], σ = par[:σz], xavg = par[:zss] ),
        5,      # number of states
        nσ = 2, # μ±2σ
    )
end

# (optional) define: helper function of getting disposable income
function get_y(x,z,par)::Real
    return exp(z[1]) * x[1] ^ par[:α] + (1 - par[:δ]) * x[1]
end

# define: state-dependent box constraints lbc(x,z) and ubc(x,z) for controls
lbc(x,z ; par = par, kmax = xgrid[1][end]) = begin
    y = get_y(x,z,par)

    # derived from: kp = y - c <= kmax --> c >= max{y - kmax, 0}
    return [
        max(y - kmax, 0.0)
    ]
end
ubc(x,z ; par = par, kmin = xgrid[1][1]) = begin
    y = get_y(x,z,par)
    
    # derived from: kp = y - c >= kmin --> c <= y - kmin
    @assert (y - kmin) >= 0 "infeasible income for consumption"

    return [
        y - kmin
    ]
end


# define: flow utility function u(x,z,c), CRRA
u(x,z,c ; par = par) = begin
    y = get_y(x,z,par)

    return if c[1] < 0
        # if infeasible, return something without throwing an error
        -114514.0
    else
        # avoid numerical error at exact 0
        (c[1] + eps())^(1-par[:γ])/(1-par[:γ])
    end
end


# define: state equation of x, the budget constraint
# DX = 1
f(x,z,c ; par = par) = begin
    y  = get_y(x,z,par)
    kp = y - c[1]
    return [
        kp
    ]
end


# define: control's generic constraints, the non-negativity constraint, c >= 0
# hint: neutralized in this example by specifying `lbc` function
# DG = 0
g(x,z,c) = begin
    return []
end


# define: extra statistics that may be used later: income y & saving rate 1-c/y
# DS = 2
s(x,z,c; par = par) = begin
    y = get_y(x,z,par)
    srate = 1 - c[1]/y
    return [
        y,
        srate,
    ]
end


# define: dynamic programming problem
# type parameter: {DC,DG,DS}, number of: 
# - controls
# - non-linear constraints
# - statistics
dp = ltp.InfiniteHorizonDP{1,0,2}(
    xgrid, zproc,
    u, f, lbc, ubc, g, s,
    β = par[:β]
)


# define: result data pack
dpr = ltp.InfiniteHorizonDPResult(dp);


# specify: iteration options
opts = ltp.IterOptions(
    maxiter         = 200,
    tol             = 1E-5,
    parallel        = true,
    optim_algorithm = :brent,
    verbose         = true,
    showevery       = 50
)


# run: value function iteration
# returns: a vector of iteration error trace over iterations
errTrace = ltp.vfi!(dpr, options = opts)




# The following visualization code can help with better understanding the data
# structures.


# viz: convergence pattern
let
    fig = plt.plot(errTrace .|> log, title = "log(error)", xlabel = "iteration")
    plt.hline!(fig, [opts.tol |> log,], color = :red, label = "log(tolerance)")
    fig
end

# viz: value function v(x;z), split lines by z's value
let 
    fig = plt.plot(title = "v(k,z)", xlabel = "k (capital)")
    Ks  = dpr.dp.xgrid[1]
    for (iz, vx) in enumerate(dpr.Vs)
        z = dpr.dp.zproc.states[iz][1]
        plt.plot!(fig, Ks, vx, label = string("z = ", round(z,digits=2)))
    end
    fig
end

# viz: value function v(x;z), assuming z is continuous and interpolate over z
let 
    xzGrids = (
        dpr.dp.xgrid[1],
        dpr.dp.zproc.states .|> first
    )

    # hint: import Interpolations as itp, within in LifetimeProblems.jl
    itpVxz = ltp.itp.linear_interpolation(
        xzGrids,
        reduce(hcat, dpr.Vs),
    )
    fig = plt.surface(
        LinRange(xzGrids[1][1],xzGrids[1][end],70),
        LinRange(xzGrids[2][1],xzGrids[2][end],70),
        (k,z) -> itpVxz(k,z),
        title  = "v(k,z)",
        xlabel = "k",
        ylabel = "z",
        camera = (-30,30),
        alpha = 0.5,
        colorbar = false,
    )
    fig
end

# viz: (equilibrium) policy function c(x;z), split lines by z's states
let 
    # Hint: dpr.Cs[ic,iz], row is ic-th policy, col is iz-th state of exog z
    fig = plt.plot(title = "c(k,z)", xlabel = "k (capital)")
    Ks  = dpr.dp.xgrid[1]
    for (iz, cx) in enumerate(dpr.Cs[1,:])
        z = dpr.dp.zproc.states[iz][1]
        plt.plot!(fig, Ks, cx, label = string("z = ", round(z,digits=2)))
    end
    fig
end

# viz: (equilibrium) state equation x' = f(x;z), split lines by z's states
let 
    # Hint: dpr.Xps[ix,iz], row is ix-th state, col is iz-th state of exog z
    fig = plt.plot(title = "k'(k,z)", xlabel = "k (capital)")
    Ks  = dpr.dp.xgrid[1]
    for (iz, kpx) in enumerate(dpr.Xps[1,:])
        z = dpr.dp.zproc.states[iz][1]
        plt.plot!(fig, Ks, kpx, label = string("z = ", round(z,digits=2)))
    end
    # ref line: 45-degree
    plt.plot!(fig, Ks, Ks, label = "45-degree", color = :red, linestyle = :dash)
    fig
end


# viz: (equilibrium) extra statistics s(x;z), split lines by z's states
let 
    # Hint: dpr.Ss[is,iz], row is is-th stats, col is iz-th state of exog z
    fig = plt.plot(title = "saving rate: 1-c/y", xlabel = "k (capital)")
    Ks  = dpr.dp.xgrid[1]
    for (iz, sx) in enumerate(dpr.Ss[2,:])
        z = dpr.dp.zproc.states[iz][1]
        plt.plot!(fig, Ks, sx, label = string("z = ", round(z,digits=2)))
    end
    fig
end




