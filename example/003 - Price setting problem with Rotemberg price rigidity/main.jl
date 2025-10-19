#===============================================================================
Example: Price setting problem with Rotemberg price rigidity
--------------------------------------------------------------------------------
dimensionalities:
- DX = 1, (last period's price)
- DZ = 2, (marginal cost, output demand)
- DC = 1, (gross inflation rate)
- DG = 0, ()
- DS = 3, (premium, average adjustment cost, total profit amount)
--------------------------------------------------------------------------------
state orders:
- x := (p_{-1},)
- z := (m,y)
- c := (π,)
- s := (q,φ,w)
--------------------------------------------------------------------------------
Abbreviations:
- `viz`: visualization
- `par`: parameter
- `ss`: (deterministic) steady state
===============================================================================#
# import LifetimeProblems as ltp

# define: parameters
par = Dict(
    :θ    => 0.5,  # Rotemberg adjustment cost coefficient
    :πbar => 1.0,  # inflation target    
    
    :β    => 0.95, # discounting

    :pmin => 0.01,
    :pmax => 100.0,

    :ρ      => [0.9 0.0; 0.9 0.0],
    :mybar  => [20.0, 30.0],
    :σm     => 1.0,  
    :σy     => 2.0,
    :cor_my => 0.7, # (not really used in this illustrative example)

)

# specify: grid space of endo states
xgrid = begin
    
    # even-sapce grid, sufficient number of points for quantitative analysis
    ltp.bdm.TensorDomain(
        [par[:pmin],],
        [par[:pmax],],
        [100,],
    )
end

# specify: Markov chain of exog states
zproc = begin
    
    # make marginal grids of (m,y), then cartesian join them
    mgrid = par[:mybar][1] .+ [-2,-1,0,1,2] .* par[:σm]
    ygrid = par[:mybar][1] .+ [-2,-1,0,1,2] .* par[:σy]
    zgrid = Iterators.product(mgrid,ygrid) |> collect .|> collect |> vec
    nz = length(zgrid)

    # mimic the transition matrix with an AR(1) process
    P = ltp.mmc.tauchen(
        ltp.mmc.AR1(ρ = par[:ρ][1], σ = par[:σm]),
        nz,
        nσ = 3
    ).Pr

    ltp.mmc.MultivariateMarkovChain(zgrid, P)
end

# (optional) define: helper function of getting the adjustment cost coefficient
function get_phicoef(x,z,c,par)::Real
    pLast = x[1]
    p     = c[1]
    return 0.5 * par[:θ] * (p/pLast - par[:πbar])^2
end

# define: state-dependent box constraints lbc(x,z) and ubc(x,z) for controls
lbc(x,z ; par = par) = begin
    return [
        par[:pmin]
    ]
end
ubc(x,z ; par = par) = begin
    return [
        par[:pmax]
    ]
end


# define: flow utility function u(x,z,c), net profit
u(x,z,c ; par = par) = begin
    pLast = x[1]
    p = c[1]
    return ((1 - get_phicoef(x,z,c,par)) * p - z[1]) * z[2]
end


# define: state equation of x
f(x,z,c ; par = par) = begin
    return [
        c[1]
    ]
end


# define: control's generic constraints, the non-negativity constraint
g(x,z,c) = begin
    return []
end


# define: extra statistics that may be used later
s(x,z,c; par = par) = begin
    p = c[1]
    pLast = x[1]
    q = p - z[1]
    phi = get_phicoef(x,z,c,par) * p
    return [
        q,
        phi,
        (q - phi) * z[2],
        p/pLast
    ]
end


# define: dynamic programming problem
# type parameter: {DC,DG,DS}, number of: 
# - controls
# - non-linear constraints
# - statistics
dp = ltp.InfiniteHorizonDP{1,0,4}(
    xgrid, zproc,
    u, f, lbc, ubc, g, s,
    β = par[:β]
)


# define: result data pack
dpr = ltp.InfiniteHorizonDPResult(dp);


# specify: iteration options
opts = ltp.IterOptions(
    maxiter         = 500,
    tol             = 1E-3,
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
import Plots as plt

# viz: convergence pattern
let
    fig = plt.plot(errTrace .|> log, title = "log(error)", xlabel = "iteration")
    plt.hline!(fig, [opts.tol |> log,], color = :red, label = "log(tolerance)")
    fig
end

# viz: value function v(x;z), split lines by z's value
let 
    fig = plt.plot(title = "v(p{-1};m,y)", xlabel = "p_{t-1}")
    Ps  = dpr.dp.xgrid[1]
    for (iz, vx) in enumerate(dpr.Vs)
        z = dpr.dp.zproc.states[iz]
        plt.plot!(
            fig, Ps, vx, 
            label = ltp.@sprintf(
                "(m=%.1f, y=%.1f)",
                z...
            )
        )
    end
    fig
end


# viz: policy functions, layered by z's value (assume z is continuous)
let 

    fig1 = plt.plot(
        dpr.Xps[1,:],
        title = "p(p{-1};m,y)", xlabel = "p_{t-1}",
        legend = false
    )
    plt.plot!(fig1, 
        dpr.dp.xgrid[1],
        dpr.dp.xgrid[1],
        color = :red, linestyle = :dash
    )
    fig2 = plt.plot(
        dpr.Ss[4,:], 
        title = "π(p{-1};m,y)", xlabel = "p_{t-1}",
        legend = false
    )

    plt.plot(fig1,fig2)
end



