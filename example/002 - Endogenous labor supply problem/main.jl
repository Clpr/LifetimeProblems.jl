#===============================================================================
Example: Endogenous labor supply problem
--------------------------------------------------------------------------------
dimensionalities:
- DX = 1, (bond holding)
- DZ = 1, (wage)
- DC = 2, (new bond holding, labor supply)
- DG = 1, (non-negative consumption constraint)
- DS = 3, (income y, consumption, saving rate 1-c/y)
--------------------------------------------------------------------------------
state orders:
- x := (a,)
- z := (z,)
- c := (ap,n)
- s := (y,c,1-c/y)
--------------------------------------------------------------------------------
Abbreviations:
- `viz`: visualization
- `par`: parameter
- `ss`: (deterministic) steady state
===============================================================================#
# import LifetimeProblems as ltp

# define: parameters
par = Dict(
    :α   => 3.0,  # preference on labor disutility
    :β   => 0.99,  # discounting
    :γ   => 2.00,  # risk aversion'
    :ν   => 1.00,  # labor supply elasticity

    :ρz  => 0.95,  # AR1 coefficient of z
    :σz  => 0.075, # volatility of z
    :zss => 0.5,   # long-term mean of z

    :r   => 0.02,  # returns on the risk-free bond

    :amin => 0.05,
    :amax => 100.0,
)

# specify: grid space of endo states
xgrid = begin
    
    # even-sapce grid, sufficient number of points for quantitative analysis
    ltp.bdm.TensorDomain(
        [par[:amin],],
        [par[:amax],],
        [100,],
    )
end

# specify: Markov chain of exog states
zproc = begin
    # discretize z's AR(1) to a Markov chain, using Tachen 1986
    # 5 states, spanning 2σ around the long-term mean
    ltp.mmc.tauchen(
        ltp.mmc.AR1(ρ = par[:ρz], σ = par[:σz], xavg = par[:zss] ),
        3,      # number of states
        nσ = 2, # μ±2σ
    )
end

# (optional) define: helper function of getting disposable income
function get_y(x,z,c,par)::Real
    return (1+par[:r])*x[1] + exp(z[1])*c[2]
end

# define: state-dependent box constraints lbc(x,z) and ubc(x,z) for controls
lbc(x,z ; par = par) = begin
    return [
        par[:amin],
        0.0,
    ]
end
ubc(x,z ; par = par) = begin
    ymax = (1+par[:r])*x[1] + exp(z[1])
    return [
        min(ymax,par[:amax]),
        1.0,
    ]
end


# define: flow utility function u(x,z,c), CRRA
u(x,z,c ; par = par) = begin
    y = get_y(x,z,c,par)
    consumption = y - c[1]

    # if infeasible, return something very negative without throwing
    if consumption < 0
        return -2.33E33
    end

    cpart = (consumption + eps())^(1-par[:γ])/(1-par[:γ])
    npart = c[2] ^ (1+par[:ν]) / (1+par[:ν])
    
    return cpart - par[:α] * npart
end


# define: state equation of x
f(x,z,c ; par = par) = begin
    # hint: directly choose a' in this example
    return [
        c[1]
    ]
end


# define: control's generic constraints, the non-negativity constraint
g(x,z,c; par = par) = begin
    y = get_y(x,z,c,par)
    return [
        c[1] - y
    ]
end


# define: extra statistics that may be used later: income y & saving rate 1-c/y
s(x,z,c; par = par) = begin
    y = get_y(x,z,c,par)
    consumption = y - c[1]
    srate = 1 - consumption/y
    return [
        y,
        consumption,
        srate,
    ]
end


# define: dynamic programming problem
# type parameter: {DC,DG,DS}, number of: 
# - controls
# - non-linear constraints
# - statistics
dp = ltp.InfiniteHorizonDP{2,1,3}(
    xgrid, zproc,
    u, f, lbc, ubc, g, s,
    β = par[:β]
)


# define: result data pack
dpr = ltp.InfiniteHorizonDPResult(dp);


# specify: iteration options
opts = ltp.IterOptions(
    maxiter         = 800,
    tol             = 1E-3,
    parallel        = true,
    optim_algorithm = :constrainedsimplex,
    verbose         = true,
    showevery       = 100
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
    fig = plt.plot(title = "v(a,z)", xlabel = "a (bond holding)")
    As  = dpr.dp.xgrid[1]
    for (iz, vx) in enumerate(dpr.Vs)
        z = dpr.dp.zproc.states[iz][1]
        plt.plot!(fig, As, vx, label = string("z = ", round(z,digits=2)))
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
        (a,z) -> itpVxz(a,z),
        title  = "v(a,z)",
        xlabel = "a",
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
    fig1 = plt.plot(title = "a'(a,z)", xlabel = "a (bond holding)")
    fig2 = plt.plot(title = "n(a,z)", xlabel = "a (bond holding)")
    As  = dpr.dp.xgrid[1]
    for iz in 1:length(dpr.Vs)
        z = dpr.dp.zproc.states[iz][1]
        cx1 = dpr.Cs[1,iz] # 1st policy: a'
        cx2 = dpr.Cs[2,iz] # 2nd policy: n
        plt.plot!(fig1, As, cx1, label = string("z = ", round(z,digits=2)))
        plt.plot!(fig2, As, cx2, label = string("z = ", round(z,digits=2)))
    end
    plt.plot(fig1,fig2)
end


# viz: (equilibrium) extra statistics s(x;z), split lines by z's states
let 
    # Hint: dpr.Ss[is,iz], row is is-th stats, col is iz-th state of exog z

    As = dpr.dp.xgrid[1]

    figs = plt.Plot[]
    for (is,sName) in ["y","c","srate"] |> enumerate
        _fig = plt.plot(title = string(sName,"(a,z)"), xlabel = "a")
        for (iz, sx) in enumerate(dpr.Ss[is,:])
            z = dpr.dp.zproc.states[iz][1]
            plt.plot!(_fig, As, sx, label = string("z = ", round(z,digits=2)))
        end
        push!(figs,_fig)
    end
    plt.plot(figs...)
end




