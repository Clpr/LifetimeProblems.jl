#===============================================================================
Example: Lucas (1988) human capital accumulation with productivity shock
--------------------------------------------------------------------------------
dimensionalities:
- DX = 2, (capital, human capital)
- DZ = 1, (productivity)
- DC = 2, (kp, share of human capital into work)
- DG = 1, (non-negative consumption)
- DS = 2, (income, consumption)
--------------------------------------------------------------------------------
state orders:
- x := (k,h)
- z := (z,)
- c := (kp,n)
- s := (y,c)
--------------------------------------------------------------------------------
Abbreviations:
- `viz`: visualization
- `par`: parameter
- `ss`: (deterministic) steady state
===============================================================================#
# import LifetimeProblems as ltp

# define: parameters
par = Dict(
    :θ    => 0.5,  # preference on labor disutility
    :γ    => 2.0,  # risk aversion
    :ν    => 1.0,  # labor supply elasticity

    :β    => 0.95, # discounting

    :α    => 0.30, # capital income share

    :δk   => 0.025, # physical capital depreciation
    :δh   => 0.100, # human capital depreciation

    :ϕ    => 1.50,  # technology of human capital investment

    :kmin => 0.05,
    :kmax => 10.0,

    :hmin => 0.05,
    :hmax => 10.0,

    :ρz     => 0.9,
    :σz     => 0.01,  
    :zbar   => 0.0,

)

# specify: grid space of endo states
xgrid = begin
    
    # even-sapce grid, sufficient number of points for quantitative analysis
    ltp.bdm.TensorDomain(
        [par[:kmin], par[:hmin]],
        [par[:kmax], par[:hmax]],
        [50,50],
    )
end

# specify: Markov chain of exog states
zproc = begin
    ltp.mmc.tauchen(
        ltp.mmc.AR1(ρ = par[:ρz], σ = par[:σz], xavg = par[:zbar]),
        4,
        nσ = 2,
    )
end

# (optional) define: helper function of getting the gross income
function get_y(x,z,c,par)::Real
    output = exp(z[1]) * x[1] ^ par[:α] * (c[2] * x[2]) ^ (1 - par[:α])
    kdepre = (1 - par[:δk]) * x[1]
    return output + kdepre
end

# define: state-dependent box constraints lbc(x,z) and ubc(x,z) for controls
lbc(x,z ; par = par) = begin
    return [
        par[:kmin],
        max(
            0.0,
            1 - (par[:hmax] / x[2] - (1 - par[:δh])) / par[:ϕ],
        )
    ]
end
ubc(x,z ; par = par) = begin
    return [
        par[:kmax],
        min(
            1.0,
            1 - (par[:hmin] / x[2] - (1 - par[:δh])) / par[:ϕ],
        )
    ]
end


# define: flow utility function u(x,z,c), net profit
u(x,z,c ; par = par) = begin
    y = get_y(x,z,c,par)
    consumption = y - c[1]
    cpart = if consumption < 0.0
        -2.33E33
    else
        (consumption + eps())^(1-par[:γ]) / (1-par[:γ])
    end
    npart = c[2]^(1+par[:ν]) / (1+par[:ν])
    return cpart - par[:α] * npart
end


# define: state equation of x
f(x,z,c ; par = par) = begin
    return [
        c[1],
        (1-par[:δh]) * x[2] + par[:ϕ] * (1 - c[2]) * x[2],
    ]
end


# define: control's generic constraints, the non-negativity constraint
g(x,z,c ; par = par) = begin
    return [
        c[1] - get_y(x,z,c,par)
    ]
end


# define: extra statistics that may be used later
s(x,z,c; par = par) = begin
    y = get_y(x,z,c,par)
    consumption = y - c[1]
    return [
        y,
        consumption
    ]
end


# define: dynamic programming problem
# type parameter: {DC,DG,DS}, number of: 
# - controls
# - non-linear constraints
# - statistics
dp = ltp.InfiniteHorizonDP{2,1,2}(
    xgrid, zproc,
    u, f, lbc, ubc, g, s,
    β = par[:β]
)


# define: result data pack
dpr = ltp.InfiniteHorizonDPResult(dp);


# specify: iteration options
opts = ltp.IterOptions(
    maxiter         = 300,
    tol             = 1E-3,
    parallel        = true,
    optim_algorithm = :constrainedsimplex,
    verbose         = true,
    showevery       = 30
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


# viz: value function v(x;z), overlayed surfaces by z's value
let 
    figs = plt.Plot[]
    nz   = length(dpr.Vs)

    for iz in 1:nz
        z = dpr.dp.zproc.states[iz][1]
        push!(figs, 
            plt.surface(
                collect(dpr.dp.xgrid)...,
                dpr.Vs[iz],
                title = ltp.@sprintf("v(k,h;z=%.2f)",z),
                xlabel = "k",
                ylabel = "x",
                camera = (-30,30),
                alpha  = 0.5,
                colorbar = false,
            )
        )
    end

    _layout = (
        (nz ÷ 2) + ((nz % 2) == 0 ? 0 : 1),
        2
    )
    plt.plot(figs..., layout = _layout, size = 400 .* reverse(_layout) )
end


# viz: state equations, layered by z's value (assume z is continuous)
let which_policy = :hp

    figs = plt.Plot[]
    nz   = length(dpr.Vs)
    c2ic = Dict(:kp => 1, :hp => 2)

    for iz in 1:nz
        z = dpr.dp.zproc.states[iz][1]
        push!(figs, 
            plt.surface(
                collect(dpr.dp.xgrid)...,
                dpr.Xps[c2ic[which_policy],iz]',
                title = ltp.@sprintf("%s(k,h;z=%.2f)",which_policy,z),
                xlabel = "k",
                ylabel = "x",
                camera = (-30,30),
                colorbar = false,
                alpha  = 0.5,
            )
        )
    end

    _layout = (
        (nz ÷ 2) + ((nz % 2) == 0 ? 0 : 1),
        2
    )
    plt.plot(figs..., layout = _layout, size = 400 .* reverse(_layout) )
end

