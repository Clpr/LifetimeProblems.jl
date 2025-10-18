#===============================================================================
WRAPPERS FOR OPTIMIZATION ROUTINES
===============================================================================#





# ------------------------------------------------------------------------------
# OOP INTERFACE
# ------------------------------------------------------------------------------
abstract type OptimProblem{DC,DG} <: Any end
OptimResult{DC} = Tuple{Float64,SV64{DC},Bool}
# ------------------------------------------------------------------------------
"""
    ContinuousOptimProblem{DC,DG}

A wrapper type for Q-function optimization problem of all continuous controls.
"""
mutable struct ContinuousOptimProblem{DC,DG} <: OptimProblem{DC,DG}

    fobj::Function  # fobj(c); Q-function/Lagrangian; x,z are masked/closured

    lbc::SizedV64{DC}
    ubc::SizedV64{DC}

    g::Function  # g(c) <= 0; x,z are masked/closured

end # ContinuousOptimProblem
function Base.show(io::IO, prob::ContinuousOptimProblem{DC,DG}) where {DC,DG}
    println(io, 
        "Optimization problem (#controls = ", DC, "), all continous"
    )
    for i in 1:DC
        @printf(io, 
            "%.3f <= c[%d] <= %.3f\n",
            prob.lbc[i], i, prob.ubc[i]
        )
    end
    println(io, "#non-linear constraints: ", DG)
    return nothing
end # show
# ------------------------------------------------------------------------------
"""
    DiscreteOptimProblem{DC,DG}

A wrapper type for Q-function optimization problem of all discrete controls.
"""
mutable struct DiscreteOptimProblem{DC,DG} <: OptimProblem{DC,DG}

    fobj::Function

    lbc::SizedV64{DC}
    ubc::SizedV64{DC}

    cgrid::Union{bdm.TensorDomain{DC},bdm.CustomTensorDomain{DC}}

    g::Function

end # DiscreteOptimProblem
function Base.show(io::IO, prob::DiscreteOptimProblem{DC,DG}) where {DC,DG}
    println(io, 
        "Optimization problem (#controls = ", DC, "), all discrete"
    )
    for i in 1:DC
        @printf(io, 
            "%.3f <= c[%d] <= %.3f, #nodes = %d\n",
            prob.cgrid.lb[i],
            i,
            prob.cgrid.ub[i],
            prob.cgrid.Ns[i]
        )
    end
    println(io, "#non-linear constraints: ", DG)
    return nothing
end # show
# ------------------------------------------------------------------------------
"""
    MixedOptimProblem{DC,DG}

A wrapper type for Q-function optimization problem of some continuous controls
and some discrete controls.
"""
mutable struct MixedOptimProblem{DC,DG} <: OptimProblem{DC,DG}

    fobj::Function

    lbc::SizedV64{DC}
    ubc::SizedV64{DC}

    ccont::sa.SizedVector{DC,Bool}
    cgrid::Union{bdm.TensorDomain{DC},bdm.CustomTensorDomain{DC}}

    g::Function

end # MixedOptimProblem
function Base.show(io::IO, prob::MixedOptimProblem{DC,DG}) where {DC,DG}
    println(io, 
        "Optimization problem (#controls = ", DC, "), mixed continuity"
    )
    for i in 1:DC
        if prob.ccont[i]
            @printf(io, 
                "%.3f <= c[%d] <= %.3f, continuous\n",
                prob.lbc[i],
                i,
                prob.ubc[i],
            )
        else
            @printf(io, 
                "%.3f <= c[%d] <= %.3f, discrete, #nodes = %d\n",
                prob.lbc[i],
                i,
                prob.ubc[i],
                prob.cgrid.Ns[i]
            )
        end
    end
    println(io, "#non-linear constraints: ", DG)
    return nothing
end # show








# ------------------------------------------------------------------------------
# TYPICAL CONSTRUCTORS
# ------------------------------------------------------------------------------
"""
    defopt(
        dpr  ::InfiniteHorizonDPResult{DX,DZ,DC,DG,DS,NZ},
        xSV  ::SV64{DX},
        zSV  ::SV64{DZ},
        itpEv::itp.Extrapolation{Float64,DX}
    )::OptimProblem{DC} where {DX,DZ,DC,DG,DS,NZ}

Defines optimization problem for a given grid point (x,z). `itpEv` is the a
linear interpolant of the conditional expected value function ev(x):=E{v(x,z)|z}
"""
function defopt(
    dpr  ::InfiniteHorizonDPResult{DX,DZ,DC,DG,DS,NZ},
    xSV  ::SV64{DX},
    zSV  ::SV64{DZ},
    itpEv::itp.Extrapolation{Float64,DX}  # v(x)|z, interpolated over x's grid
)::OptimProblem{DC} where {DX,DZ,DC,DG,DS,NZ}

    if DC <= 0
        error("no control variable claimed, why doing the optimization stage?")
    end

    _lb = dpr.dp.lbc(xSV,zSV) |> SizedV64{DC}
    _ub = dpr.dp.ubc(xSV,zSV) |> SizedV64{DC}

    @assert all(_lb .<= _ub) "lower bounds > upper bounds found"
    @assert all(isfinite,_lb) "Inf lower bound(s) found"
    @assert all(isfinite,_ub) "Inf upper bound(s) found"

    # objective function to MINimize
    # (NEGATIVE Q-function, or equivalently, -Lagrangian)
    _fobj(cVec) = -( dpr.dp.u(xSV,zSV,cVec) + dpr.dp.β * itpEv(xSV...) )

    # inequality constraints g(c) <= 0
    _gcons(cVec) = dpr.dp.g(xSV,zSV,cVec)

    return if all(dpr.dp.ccont)
        # case: all controls are continuous
        return ContinuousOptimProblem{DC,DG}(_fobj, _lb, _ub, _gcons)
    elseif all(.!dpr.dp.ccont)
        # case: all controls are discrete
        return DiscreteOptimProblem{DC,DG}(_fobj,_lb,_ub,dpr.dp.cgrid,_gcons)
    else
        # case: mixed integer problem (some are discrete)
        return MixedOptimProblem{DC,DG}(
            _fobj,
            _lb,_ub,
            dpr.dp.ccont,
            dpr.dp.cgrid,
            _gcons
        )
    end
end # ContinuousOptimProblem









#=------------------------------------------------------------------------------
SOLVERS: PRIVATE IMPLEMENTATION FOR: ContinuousOptimProblem

abbreviation:
- pvt: private
- dimensionality of control variables
    - 1d: 1-dimensional optimization problem (DC == 1)
    - nd: n-dimensional optimization problem (DC > 1)
- continuity of control variables
    - c : all control variables are continuous
    - d : all control variables are discrete 
    - m : mixed, some control variables are discrete, some are continuous
- algorithms
    - brent: Brent's method, for 1D box-constrained problem
        - no x0 guess needed
        - no generic g(x,z,c)<=0 constraint allowed
        - IterOptions:
            - `optim_xtol` works, but `optim_ftol` does not work
    - aps: adaptive particle swarm, for N-D box-constrained problem
        - no x0 guess needed
        - no generic g(x,z,c)<=0 constraint allowed
        - IterOptions:
            - `aps_nparticle >= 3` required
    - alterdirect: alternating direction search, with golden section
        - no x0 guess needed (default midpoint of the box space)
        - no generic g(x,z,c)<=0 constraint allowed
        - IterOptions:
            - the larger one of `optim_xtol` and `optim_ftol` is used
    - css: constrained Nelder-Mead (simplex), for N-D box-constrained problem
        - no x0 guess needed (use largest initial simplex)
        - generic g() constraint allowed
        - IterOptions:
            - many options available in IterOptions leading with `css_`
            - `optim_xtol` and `optim_ftol` both work

## Notes
- In econ scenario, almost all optimization problems are at least bouned by a
box space, so we skip all unbounded optimization methods.
------------------------------------------------------------------------------=#
function _pvt_solve_1d_c_brent(
    prob::ContinuousOptimProblem{DC,DG},
    options::IterOptions
)::OptimResult{DC} where {DC,DG}
    # private method: solve a ContinuousOptimProblem{DC,DG} using Brent's method

    # hint: Brent's method dominates golden section in most cases.
    #       no initial guess needed.
    # caution: flip back the sign of objective function!
    # notes: check the source code of Optim.jl to correctly specify tolerance
    #        and other solver-specific options.

    res = opt.optimize(
        cScalar -> prob.fobj([cScalar,]),
        prob.lbc[1], prob.ubc[1],
        opt.Brent(),
        abs_tol = options.optim_xtol,
        iterations = options.optim_maxiter,
    )
    return Float64(-res.minimum), SV64{DC}(res.minimizer), opt.converged(res)
end
# ------------------------------------------------------------------------------
function _pvt_solve_nd_c_aps(
    prob   ::ContinuousOptimProblem{DC,DG},
    options::IterOptions
)::OptimResult{DC} where {DC,DG}
    # private method: solve a ContinuousOptimProblem{DC,DG} using Adaptive
    # Particle Swarm method
    # caution: the converged() has not been implemented in Optim.jl, so well as
    #          the control of tolerance. be careful when using APS.
    res = opt.optimize(
        cVec -> prob.fobj(cVec),
        prob.lbc, prob.ubc,
        opt.ParticleSwarm(n_particles = options.aps_nparticle),
        opt.Options(
            x_abstol = options.optim_xtol,
            f_abstol = options.optim_ftol,
            iterations = options.optim_maxiter,
        )
    )
    return Float64(-res.minimum), SV64{DC}(res.minimizer), true
end
# ------------------------------------------------------------------------------
function _pvt_solve_nd_c_ad(
    prob   ::ContinuousOptimProblem{DC,DG},
    options::IterOptions
)::OptimResult{DC} where {DC,DG}
    # private method: solve a ContinuousOptimProblem{DC,DG} using alternating
    # direction, where line search is done by golden section search
    res = alterdirect(
        prob.fobj,
        (prob.lbc .+ prob.ubc) ./ 2.0,
        prob.lbc, prob.ubc,
        tol = max(options.optim_xtol, options.optim_ftol),
        maxiter = options.optim_maxiter
    )
    return Float64(-res[2]), SV64{DC}(res[1]), res[3]
end
# ------------------------------------------------------------------------------
function _pvt_solve_nd_c_css(
    prob   ::ContinuousOptimProblem{DC,DG},
    options::IterOptions
)::OptimResult{DC} where {DC,DG}
    # private method: solve a ContinuousOptimProblem{DC,DG} using
    # constrained simplex search provided in `ConstrainedSimplexSearch.jl`
    
    # initial point x0 required to create simplex

    mp = css.MinimizeProblem(
        prob.fobj, prob.g, cVec -> Float64[],
        DC,        DG,     0,
        lb = prob.lbc,
        ub = prob.ubc,
    )

    res = css.solve(
        mp,
        x0 = prob.lbc .+ 0.05 .* (prob.ubc .- prob.lbc),

        radius   = options.css_radius,
        towards  = :upper,

        verbose   = false,
        showevery = 1,    

        δ    = options.css_δ   ,
        R    = options.css_R   ,
        α    = options.css_α   ,
        γ    = options.css_γ   ,
        ρout = options.css_ρout,
        ρin  = options.css_ρin ,
        σ    = options.css_σ   ,

        ftol    = options.optim_ftol,
        xtol    = options.optim_xtol,
        maxiter = options.optim_maxiter,
    )

    flag_success::Bool = res.converged & res.admissible

    return Float64(-res.f), SV64{DC}(res.x), flag_success
end




#=------------------------------------------------------------------------------
SOLVERS: PRIVATE IMPLEMENTATION FOR: DiscreteOptimProblem

abbreviation:
- pvt: private
- dimensionality of control variables
    - 1d: 1-dimensional optimization problem (DC == 1)
    - nd: n-dimensional optimization problem (DC > 1)
- continuity of control variables
    - c : all control variables are continuous
    - d : all control variables are discrete 
    - m : mixed, some control variables are discrete, some are continuous
- algorithms
    - grid search
        - no x0 guess needed
        - generic g(x,z,c)<=0 constraint allowed, as "filters"
        - IterOptions: nothing needs to be specified


## Notes
- In econ scenario, we need global optimum, so grid search is the only option
------------------------------------------------------------------------------=#
function _pvt_solve_nd_d_grid(
    prob::DiscreteOptimProblem{DC,DG},
    options::IterOptions
)::OptimResult{DC} where {DC,DG}
    # private method: solve a DiscreteOptimProblem using grid search with filter

    # eval fobj at all c grid points
    fMin, cMinSub = findmin(
        cTup -> begin
            cSV = SV64{DC}(cTup)
            flg1 = any(prob.g(cSV) .> 0)
            flg2 = any(cSV .< prob.lbc) || any(cSV .> prob.ubc)
            if flg1 || flg2
                return Inf
            else
                return prob.fobj(cSV)
            end
        end,
        prob.cgrid |> Iterators.product |> collect
    )
    @assert fMin != Inf "max_c Q(c|x,z) == -Inf found, no admissible control"

    # explicit CartesianIndex() for 1-dim case
    return Float64(-fMin), SV64{DC}(prob.cgrid[CartesianIndex(cMinSub)]), true
end












# ------------------------------------------------------------------------------
# SOLVERS: PUBLIC API
# ------------------------------------------------------------------------------
function solve(
    prob   ::ContinuousOptimProblem{DC,DG}, 
    options::IterOptions
)::OptimResult{DC} where {DC,DG}

    return if options.optim_algorithm == :brent
        if DC == 1
            if DG > 0
                @warn "non-linear constraint(s) will be ignored by Brent's method"
            end
            return _pvt_solve_1d_c_brent(prob, options)
        else
            error("specified Brent's method for non 1-dimensional problem")
        end
    elseif options.optim_algorithm == :aps
        if DC == 1
            error("specified Adaptive Particle Swarm (APS) method for 1-dimensional problem; change to Brent's method (optim_algorithm = :brent)!")
        else
            if DG > 0
                @warn "non-linear constraint(s) will be ignored by Adaptive Particle Swarm method"
            end
            @warn "Optim.Options() has not been well-implemented with APS method yet. All iterations and tolerance options fail!"
            return _pvt_solve_nd_c_aps(prob, options)
        end
    elseif options.optim_algorithm == :alterdirect
        if DC == 1
            error("specified alternating direction method for 1-dimensional problem; change to Brent's method (optim_algorithm = :brent)!")
        else
            if DG > 0
                @warn "non-linear constraint(s) will be ignored by alternating direction method"
            end
            return _pvt_solve_nd_c_ad(prob, options)
        end
    elseif options.optim_algorithm == :constrainedsimplex
        if DG == 0
            error("no non-linear constraints found. a box constraint is sufficient and more efficient, optim_algorithm = :brent, :aps, :alterdirect")
        else
            return _pvt_solve_nd_c_css(prob, options)
        end
    else
        error("unsupported algorithm type: ", options.optim_algorithm)
    end
end # solve
# ------------------------------------------------------------------------------
function solve(
    prob   ::DiscreteOptimProblem{DC,DG}, 
    options::IterOptions
)::OptimResult{DC} where {DC,DG}

    if options.optim_algorithm == :gridsearch
        return _pvt_solve_nd_d_grid(prob,options)
    else
        error("unsupported algorithm type: ", options.optim_algorithm)
    end
end # solve









#=------------------------------------------------------------------------------
SOLVERS: PRIVATE IMPLEMENTATION FOR: MixedOptimProblem

abbreviation:
- pvt: private
- dimensionality of control variables
    - 1d: 1-dimensional optimization problem (DC == 1)
    - nd: n-dimensional optimization problem (DC > 1)
- continuity of control variables
    - c : all control variables are continuous
    - d : all control variables are discrete 
    - m : mixed, some control variables are discrete, some are continuous
- algorithms


CAUTION: stuff of MixedOptimProblem is placed after stuff for continuous and
discrete problems, because _pvt_solve_nd_m depends on `solve` for the continuous
problem (not specific private solver, because we use the overloaded `solve` for
continuous problems to check if the user-provided algorithm is valid but not
duplicate another set of branchings.


## Notes
- the basic idea is that: for every grid point of those discrete dimensions, we 
solve an all-continuous control (sub) problem by varying the other continuous
dimensions and fixing the discrete dimensions. Then, we compare the result of
all the grid points of discrete dimensions to obtain the final solution.
- depending on types of the constraints, the `optim_algorithm` can be one of the
algorithms that work for `ContinuousOptimProblem`.
- This problem is basically a non-linear mixed integer problem, which is NP-hard
------------------------------------------------------------------------------=#
function _pvt_solve_nd_m(
    prob::MixedOptimProblem{DC,DG},
    options::IterOptions
)::OptimResult{DC} where {DC,DG}
    # private method: solve a MixedOptimProblem by solving children continuous
    # problems

    @assert DC > 1 "1-dim mixed continuity problem claimed, impossible."

    # get which dims are discrete
    dimCont::Vector{Int} = findall(prob.ccont)
    dimDisc::Vector{Int} = setdiff(1:DC, dimCont)
    DCcont = length(dimCont)
    DCdisc = DC - DCcont

    # create a grid of those discrete dimensions
    cgridDisc = prob.cgrid[dimDisc]


    # define a wrapper to quickly create a full c vector by fixing dimDisc
    _pvt_fullc(cContVec, cDiscVec) = begin
        cFull = Vector{Float64}(undef,DC)
        cFull[dimCont] .= cContVec
        cFull[dimDisc] .= cDiscVec
        return cFull       
    end

    # solve all-continuous children problems and stack them
    res = Array{OptimResult{DCdisc}}(undef, cgridDisc |> size)
    for cDiscSub in cgridDisc |> CartesianIndices
        cDiscVec = cgridDisc[cDiscSub]

        # define child all-continuous problem
        subprob = ContinuousOptimProblem{DCcont,DG}(
            cContVec -> prob.fobj(_pvt_fullc(cContVec,cDiscVec)),
            prob.lbc[dimCont],
            prob.ubc[dimCont],
            cContVec -> prob.g(_pvt_fullc(cContVec,cDiscVec))
        )

        # solve it
        res[cDiscSub] = solve(subprob, options)

    end # cDiscSub

    # find optimal
    fMin, _cDiscOptSub = findmin(
        optres -> begin
            if optres[3]
                # case: if the child problem is solved successfully
                optres[1]
            else
                # case: if fails to solve the child problem
                Inf
            end
        end,
        res
    )

    # compose final result
    cDiscOptSub = CartesianIndex(_cDiscOptSub) # in case of 1-dim, which is Int
    cOpt = Vector{Float64}(undef,DC)
    cOpt[dimDisc] .= cgridDisc[cDiscOptSub]
    cOpt[dimCont] .= res[cDiscOptSub][2]
    success = res[cDiscOptSub][3]

    return Float64(-fMin), SV64{DC}(cOpt), success
end
# ------------------------------------------------------------------------------
# SOLVERS: PUBLIC API
# ------------------------------------------------------------------------------
function solve(
    prob   ::MixedOptimProblem{DC,DG}, 
    options::IterOptions
)::OptimResult{DC} where {DC,DG}

    DCont::Int = prob.ccont |> sum

    @assert DC > 1 "<= 1 controls claimed. cannot be a mixed continuity problem"

    if 0 < DCont < DC
        return _pvt_solve_nd_m(prob, options)
    elseif DCont == DC
        error("all controls are claimed continuous")
    elseif DCont > DC
        error("more continuous controls than total number of controls, sure?")
    else
        # DCcont <= 0
        error("no continuous controls, is it an all-discrete control problem?")
    end

    return _pvt_solve_nd_m
end # solve

