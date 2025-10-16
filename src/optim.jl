#===============================================================================
WRAPPERS FOR OPTIMIZATION ROUTINES
===============================================================================#





# ------------------------------------------------------------------------------
# OOP INTERFACE
# ------------------------------------------------------------------------------
abstract type OptimProblem{DC} <: Any end
# ------------------------------------------------------------------------------
"""
    ContinuousOptimProblem{DC}

A wrapper type for Q-function optimization problem of all continuous controls.
"""
mutable struct ContinuousOptimProblem{DC} <: OptimProblem{DC}

    cbox::bdm.AbstractBoxDomain{DC}

    fobj::Function  # fobj(c); Q-function/Lagrangian; x,z are masked/closured

    g::Function  # g(c) <= 0; x,z are masked/closured

end # ContinuousOptimProblem
# ------------------------------------------------------------------------------
"""
    DiscreteOptimProblem{DC}

A wrapper type for Q-function optimization problem of all discrete controls.
"""
mutable struct DiscreteOptimProblem{DC} <: OptimProblem{DC}

    cbox::bdm.AbstractBoxDomain{DC}

    fobj::Function

    g::Function

end # DiscreteOptimProblem
# ------------------------------------------------------------------------------
"""
    MixedOptimProblem{DC}

A wrapper type for Q-function optimization problem of some continuous controls
and some discrete controls.
"""
mutable struct MixedOptimProblem{DC} <: OptimProblem{DC}

    ccont::sa.SVector{DC,Bool}

    cbox::bdm.AbstractBoxDomain{DC}

    fobj::Function

    g::Function

end # MixedOptimProblem








# ------------------------------------------------------------------------------
# TYPICAL CONSTRUCTORS
# ------------------------------------------------------------------------------
function defopt(
    dpr  ::InfiniteHorizonDPResult{DX,DZ,DC,NZ},
    xSV  ::SV64{DX},
    zSV  ::SV64{DZ},
    itpEv::itp.Extrapolation
)::OptimProblem{DC} where {DX,DZ,DC,NZ}
    return if all(dpr.dp.ccont)
        # case: all controls are continuous
        return ContinuousOptimProblem{DC}(
            dpr.dp.cbox,

            # objective function 
            # (NEGATIVE Q-function, or equivalently, -Lagrangian)
            c -> -( dpr.dp.u(xSV,zSV,c) + dpr.dp.β * itpEv(xSV...) ),

            # inequality constraints g(c) <= 0
            c -> dpr.dp.g(xSV,zSV,c)
        )
    else
        error("not implemented yet")
    end
end # ContinuousOptimProblem









# ------------------------------------------------------------------------------
# SOLVERS
# ------------------------------------------------------------------------------
function solve(
    prob   ::ContinuousOptimProblem{DC}, 
    options::IterOptions
)::Tuple{Float64,SV64{DC}} where DC
    # init guess: (left bottom corner, span radius to the top right)
    c0 = prob.cbox.lb .+ 0.05 .* (prob.cbox.ub .- prob.cbox.lb)

    # trial: try eval `g` to get how many constraints there are
    nCons = let
        _res = prob.g(c0)
        @assert isa(_res,AbstractVector) "g(c) should return a vector-like"
        length(_res)
    end

    # define: constrained minimization problem
    mp = css.MinimizeProblem(
        prob.fobj,
        prob.g,
        c -> Float64[],
        DC, nCons, 0,
        lb = prob.cbox.lb,
        ub = prob.cbox.ub,
    )

    # solve
    optRes = css.solve(
        mp,
        x0 = c0,
        
        verbose = false,
        
        radius  = options.css_radius ,
        δ       = options.css_δ      ,
        R       = options.css_R      ,
        α       = options.css_α      ,
        γ       = options.css_γ      ,
        ρout    = options.css_ρout   ,
        ρin     = options.css_ρin    ,
        σ       = options.css_σ      ,
        ftol    = options.css_ftol   ,
        xtol    = options.css_xtol   ,
        maxiter = options.css_maxiter,
    )

    if optRes.converged & (!optRes.admissible)
        error("optimization converged to non-admissible point")
    elseif (!optRes.converged) & optRes.admissible
        @warn("optimization not converged but admissible")
    elseif (!optRes.converged) & (!optRes.admissible)
        error("optimization not converged and non-admissible")
    end

    if isnan(optRes.f)
        error("optimization converged to NaN")
    elseif isinf(optRes.f)
        error("optimization converged to Inf")
    end

    # CAUTION: flip the sign of `f` to get the actual Q-function
    return Float64(-optRes.f), SV64{DC}(optRes.x)
end # solve




























