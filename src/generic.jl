#===============================================================================
GENERIC HELPERS
===============================================================================#




# ------------------------------------------------------------------------------
# CONTROL FLOWS
# ------------------------------------------------------------------------------
"""
    maybe_threads(flag::Bool, expr::Expr)

Wrap an expression `expr` with `Threads.@threads` macro if `flag` is `true`.

Usage: `@maybe_threads true for i in .....`
"""
macro maybe_threads(flag, expr)
    quote
        if $(flag)
            Threads.@threads $expr
        else
            $expr
        end
    end |> esc
end # maybe_threads



# ------------------------------------------------------------------------------
# ELEMENTARY MATH
# ------------------------------------------------------------------------------
function perturb(x::AbstractVector, i::Int, v)
    x1 = x |> copy; x1[i] = v
    return x1
end # perturb
function perturb!(x::AbstractVector, i::Int, v)
    x[i] = v
    return nothing
end # perturb!





# ------------------------------------------------------------------------------
# NUMERICAL METHODS
# ------------------------------------------------------------------------------
"""
    golden(f, a, b; tol=1e-8)

Minimize a unimodal function `f(x)` on the interval [`a`, `b`] using the golden 
section search. Returns the approximate minimizer and the found minimum.

Estimated times of function evaluations: `N = 2 + log(2*tol/(b-a)) / log(0.618)`
"""
function golden(f::Function, a::Real, b::Real; tol::Float64 = 1E-8)
    ϕ = (sqrt(5) - 1) / 2  # golden ratio conjugate
    c = b - ϕ * (b - a)
    d = a + ϕ * (b - a)
    fc = f(c)
    fd = f(d)
    while abs(b - a) > tol
        if fc < fd
            b, d, fd = d, c, fc
            c = b - ϕ * (b - a)
            fc = f(c)
        else
            a, c, fc = c, d, fd
            d = a + ϕ * (b - a)
            fd = f(d)
        end
    end
    x_min = (a + b) / 2
    f_min = f(x_min)
    return x_min, f_min
end # golden
# ------------------------------------------------------------------------------
"""
    alterdirect(
        f::Function, 
        x0::AbstractVector{<:Real},
        lb::AbstractVector{<:Real},
        ub::AbstractVector{<:Real}; 
        tol::Float64 = 1E-8,
        maxiter::Int = 1000
    ) -> Tuple{Vector{Float64}, Float64}

Performs coordinate-wise direct search optimization (alternating direction 
search) to minimize the objective function `f` over a box-constrained domain.

# Arguments
- `f::Function`: The objective function to minimize. Should accept a vector of 
real numbers and return a real value.
- `x0::AbstractVector{<:Real}`: Initial guess for the minimizer. Must satisfy 
`lb .<= x0 .<= ub`.
- `lb::AbstractVector{<:Real}`: Lower bounds for each variable.
- `ub::AbstractVector{<:Real}`: Upper bounds for each variable.

# Keyword Arguments
- `tol::Float64=1E-8`: Tolerance for convergence. The algorithm stops if the 
change in solution or objective value is less than `tol`.
- `maxiter::Int=1000`: Maximum number of iterations.

# Returns
- `x_min::Vector{Float64}`: The estimated minimizer.
- `f_min::Float64`: The minimum value of `f` found.

# Notes
- For the 1-dimensional case, uses the golden section search method.
- For higher dimensions, alternates optimization along each coordinate using 
golden section search.
- The function asserts that the initial guess is within the bounds and that all 
parameters are valid.
"""
function alterdirect(
    f ::Function, 
    x0::AbstractVector{<:Real},
    lb::AbstractVector{<:Real},
    ub::AbstractVector{<:Real} ;
    tol::Float64 = 1E-8,
    maxiter::Int = 1000
)
    D = length(x0)

    @assert all(lb .<= x0 .<= ub) "x0 not in [lb,ub] box region, and lb .<= ub"
    @assert tol > 0 "tol must be positive"
    @assert maxiter > 0 "max iteration must be positive"
    @assert D > 0 "x0 must have at least one element"

    # special case: unimodal
    if D == 1
        x_min, f_min = golden(
            xj -> f([xj,]),
            lb[1], ub[1], tol = tol
        )
        return [x_min,], f_min
    end

    x = copy(x0)
    f_prev = f(x)
    for _ in 1:maxiter
        x_old = copy(x)
        for j in 1:D
            xj_min, _ = golden(
                xj -> (xj_vec = copy(x); xj_vec[j] = xj; f(xj_vec)),
                lb[j], ub[j], tol = tol
            )
            x[j] = xj_min
        end
        f_curr = f(x)
        if norm(x - x_old, Inf) < tol || abs(f_curr - f_prev) < tol
            return x, f_curr
        end
        f_prev = f_curr
    end
    return x, f_prev
end # alterdirect











































