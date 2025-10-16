#===============================================================================
RESULT TYPE FOR BELLMAN EQUATIONS
===============================================================================#
export InfiniteHorizonDPResult



# ------------------------------------------------------------------------------
"""
    InfiniteHorizonDPResult(dp::InfiniteHorizonDP)

Results of solving a dynamic programming problem. Should be created before doing
any iteration to avoid stack overflow.

## Example
```julia
dpr = ltp.InfiniteHorizonDPResult(dp)
```
"""
mutable struct InfiniteHorizonDPResult{DX,DZ,DC,NZ}

    # DP problem
    dp::InfiniteHorizonDP{DX,DZ,DC}

    # value function's stacking on (x,) grid; collected total NZ such stakcings,
    # where NZ is the number of states of the Markov chain of z.
    # v(x,z)
    V ::sa.SizedVector{NZ,Array{Float64,DX}}

    # state transition functions stacking on (x,) grid
    # x[j]' = f(x,z,c(x,z)), j = 1,...,DX
    # every dim of x has NZ stacking
    Xps::sa.SizedMatrix{DX,NZ,Array{Float64,DX}}

    # policy functions stakcing on (x,) grid, 
    # c[j](x,z), j = 1,...,DC
    # every dim of c has NZ stacking
    Cs::sa.SizedMatrix{DC,NZ,Array{Float64,DX}}

    function InfiniteHorizonDPResult(
        dp::InfiniteHorizonDP{dx,dz,dc}
    ) where {dx,dz,dc}
        # Note: define nz == 1 for deterministic case, for convenience in vfi
        #       as nz is supposed to be consistent with the length of dpr.V
        #       to query elements. this is not conflict with dz == 0.
        nz ::Int            = isnothing(dp.zproc) ? 1 : length(dp.zproc)
        nxs::NTuple{dx,Int} = size(dp.xgrid)

        new{dx,dz,dc,nz}(
            dp,

            # V
            sa.SizedVector{nz,Array{Float64,dx}}([
                Array{Float64,dx}(undef, nxs) for _ in 1:nz
            ]),
            
            # Xps
            sa.SizedMatrix{dx,nz,Array{Float64,dx}}([
                Array{Float64,dx}(undef, nxs)
                for _ in 1:dx, _ in 1:nz
            ]),

            # Cs
            sa.SizedMatrix{dc,nz,Array{Float64,dx}}([
                Array{Float64,dx}(undef, nxs)
                for _ in 1:dc, _ in 1:nz
            ])
        )
    end # constructor
end # DPResult
function Base.show(
    io::IO, 
    dpr::InfiniteHorizonDPResult{DX,DZ,DC,NZ}
) where {DX,DZ,DC,NZ}
    xsize  = dpr.dp.xgrid |> size
    zsize  = isnothing(dpr.dp.zproc) ? 0 : length(dpr.dp.zproc)
    println(io, "Results of Infinite Horizon Dynamic Programming")
    @printf(io, "- dimensionalities: #x = %d, #z = %d, #c = %d\n", DX,DZ,DC)
    println(io, "- size(x nodes)   : ", xsize, ", total = ", xsize |> prod)
    println(io, "- size(z states)  : ", zsize)
    @printf(io, "- RAM usage       : %.3f MB", mbsize(dpr))
end # show




# ------------------------------------------------------------------------------
# Helpers
# 
# interpolate, update E{v|z}
# ------------------------------------------------------------------------------
"""
    expect!(
        EV ::Array{Float64,DX},
        dpr::InfiniteHorizonDPResult{DX,DZ,DC,NZ},
        iz ::Int, # conditional on which z state to take the expectation
    ) where {DX,DZ,DC,NZ}

Computes expected value function `EV(x) = E{v(x,z)|z}` by specifying conditional
on which z state to take the expectation. `EV` is an array that has the same 
size as each stacking array in `dpr.V`.
"""
function expect!(
    EV ::Array{Float64,DX},
    dpr::InfiniteHorizonDPResult{DX,DZ,DC,NZ},
    iz ::Int, # conditional on which z state to take the expectation
) where {DX,DZ,DC,NZ}
    
    @assert all(size(EV) .== size(dpr.dp.xgrid)) "size(EV) != size(xgrid)"
    if DZ > 0
        # stochastic model
        @assert 1 <= iz <= NZ "invalid iz, iz must be an integer in [1,NZ]"
    end

    EV .= sum(dpr.V .* (DZ == 0 ? [1.0,] : dpr.dp.zproc.Pr[iz,:]))
    return nothing
end # expect!
# ------------------------------------------------------------------------------
function expect(
    dpr::InfiniteHorizonDPResult{DX,DZ,DC,NZ},
    iz ::Int,
)::Array{Float64,DX} where {DX,DZ,DC,NZ}
    EV = similar(dpr.V[1])
    expect!(EV,dpr,iz)
    return EV
end # expect
# ------------------------------------------------------------------------------
function interp_fx(
    fxStack  ::Array{Float64,DX},
    xs       ::Tuple,
    itpmethod::Symbol ;
    extrapolation_bc = itp.Flat()
) where {DX}
    if itpmethod == :linear
        return itp.linear_interpolation(
            xs, fxStack, 
            extrapolation_bc = extrapolation_bc
        )
    elseif itpmethod == :cubic
        return itp.cubic_spline_interpolation(
            xs, fxStack, 
            extrapolation_bc = extrapolation_bc
        )
    else
        error("interpolation method $(itpmethod) not yet supported")
    end
end # interpolate
