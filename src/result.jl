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
mutable struct InfiniteHorizonDPResult{DX,DZ,DC,DG,DS,NZ}

    # DP problem
    dp::InfiniteHorizonDP{DX,DZ,DC,DG,DS}

    # value function's stacking on (x,) grid; collected total NZ such stakcings,
    # where NZ is the number of states of the Markov chain of z.
    # v(x,z)
    Vs::sa.SizedVector{NZ,Array{Float64,DX}}

    # state transition functions stacking on (x,) grid
    # x[j]' = f(x,z,c(x,z)), j = 1,...,DX
    # every dim of x has NZ stacking
    Xps::sa.SizedMatrix{DX,NZ,Array{Float64,DX}}

    # policy functions stakcing on (x,) grid, 
    # c[j](x,z), j = 1,...,DC
    # every dim of c has NZ stacking
    Cs::sa.SizedMatrix{DC,NZ,Array{Float64,DX}}

    # extra statistics stacking on (x,) grid,
    # s[j](x,z), j = 1,...,DS
    Ss::sa.SizedMatrix{DS,NZ,Array{Float64,DX}}

    function InfiniteHorizonDPResult(
        dp::InfiniteHorizonDP{dx,dz,dc,dg,ds}
    ) where {dx,dz,dc,dg,ds}
        # Note: define nz == 1 for deterministic case, for convenience in vfi
        #       as nz is supposed to be consistent with the length of dpr.V
        #       to query elements. this is not conflict with dz == 0.
        nz ::Int            = isnothing(dp.zproc) ? 1 : length(dp.zproc)
        nxs::NTuple{dx,Int} = size(dp.xgrid)

        new{dx,dz,dc,dg,ds,nz}(
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
            ]),

            # Ss
            sa.SizedMatrix{ds,nz,Array{Float64,dx}}([
                Array{Float64,dx}(undef, nxs)
                for _ in 1:ds, _ in 1:nz
            ]),
        )
    end # constructor
end # DPResult
function Base.show(
    io::IO, 
    dpr::InfiniteHorizonDPResult{DX,DZ,DC,DG,DS,NZ}
) where {DX,DZ,DC,DG,DS,NZ}
    
    xsize  = dpr.dp.xgrid |> size
    zsize  = isnothing(dpr.dp.zproc) ? 0 : length(dpr.dp.zproc)
    
    println(io, "Results of Time-homogenous fnfinite horizon lifetime problem")
    println(io, "- #endo states: ", DX)
    println(io, "- #exog states: ", DZ)
    println(io, "- #constraints: ", DG)
    println(io, "- #statistics : ", DS)
    
    println(io, "- size(x nodes)   : ", xsize, ", total = ", xsize |> prod)
    println(io, "- size(z states)  : ", zsize)
    @printf(io, "- RAM usage       : %.3f MB\n", mbsize(dpr))
end # show




# ------------------------------------------------------------------------------
# Helpers: expected value function
# 
# interpolate, update E{v|z}
# ------------------------------------------------------------------------------
"""
    initv!(
        dpr::InfiniteHorizonDPResult, 
        v0 ::Union{Function,Float64} = 0.0,
    )

Initialize the on-grid stacking of value function guess. `v0` can be a scalar
to fill the whole arrays, or a function `v0(x,z)` applying to different states,
where `x` and `z` should accept `SVector{D,Float64}` and the function should
return a real number.

For deterministic problems, an empty `sa.SVector{0,Float64}()` is passed to `v0`
as `z`.
"""
function initv!(
    dpr::InfiniteHorizonDPResult{DX,DZ,DC,DG,DS,NZ}, 
    v0 ::Union{Function,Float64} = 0.0,
) where {DX,DZ,DC,DG,DS,NZ}
    if isa(v0,Float64)
        for V in dpr.Vs
            V .= v0
        end
    else
        xSubAll = dpr.dp.xgrid |> CartesianIndices

        if DZ == 0
            for xSub in xSubAll
                dpr.Vs[1][xSub] = v0(
                    SV64{DX}(dpr.dp.xgrid[xSub]),
                    SV64{0}()
                )
            end
        else
            for (iz,zSV) in enumerate(dpr.dp.zproc.states)
                for xSub in xSubAll
                    dpr.Vs[iz][xSub] = v0(
                        SV64{DX}(dpr.dp.xgrid[xSub]),
                        zSV
                    )
                end
            end
        end

    end
    return nothing
end # initv!
# ------------------------------------------------------------------------------
"""
    expect!(
        EV ::Array{Float64,DX},
        dpr::InfiniteHorizonDPResult{DX,DZ,DC,DG,DS,NZ},
        iz ::Int, # conditional on which z state to take the expectation
    ) where {DX,DZ,DC,DG,DS,NZ}

Computes expected value function `EV(x) = E{v(x,z)|z}` by specifying conditional
on which z state to take the expectation. `EV` is an array that has the same 
size as each stacking array in `dpr.V`.

## Hint
- Under multi-linear interpolation & same grid assumptions, interpolating the 
expected value function (after evaluating expected value at grid points) is
equivalent to take expectation/average over every single evaluation of value
function interpolants. This gives the convenience to interpolate once, then eval
everywhere.
"""
function expect!(
    EV ::Array{Float64,DX},
    dpr::InfiniteHorizonDPResult{DX,DZ,DC,DG,DS,NZ},
    iz ::Int, # conditional on which z state to take the expectation
) where {DX,DZ,DC,DG,DS,NZ}
    
    @assert all(size(EV) .== size(dpr.dp.xgrid)) "size(EV) != size(xgrid)"
    if DZ > 0
        # stochastic model
        @assert 1 <= iz <= NZ "invalid iz, iz must be an integer in [1,NZ]"
    end

    EV .= sum(dpr.Vs .* (DZ == 0 ? [1.0,] : dpr.dp.zproc.Pr[iz,:]))
    return nothing
end # expect!
# ------------------------------------------------------------------------------
function expect(
    dpr::InfiniteHorizonDPResult{DX,DZ,DC,DG,DS,NZ},
    iz ::Int,
)::Array{Float64,DX} where {DX,DZ,DC,DG,DS,NZ}
    EV = similar(dpr.Vs[1])
    expect!(EV,dpr,iz)
    return EV
end # expect
# ------------------------------------------------------------------------------
function interp_fx(
    fxStack  ::Array{Float64,DX},
    dpr      ::InfiniteHorizonDPResult{DX,DZ,DC,DG,DS,NZ} ;
    extrapolation_bc = itp.Flat()
)::itp.Extrapolation where {DX,DZ,DC,DG,DS,NZ}
    return itp.linear_interpolation(
        dpr.dp.xgrid |> collect,
        fxStack, 
        extrapolation_bc = extrapolation_bc
    )
end # interpolate
