#===============================================================================
VALUE FUNCTION ITERATION

## Notes

- there is a switch to decide if to do optimization while iterating the value
  function. Shutting down optimizations is used in policy iteration.
===============================================================================#
export vfi!


# ------------------------------------------------------------------------------
# Main iteration pipeline
# ------------------------------------------------------------------------------
"""
    vfi!(
        dpr     ::InfiniteHorizonDPResult{DX,DZ,DC,NZ} ;
        options ::IterOptions = IterOptions(),
        io      ::IO = stdout
    ) where {DX,DZ,DC,NZ}

Run value function iteration (vfi) over a given dynamic programming problem. The
algorithm behaviors are controlled by the options in `options` object.

## Arguments
- `dpr`: an initialized result object.
- `options`: options to specify how the algorithm runs
- `io`: where to write the log (if `options.verbose == true`); a file pointer is
  supported if you would like to write to an external file (create it using e.g.
  `open("file.log","w")`.

## Returns
- `errTrace::Vector{Float64}`: a vector of errors over iterations, for diagnosis

## Notes
- if `options.optimization==false`, i.e. the optimization procedure is turned
  off, then the routine interpolates 
"""
function vfi!(
    dpr     ::InfiniteHorizonDPResult{DX,DZ,DC,NZ} ;
    options ::IterOptions = IterOptions(),
    io      ::IO = stdout
) where {DX,DZ,DC,NZ}

    # TODO
    if !all(dpr.dp.ccont)
        error("VFI with discrete controls not yet implemented.")
    end


    # timer
    _tic = time()

    options.verbose && println(io, 
        "="^100,
        "\nValue function iteration: ",
        nowstr()
    )
    options.verbose && print(io, "- Initializing...")

    # init guesses of (conditional) value function {v(x,z) : z = 1,...,NZ}
    if !options.use_current_value_guess
        for d in 1:DX
            dpr.V[d] .= 0.0
        end
    end

    # init EV(x)|z := E{v(x,z)|z}
    EVs = [similar(dpr.V[1]) for _ in 1:NZ]

    # malloc: updated {v(x,z)}_z
    V2  = sa.SizedVector{NZ}([
        Array{Float64,DX}(undef, dpr.dp.xgrid |> size)
        for _ in 1:NZ
    ])
    
    # malloc: error trace
    errTrace = Float64[]
    
    # precond: collection of marginal grid of x
    # notes: cuz we will repeatly do interpolation
    xmargins = dpr.dp.xgrid |> collect
    xSubAll  = dpr.dp.xgrid |> CartesianIndices

    # interp: {EV(x)|z}_z; no overshooting allowed beyond boundaries
    itpEVs = Vector{itp.Extrapolation}(undef,NZ)

    # precond: interpolate policy functions (if no optimization required)
    itpCs = [
        interp_fx(dpr.Cs[ic,iz], xmargins, options.interpmethod)
        for ic in 1:DC, iz in 1:NZ
    ]

    options.verbose && @printf(io, "elapsed %.1f sec\n", time() - _tic)
    
    # vfi
    options.verbose && println(io, "- Backward iterating...")
    for t in 1:options.maxiter
        
        ok2print = options.verbose && ((t == 1) || (t % options.showevery == 0))
        ok2print && println(io, "-"^80)


        # precond: update EV(x)|z := E{v(x,z)|z}, stacking & interpolators
        for (iz,evStack) in EVs |> enumerate
            expect!(evStack,dpr,iz)
            itpEVs[iz] = interp_fx(evStack, xmargins, options.interpmethod)
        end

        # malloc: thread-wise results
        thdRes = [
            Pair{
                Tuple{
                    Int,               # iz
                    CartesianIndex{DX} # x's index in its grid
                },
                @NamedTuple{
                    v2::Float64,
                    xps::SV64{DX},
                    cs ::SV64{DC}
                }
            }[]
            for _ in 1:Threads.nthreads()
        ]

        # optimize: Q-function/Lagrangian (optional)
        pb = maybe_pbar(
            xSubAll, 
            ok2print & options.progressbar, 
            io = stdout
        ) # if print, then only print in console
        @maybe_threads options.parallel for xSub in pb

            tid::Int      = Threads.threadid()
            xSV::SV64{DX} = dpr.dp.xgrid[xSub] |> SV64{DX}

            for iz in 1:NZ

                zSV::SV64{DZ} = DZ == 0 ? SV64{0}() : dpr.dp.zproc.states[iz]

                qOpt, xpOpt, cOpt = if options.optimization
                    # case: optimization required

                    # define: optimization problem
                    optProblem = defopt(dpr,xSV,zSV,itpEVs[iz])

                    # solve, using golden section or constrained simplex search
                    _q, _c = solve(optProblem,options)

                    # apply: endo state equation
                    _xp = dpr.dp.f(xSV,zSV,_c)

                    Float64(_q), SV64{DX}(_xp), _c
                else
                    # case: skipping optimization
                    # do  : interpolate the current stored policy functions, 
                    # then use it to do updating.

                    # apply: given policy functions
                    _c = [itpCs[ic,iz](xSV...) for ic in 1:DC] |> SV64{DC}

                    # apply: state equations x' = f(x,z,c)
                    _xp = SV64(dpr.dp.f(xSV,zSV,_c))

                    # apply: Q-function/Lagrangian
                    _upart = dpr.dp.u(xSV,zSV,_c)
                    _ev    = itpEVs[iz](xSV...)
                    _q     = _upart + dpr.dp.β * _ev

                    Float64(_q), SV64{DX}(_xp), _c
                end # if

                push!(
                    thdRes[tid],
                    (iz,xSub) => (
                        v2  = qOpt,
                        xps = xpOpt,
                        cs  = cOpt
                    )
                )
            end # iz
        end # xSub
        
        # push: thread-wise results
        for resVec in thdRes, ((iz,xSub),res) in resVec

            V2[iz][xSub] = res.v2

            for ix in 1:DX
                dpr.Xps[ix,iz][xSub] = res.xps[ix]
            end

            for ic in 1:DC
                dpr.Cs[ic,iz][xSub] = res.cs[ic]
            end

        end

        # aggregate: error of v(x,z)
        vError = norm.(V2 .- dpr.V, options.pnorm) |> maximum
        push!(errTrace, vError)

        
        # summarize: iteration
        if ok2print

            secSpent = time() - _tic
            secPerIter, leftMin = estimate_lefttime(
                t, 
                options.maxiter, 
                secSpent
            )
            totalMin = secPerIter * options.maxiter / 60

            @printf(io, 
                "- iteration: %d/%d (%.2f%%)\n", 
                t, options.maxiter, t / options.maxiter * 100
            )
            @printf(io,
                "- error: %.2e vs tol %.2e\n",
                vError, options.tol
            )
            @printf(io,
                "- max/spent/left time (minutes): %.1f/%.1f/%.1f\n",
                totalMin, secSpent / 60, leftMin
            )

        end # if

        # check: convergence
        if vError < options.tol
            if options.verbose
                secSpent = time() - _tic
                println(io, "-"^80)
                println(io, nowstr())
                @printf(io, 
                    "Converged in %d rounds, error %.2e, elapsed %.1f min\n",
                    t, vError, secSpent / 60
                )
                println(io, "="^100)
            end
            break
        elseif t == options.maxiter
            if options.verbose
                secSpent = time() - _tic
                println(io, "-"^80)
                println(io, nowstr())
                @printf(io, 
                    "NOT converge in %d rounds, error %.2e, elapsed %.1f min\n",
                    t, vError, secSpent / 60
                )
                println(io, "="^100)
            end
        end

        # update: v(x,z) guess
        for iz in 1:NZ
            dpr.V[iz] .= V2[iz]
        end

    end # t

    return errTrace
end # vfi