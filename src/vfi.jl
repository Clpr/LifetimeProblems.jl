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
        dpr     ::InfiniteHorizonDPResult{DX,DZ,DC,DG,DS,NZ} ;
        options ::IterOptions = IterOptions(),
        io      ::IO = stdout
    ) where {DX,DZ,DC,DG,DS,NZ}

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
    dpr     ::InfiniteHorizonDPResult{DX,DZ,DC,DG,DS,NZ} ;
    options ::IterOptions = IterOptions(),
    io      ::IO = stdout
) where {DX,DZ,DC,DG,DS,NZ}

    # timer
    _tic = time()

    if options.verbose
        println(io, 
            "="^100,
            "\nValue function iteration: ",
            nowstr()
        )
        @printf(
            io,
            "- optimization type: %s\n",
            if all(dpr.dp.ccont)
                "continuous"
            elseif all(.!dpr.dp.ccont)
                "discrete"
            else
                "mixed (MINLP)"
            end
        )
        @printf(
            io, 
            "- #endo states (x): %d, #exog states (z): %d, #controls (c): %d\n",
            DX, DZ, DC
        )
        @printf(
            io,
            "- #non-linear constraints (g): %d, #statistics (s): %d\n",
            DG, DS
        )
        @printf(
            io,
            "- #threads = %d\n",
            options.parallel ? Threads.nthreads() : 1
        )
    end


    # initialization
    options.verbose && print(io, "- Initializing...")
    begin

        # malloc: error trace
        errTrace = Float64[]

        # init guesses of (conditional) value function {v(x,z) : z = 1,...,NZ}
        if !options.use_current_value_guess
            # TODO: allow arbitrary initialization
            initv!(dpr, v0 = 0.0)
        end

        # malloc: updated {v(x,z)}_z
        Vs2  = sa.SizedVector{NZ}([
            Array{Float64,DX}(undef, dpr.dp.xgrid |> size)
            for _ in 1:NZ
        ])
        
        # precond: collection of marginal grids of x
        # notes: cuz we will repeatly do interpolation
        xSubAll  = dpr.dp.xgrid |> CartesianIndices

        # init EV(x)|z := E{v(x,z)|z}
        EVs = [similar(dpr.Vs[1]) for _ in 1:NZ]

        # interp: {EV(x)|z}_z; no overshooting allowed beyond boundaries
        itpEVs = Vector{itp.Extrapolation}(undef, NZ)

        # precond: interpolate policy functions (if no optimization required)
        # notes: if no optimization required, then the same policy function itp
        #        is used for every iteration.
        # notes: no need to interpolate policy functions, as we only need their
        #        on-grid values
        # TODO: allow function-apply style of given policies, in the future
        if !options.optimization
            nothing
        end

    end # begin
    options.verbose && @printf(io, "elapsed %.1f sec\n", time() - _tic)
    
    

    # vfi loop
    options.verbose && println(io, "- Backward iteration starts...")
    for t in 1:options.maxiter
        
        ok2print = options.verbose && ((t == 1) || (t % options.showevery == 0))
        ok2print && println(io, "-"^80, " ", nowstr())

        # precond: update EV(x)|z := E{v(x,z)|z}, stacking & interpolators
        # notes: EVs is undef in 1st round, so loop index to avoid undef ref
        for iz in 1:NZ
            expect!(EVs[iz],dpr,iz)
            itpEVs[iz] = interp_fx(EVs[iz], dpr)
        end

        # statistics of successful optimizations (for diagnosis)
        ctrSuccess::Int = 0

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
                    cs ::SV64{DC},
                    success::Bool,
                    s  ::SV64{DS},
                }
            }[]
            for _ in 1:Threads.nthreads()
        ]


        # optimize: Q-function/Lagrangian (optional)
        
        pb = maybe_pbar(
            xSubAll, 
            ok2print & options.progressbar, 
            io = stdout
        ) # if print a progress bar, then only print to stdout/console

        @maybe_threads options.parallel for xSub in pb

            tid::Int      = Threads.threadid()
            xSV::SV64{DX} = dpr.dp.xgrid[xSub] |> SV64{DX}

            for iz in 1:NZ

                zSV::SV64{DZ} = DZ == 0 ? SV64{0}() : dpr.dp.zproc.states[iz]

                qOpt::Float64, xpOpt::SV64{DX}, cOpt::SV64{DC}, succOpt::Bool, sOpt::SV64{DS} = if options.optimization
                    # case: optimization required

                    # define: optimization problem
                    optProblem = defopt(dpr,xSV,zSV,itpEVs[iz])

                    # solve, using golden section or constrained simplex search
                    _q, _c, flag_success = solve(optProblem,options)

                    # apply: endo state equation
                    _xp = dpr.dp.f(xSV,zSV,_c)

                    # extra ex-post statistics
                    _s = dpr.dp.s(xSV,zSV,_c)

                    _q, _xp, _c, flag_success, _s
                else
                    # case: skipping optimization
                    # do  : use the current policy function to do updating.

                    # apply: given policy functions
                    _c::SV64{DC} = [dpr.Cs[ic,iz][xSub] for ic in 1:DC]

                    # apply: state equations x' = f(x,z,c)
                    _xp::SV64{DX} = dpr.dp.f(xSV,zSV,_c)

                    # apply: Q-function/Lagrangian
                    _upart      = dpr.dp.u(xSV,zSV,_c)
                    _ev         = itpEVs[iz](xSV...)
                    _q::Float64 = _upart + dpr.dp.β * _ev

                    # extra ex-post statistics
                    _s = dpr.dp.s(xSV,zSV,_c)

                    _q, _xp, _c, true, _s
                end # if

                push!(
                    thdRes[tid],
                    (iz,xSub) => (
                        v2  = qOpt,
                        xps = xpOpt,
                        cs  = cOpt,
                        success = succOpt,
                        s   = sOpt
                    )
                )
            end # iz
        end # xSub
        
        # push: thread-wise results
        for resVec in thdRes, ((iz,xSub),res) in resVec

            Vs2[iz][xSub] = res.v2

            for ix in 1:DX
                dpr.Xps[ix,iz][xSub] = res.xps[ix]
            end

            for ic in 1:DC
                dpr.Cs[ic,iz][xSub] = res.cs[ic]
            end

            for is in 1:DS
                dpr.Ss[is,iz][xSub] = res.s[is]
            end

            ctrSuccess += res.success

        end

        # aggregate: error of v(x,z)
        vError = norm.(Vs2 .- dpr.Vs, options.pnorm) |> maximum
        push!(errTrace, vError)

        # aggregate: pct of successfully converged optimizations
        succShare = ctrSuccess / (NZ * length(dpr.Vs[1])) * 100
        
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
                "- max/spent/left time: %.1f/%.1f/%.1f min\n",
                totalMin, secSpent / 60, leftMin
            )
            @printf(io,
                "- avg time per iter: %.2f sec\n",
                secPerIter
            )
            @printf(io,
                "- percent of successfully converged optimizations: %.2f%%\n",
                succShare
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
            dpr.Vs[iz] .= Vs2[iz]
        end

    end # t

    return errTrace
end # vfi!