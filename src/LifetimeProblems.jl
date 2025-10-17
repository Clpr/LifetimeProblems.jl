module LifetimeProblems
# ==============================================================================
using  LinearAlgebra, SparseArrays
import Printf: @printf, @sprintf
import Dates

import Interpolations as itp
import StaticArrays as sa
import ProgressBars: ProgressBar
import Optim as opt

import MultivariateMarkovChains as mmc
import BoxDomains as bdm
import ConstrainedSimplexSearch as css


# alias
SV64{D} = sa.SVector{D,Float64}
SizedV64{D} = sa.SizedVector{D,Float64}

# ------------------------------------------------------------------------------
# GENERIC HELPERS
# ------------------------------------------------------------------------------
include("generic.jl")


# ------------------------------------------------------------------------------
# ITERATION SOLVERS
# ------------------------------------------------------------------------------
include("itersol.jl")



# ------------------------------------------------------------------------------
# BELLMAN EQUATION (DYNAMIC PROGRAMMING) MODEL
# ------------------------------------------------------------------------------
include("bellman.jl")
include("result.jl")

# ------------------------------------------------------------------------------
# WRAPPER FOR OPTIMIZATION ROUTINES
# ------------------------------------------------------------------------------
include("optim.jl")


# ------------------------------------------------------------------------------
# VALUE FUNCTION ITERATION
# ------------------------------------------------------------------------------
include("vfi.jl")











# ==============================================================================
end # LifetimeProblems