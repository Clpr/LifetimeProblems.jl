module LifetimeProblems
# ==============================================================================
using  LinearAlgebra, SparseArrays
import Printf: @printf, @sprintf

import Distributions as distrib
import Interpolations as itp
import StaticArrays as sa

import MultivariateMarkovChains as mmc
import BoxDomains as bdm


# alias
SV64{D} = sa.SVector{D,Float64}


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


# ------------------------------------------------------------------------------
# VALUE FUNCTION ITERATION
# ------------------------------------------------------------------------------
include("vfi.jl")











# ==============================================================================
end # LifetimeProblems