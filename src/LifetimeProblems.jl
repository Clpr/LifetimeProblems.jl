module LifetimeProblems
# ==============================================================================
using  LinearAlgebra, SparseArrays
import Printf: @printf, @sprintf

import Distributions as distrib
import Interpolations as itp
import StaticArrays as sa

import MultivariateMarkovChains as mmc
import BoxDomains as bdm


include("generic.jl")


# ------------------------------------------------------------------------------
# INTERFACE: ITERATION SOLVERS
# ------------------------------------------------------------------------------
include("itersol.jl")



# ------------------------------------------------------------------------------
# DATA TYPE: BELLMAN EQUATION (DYNAMIC PROGRAMMING) MODEL
# ------------------------------------------------------------------------------
include("components.jl")
include("bellman.jl")















# ==============================================================================
end # LifetimeProblems