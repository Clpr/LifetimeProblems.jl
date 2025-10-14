#===============================================================================
COMPONENTS IN FORMULATING DYNAMIC PROGRAMMING PROBLEMS
===============================================================================#

abstract type Component   <: Any end
abstract type Uncertainty <: Component end
abstract type Continuity  <: Component end



# ------------------------------------------------------------------------------
# Continuity (of controls)
#
# notes:
# - 
# ------------------------------------------------------------------------------
# per-control variable's continuity
struct Continuous  <: Continuity end
struct Discrete    <: Continuity
    grid::Vector
end


