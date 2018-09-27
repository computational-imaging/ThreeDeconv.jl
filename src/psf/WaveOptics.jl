# __BEGIN_LICENSE__
#
# WaveOptics.jl
# Author: Michael Broxton (broxton@stanford.edu)
#
# Copyright (C) 2015 Stanford University.
# All rights reserved.
#
# __END_LICENSE__

#__precompile__()   # uncomment to enable precompilation of the full module
module WaveOptics
using FFTW
import ThreeDeconv

# --------
# Includes

# Core
include("core/fftfreq.jl")
include("core/fftpad.jl")
include("core/meshgrid.jl")

# PointSpreadFunction
include("psf/coordinates.jl")
include("psf/recipe.jl")
include("psf/utility.jl")
include("psf/transform.jl")
include("psf/apsf.jl")
include("psf/debye.jl")
include("psf/defocus.jl")

end # WaveOptics module
