# __BEGIN_LICENSE__
#
# WaveOptics.jl
#
# Copyright (c) 2015, Stanford University
#
# All rights reserved.
#
# Redistribution and use in source and binary forms for academic and other
# non-commercial purposes with or without modification, are permitted provided
# that the following conditions are met:
#
# * Redistributions of source code, including modified source code, must retain
#   the above copyright notice, this list of conditions and the following
#   disclaimer.
#
# * Redistributions in binary form or a modified form of the source code must
#   reproduce the above copyright notice, this list of conditions and the
#   following disclaimer in the documentation and/or other materials provided with
#   the distribution.
#
# * Neither the name of The Leland Stanford Junior University, any of its
#   trademarks, the names of its employees, nor contributors to the source code
#   may be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# * Where a modified version of the source code is redistributed publicly in
#   source or binary forms, the modified source code must be published in a freely
#   accessible manner, or otherwise redistributed at no charge to anyone
#   requesting a copy of the modified source code, subject to the same terms as
#   this agreement.
#
# THIS SOFTWARE IS PROVIDED BY THE TRUSTEES OF THE LELAND STANFORD JUNIOR
# UNIVERSITY "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE LELAND STANFORD JUNIOR
# UNIVERSITY OR ITS TRUSTEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
#
# __END_LICENSE__

abstract type OpticalRecipe end

# Aperture code types (instances are defined in the 'aperture' sub-module)
abstract type CodedAperture end

const MaskParameters = Dict{String, Float64}
const MaskOptions = Dict{String, Any}

# -------------------------------------------------------------------------------------------
#                                 Null Coded Aperture
# -------------------------------------------------------------------------------------------

mutable struct NullCodedAperture <: CodedAperture
    mask_name::String
    mask_parameters::MaskParameters
    mask_options::MaskOptions
end

function NullCodedAperture()
    return NullCodedAperture("null_phase", MaskParameters(), MaskOptions())
end

function NullCodedAperture(mask_parameters::MaskParameters, mask_options)
    return NullCodedAperture("null_phase", MaskParameters(), MaskOptions())
end

function mask(ap::NullCodedAperture,  coordinates::OpticalCoordinates, parameters = Dict{String, Float64}())
    return ones(Complex64, size(coordinates.rho))
end

function jacobian(ap::NullCodedAperture, coordinates::OpticalCoordinates, parameters = Dict{String,Float64}())
    error("No Jacobian defined for NullCodedAperture")
end


# ------------------------------------------------------------------------------------------------------
#                                   Wide Field Optical Recipe
# ------------------------------------------------------------------------------------------------------

mutable struct WideFieldOpticalRecipe <: OpticalRecipe
    objective_mag::Float64
    objective_na::Float64
    wavelength::Float64
    medium_index::Float64
    f_tubelens::Float64
    f_objective::Float64
    static_aperture_mask
    dynamic_aperture_mask
    normalize_intensity::Bool
end

"""
Generic Optical Recipe for Widefield PSF Models

**Parameters**

    objective_mag - magnification of the objective
    objective_na - numerical aperture of the objective
    wavelength - emission wavelength of the light being imaged (all simulations are monochromatic)
    medium_index - index of refraction of the sample medium
    f_tubelens - tube lens focal length (e.g. 200mm for Nikon, 180mm for Olympus, etc.)
    static_aperture_mask - Supply a back aperture mask that is a subclass of CodedAperture, or
               'NullCodedAperture()' if there is no aperture code.  The static mask is computed
               when the PSF model is constructed, and is therefore more efficient than the
               dynamic_aperture_mask.
    dynamic_aperture_mask - Supply a back aperture mask that is a subclass of CodedAperture, or
               'NullCodedAperture()' if there is no aperture code.  The dynamic_aperture_mask
               is recomputed each time apsf() is called, but this flexibility incurs additional
               performance overhead.
    normalize_intensity - Set max PSF value @ z = 0 to 1.0.  This is an arbitrary choice, but
               one that keeps PSF values relatively well scaled and makes it easy to adjust the
               "exposure" of the simulated microscope by multipyling the PSF to get some number of
               photons that is >= 1.  However, be careful when mixing optical models with different
               recipes, because their absolute intensities cannot be directly compared with this
               arbitrary scaling in effect.
"""
function WideFieldOpticalRecipe(objective_mag::Number,
                                objective_na::Float64,
                                wavelength::Float64,
                                medium_index::Float64,
                                f_tubelens::Float64;
                                static_aperture_mask::CodedAperture = NullCodedAperture(),
                                dynamic_aperture_mask::CodedAperture = NullCodedAperture(),
                                normalize_intensity::Bool = false)

    f_objective = f_tubelens / objective_mag;
    return WideFieldOpticalRecipe(objective_mag, objective_na,
                                  wavelength, medium_index,
                                  f_tubelens, f_objective,
                                  static_aperture_mask,
                                  dynamic_aperture_mask,
                                  normalize_intensity)
end
