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

using Interpolations


# -----------------------------------------------------------------------------
#           AMPLITUDE POINT SPREAD FUNCTION BASE TYPE & METHODS
# -----------------------------------------------------------------------------

# Base Type
#
# All PointSpreadFunctionModel subtypes will inherit from this base type.
abstract type PointSpreadFunctionModel end

# In a lateral shift invariant models, the image at each z plane can
# be computed using a convolution with a PSF for that z-plane.
#
# Examples: DebyeModel(), VectorialDebyeModel(), and DefocusModel().
#
abstract type LateralShiftInvariantModel <: PointSpreadFunctionModel end

# In a periodic shift invariant models, the point spread function
# displays some periodic lateral shift invariance, but is not completey
# shift invariant.  Projection operators for such models can still be
# computed fairly efficiently using a sequence of convolution operations.
#
# Subtypes of this type must define 'shift_invariance_subpixels' and
# 'shift_invariance_dimensions' to indicate the periodicity of that
# particular model.
#
# Examples: LightFieldModel()
#
abstract type PeriodicShiftInvariantModel <: PointSpreadFunctionModel end

# Shift variant models cannot be projected using convolution, since
# they allow for PSFs that change with every 3D point location.
#
# Examples: CodedApertureLightFieldModel()
#
abstract type ShiftVariantModel <: PointSpreadFunctionModel end

# --------------
# Shared Methods
#
function num_observations(model::PointSpreadFunctionModel)
    return Int64(model.num_observations)
end

function atf(model::PointSpreadFunctionModel, p, theta = Float64[]; na_limit = true, apply_tilt = true)
    error("The atf() method is not implemented for this model type.")
end

function otf_2d(model::PointSpreadFunctionModel, p, theta = Float64[])
    error("The atf() method is not implemented for this model type.")
end

function apsf(model::PointSpreadFunctionModel, p, theta = Float64[])
    error("The apsf() method is not implemented for this model type.")
end

function psf(model::PointSpreadFunctionModel, p, theta = Float64[])
    wf = WaveOptics.apsf(model, p, theta)
    return abs2.(wf)
end


"""
Estimate the diameter (in meters) of an image large enough to contain the point
spread function for a wide field microscope without any phase mask.  Most
(but not all) point spread functions with back aperture masks will fit in
this diameter.

Here, p = (x,y,z) is a tuple containing the point generating this PSF.
"""
function estimate_psf_size(model::PointSpreadFunctionModel, p)

        # Check length and unpack p
        assert(length(p) == 3)
        x,y,z = p

        objective_theta = asin(model.recipe.objective_na / model.recipe.medium_index)
        aperture_diameter = abs( 2 * z * tan(objective_theta) )
        rayleigh_limit = model.recipe.wavelength / (2 * model.recipe.objective_na)

        # Choose a PSF size that comfortably fits the full PSF, with special
        # handling for when z = 0. (The min size is 8 * rayleigh_limit,
        # otherwise it is 2 * aperture_diameter.)
        final_diameter_m = max(8*rayleigh_limit, 2*aperture_diameter)
        return final_diameter_m
end

function optimal_sampling(model::PointSpreadFunctionModel, sim_size_m; oversampling = nothing)
        """
        Given a desired simulation size, this function returns a sampling
        rate (in pixels) that will yield an accurate wave optics simulation.

        Note that this sampling rate is only an estimate based on the abbe
        limit. It may be too high in some cases, and too low in others. Use it
        as a starting point.
        """

        if oversampling == nothing
            oversampling = 1
        end

        # We use the abbe limit as an estimate of the optimal sampling rate. We
        # recommend sampling at 2x the Abbe limit.
        abbe_limit = model.recipe.wavelength / (2. * model.recipe.objective_na);
    return sim_size_m / (2*abbe_limit) * oversampling
end

# --------------------------------------------------------------------------
#                COMMON PHASE MASKS & APSF UTILITIES
# --------------------------------------------------------------------------

"""
This function takes an APSF lookup up table computed using apsf_scalar_debye
or apsf_defocus_atf() and uses the translational invariance property of the
APSF to interpolate and place it at any location x, y at the depth z for
which the apsf_template was computed.

Note, the translation could have been accomplished with a tilt in the back
aperture plane via translation_mask(), but this would have potentially led
to sampling problems if a large tilt (due to a large x or y translation)
exceeded the nyquist sampling rate of the back aperture. For general
translations which may be large, this method is more robust.
"""
function translate_apsf(sim_size_m, sim_size_px, apsf::AbstractArray{T, 2}, dx, dy) where T<:Number
    sample_period = sim_size_m / sim_size_px

    # Use 2D interpolation to shift the translationally invariant microscope PSF
    apsf_real_interp = interpolate(real(apsf), BSpline(Interpolations.Linear()))
    apsf_imag_interp = interpolate(imag(apsf), BSpline(Interpolations.Linear()))

    result = zeros(eltype(apsf), size(apsf))
    for c in 1:size(apsf,2)
        for r in 1:size(apsf,1)
            result[r,c] = apsf_real_interp(r + dy/sample_period, c - dx/sample_period) + im * apsf_imag_interp(r + dy/sample_period, c - dx/sample_period)
        end
    end
    return result
end

"""
Creates a phase mask in the back aperture plane corresponding to a
translation in the image plane. This translation mask is based on the
Fourier shift theorem. For small shifts, it should match with the
translate_apsf() function below.

fx, fy must be in 'normalized' cartesian coordinates (see fourier_plane_meshgrid()).
"""
function translation_mask(dx, dy, fx::Array{T, 2}, fy::Array{T, 2},
                          objective_na::Float64,
                          wavelength::Float64) where T<:AbstractFloat
    result = Array{ComplexF32}(size(fx))
    omega = -2. * im * pi * objective_na/wavelength
    for i in eachindex(result)
        result[i] = exp(omega * (fx[i] * dx + fy[i] * (-dy)))
    end
    return result
end


"""
Implements the defocus phase function described in

Hanser, B. M., Gustafsson, M., & Agard, D. A. (2004). Phase-retrieved pupil
functions in wide-field fluorescence microscopy. Journal of Microscopy. Vol.
216. pp. 32-48. 2004.

This phase function models roughly what happens if the microscope objective
were to be refocused to a different depth 'dz' in the object.

r, theta must be in 'normalized' polar coordinates (see fourier_plane_meshgrid()).
"""
function defocus_mask(dz::S, r::AbstractArray{T,2}, theta::AbstractArray{T,2},
                      objective_na::Float64, wavelength::Float64, medium_index::Float64) where T<:AbstractFloat where S<:Number
    M,N = size(r)
    k = medium_index/wavelength
    omega = T(k^2) .- T(objective_na/wavelength)^2 * r.^2

    # Ensure we don't try to take the sqrt() of a negative number. This should
    # only happen outside the boundary of the aperture, so this check amounts to
    # zeroing out the phase of mask pixels outside the numerical aperture.
    binary_mask = omega .>= 0

    # Since CuArrays.jl doesn't have a function to compute exp(ComplexFloat),
    # we manually compute it.
    phi = T(-2π * dz) * sqrt.(max.(omega, 0))
    phase =  (cos.(phi) + 1im .* sin.(phi)) .* binary_mask
    return phase
end


# --------------------------------------------------------------------------
#                       APODIZATION FUNCTIONS
# --------------------------------------------------------------------------

"""
Apply a the apodization function for the Abbe sine condition. This
implements eqn. 6.3.7 from Min Gu's "Advanced Optical Imaging Theory"

r, theta must be in "normalized" polar coordinates (see fourier_plane_meshgrid()).
"""
function abbe_sine_apodization(rho::Array{T}, theta::Array{T}, objective_na::AbstractFloat, medium_index::AbstractFloat) where T<:AbstractFloat

    result = zeros(ComplexF32, size(rho))
    omega = objective_na/medium_index
    for i in eachindex(rho)
        if rho[i] <= 1.0
            alpha = asin(omega * rho[i])    # Gross eq. 21-3 and 21-4
            apodization = sqrt(cos(alpha))                # Gu eq. 6.3.10
            result[i] = apodization + 0. * im
        else
            result[i] = 0.
        end
    end
    return result
end
