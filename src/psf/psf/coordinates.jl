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

# -----------------------------------------------------------------------------
#               BACK APERTURE PLANE COORDINATE SYSTEMS
# -----------------------------------------------------------------------------

# Types for fourier_plane_meshgrid()
struct CartesianCoordinateSystem end
struct PolarCoordinateSystem end
struct BothCoordinateSystem end

abstract type OpticalCoordinateUnits end
struct SpatialFrequencyUnits <: OpticalCoordinateUnits end
struct MetricUnits <: OpticalCoordinateUnits end
struct NormalizedUnits <: OpticalCoordinateUnits end

mutable struct OpticalCoordinates
    fx::Array{Float32,2}
    fy::Array{Float32,2}
    rho::Array{Float32,2}
    theta::Array{Float32,2}
    units::OpticalCoordinateUnits
end

function _fourier_plane_meshgrid_freqs(sim_size_px::Int64, sample_period::Float64,
                                       objective_na::Float64, wavelength::Float64, focal_length::Float64,
                                       units::SpatialFrequencyUnits)
    return fftshift(fftfreq(sim_size_px, sample_period))
end

function _fourier_plane_meshgrid_freqs(sim_size_px::Int64, sample_period::Float64,
                                       objective_na::Float64, wavelength::Float64, focal_length::Float64,
                                       units::MetricUnits)
    freqs = fftshift(fftfreq(sim_size_px, sample_period))
    freqs *= focal_len * wavelength
    return freqs
end

function _fourier_plane_meshgrid_freqs(sim_size_px::Int64, sample_period::Float64,
                                       objective_na::Float64, wavelength::Float64, focal_length::Float64,
                                       units::NormalizedUnits)
    freqs = fftshift(fftfreq(sim_size_px, sample_period))
    back_aperture_radius = objective_na / wavelength  # From equation 21-4 in Gross
    freqs /= back_aperture_radius
    return freqs
end


function _fourier_plane_meshgrid_mesh(freqs, coordinate_system::CartesianCoordinateSystem)
    fx, fy = WaveOptics.meshgrid(freqs, freqs)
    return (fx, fy)
end

function _fourier_plane_meshgrid_mesh(freqs, coordinate_system::PolarCoordinateSystem)
    fx, fy = WaveOptics.meshgrid(freqs, freqs)
    r = sqrt.(fx.*fx+fy.*fy)
    theta = atan.(fy, fx)
    return (r, theta)
end

function _fourier_plane_meshgrid_mesh(freqs, coordinate_system::BothCoordinateSystem)
    fx, fy = WaveOptics.meshgrid(freqs, freqs)
    r = sqrt.(fx.*fx+fy.*fy)
    theta = atan.(fy, fx)
    return (fx, fy, r, theta)
end


"""
Returns a 2D grid of spatial frequencies (in both cartesian and polar
coordinates). Frequencies are normalized such that `R == 1` is along the edge
of the aperture defined by `objective_na / wavelength`.

Many functions will repeatedly re-use the coordinates produced by
fourier_plane_meshgrid(), so we will momoize the results so that they are
cached if the arguments to the function remain the same again and again.

**Parameters**

`sim_size_px` - the size of the simulation in the object (or image) plane

`sample_period` - the period (i.e. sample spacing) in the object (or image) plane

`objective_na, wavelength` - numerical aperture and emission wavelength

`focal_length` - focal length of the objective (or tubelens).  Used when units = "meters"

`units` - string
    one of : `[ "spatial_freq", "m", "normalized" ]`

* `"spatial_freq"` - return back aperture coordinates in terms of the spatial frequency of waves in the object plane.
* `"meters"` - returns back aperture coordinates in terms of their real
* `"normalized"` - returns back aperture coordinates in a normalized coordinate space where the aperture lies on the unit disc, |r| <= 1.

`coordinate_system` - string
    one of : `[ "cartesian", "polar", "both" ]`

**Returns**

`(fx, fy)` when `coordinate_system == "cartesian"` or `(r, theta)` when `coordinate_system == "polar"`

"""
function fourier_plane_meshgrid(sim_size_px::Int64, sample_period::Float64,
                                objective_na::Float64, wavelength::Float64,
                                focal_length::Float64, units::S) where S<:OpticalCoordinateUnits
    freqs = _fourier_plane_meshgrid_freqs(sim_size_px, sample_period, objective_na,
                                          wavelength, focal_length, units)
    (fx,fy,rho,theta) = _fourier_plane_meshgrid_mesh(freqs, BothCoordinateSystem())
    return OpticalCoordinates(fx, fy, rho, theta, units)
end
