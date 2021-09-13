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
using QuadGK, SpecialFunctions

# -----------------------------------------------------------------------------
#               WIDE FIELD MICROSCOPE APSF (DEBYE MODEL)
# -----------------------------------------------------------------------------

mutable struct DebyeModel <: PointSpreadFunctionModel
    recipe::OpticalRecipe
    sim_size_m::Float64
    sim_size_px::Int64
    num_observations::Int32
    include_intensity_scalefactor::Bool

end # type DebyeModel

"""
Computes the amplitude PSF of a point source at (x,y,z) using a scalar Debye
model. This code implements Equation 3.4.15 and 6.2.17 in 'Advanced Optical Imaging
Theory' by Min Gu.

This model assumes that the 3D APSF of the microscope is tranlationally invariant
in x, y, and z. While this assumption is accurate for a large range of depths
around the native object plane, it can break down for zery large z-depths.  The
approximation only holds if eqn. 3.4.4 in Gu's book is satisfied.  See Sections 3.3
and 3.4 for more discussion, and be careful when using this PSF to model large
z offsets.

**Parameters**

    recipe - an optical recipe containing important simulation parameters
    sim_size_m - linear dimension (used for width & height) in meters (or other units
             of your choice) to be used for the PSF simulation.
    sim_size_px - the number of pixel in each linear dimension.  All simulations are square.
    include_intensity_scalefactor - Include the intensity scaling found in Gu's original
           equations.  Off by default since this makes the intensities very large, which
           can lead to numerical problems.  Intensities are not really calibrated here anyway.
"""
function DebyeModel(
    recipe::OpticalRecipe,
    sim_size_m::Float64,
    sim_size_px::Int64;
    include_intensity_scalefactor::Bool = false,
)

    num_observations = 1

    return DebyeModel(
        recipe,
        sim_size_m,
        sim_size_px,
        num_observations,
        include_intensity_scalefactor,
    )

end # DebyeModel constructor

function num_parameters(model::DebyeModel)
    return 0
end

#-------
# Numerical integration helper functions

"""
`function complex_quadrature(func, a, b)`

Allows numerical integration of complex valued functions.  The real and imaginary parts of the
function are integrated seperately and then combined together.
"""
function complex_quadrature(func, a, b)
    real_integral, err = quadgk(x -> real(func(x)), a, b, rtol = 1e-6, atol = 1e-9)
    imag_integral, err = quadgk(x -> imag(func(x)), a, b, rtol = 1e-6, atol = 1e-9)
    return real_integral + 1im * imag_integral
end

"""
The Fresnel integral function is suitable for low numerical aperture (eqn.
3.4.15). This is an approximation of the full Debye function below.
"""
function fresnel_integrand_fn(rho::Float64, u::Float64, v::Float64)
    return exp(1im * u / 2 * rho .^ 2) * besselj(0, rho * v) * 2 * pi .* rho
end

function fresnel_apsf_fn(u::Float64, v::Float64, alpha_o::Float64)
    cmplx_integral = complex_quadrature(rho -> fresnel_integrand_fn(rho, u, v), 0.0, 1.0)
    return exp(-1im * u / (4 * sin(alpha_o / 2)^2)) * cmplx_integral
end


"""
The Debye integral function (based on eqn. 6.2.17).

We use a abbe sine apodization function here: P(theta) = P(r) * sqrt(cos(theta))
"""
function debye_integrand_fn(theta::Float64, alpha::Float64, u::Float64, v::Float64)
    apodization = sqrt(cos(theta))       # Abbe sine apodization
    return apodization * exp(1im * u * sin(theta / 2) .^ 2 / (2 * sin(alpha / 2) .^ 2)) .*
           besselj(0, sin(theta) / sin(alpha) * v) .* sin(theta)
end

function debye_apsf_fn(
    u::Float64,
    v::Float64,
    alpha_o::Float64,
    k::Float64,
    z_image::Float64,
)
    cmplx_integral =
        complex_quadrature(theta -> debye_integrand_fn(theta, alpha_o, u, v), 0.0, alpha_o)
    return 1im * exp(-1im * k * z_image) * cmplx_integral
end

"""
Compute the amplitude point spread function.

**Parameters**

    p - position of the point source
    theta - the parameters to pass to the aperture_mask.  If theta = Float64[], the aperture's default
      parameters will be used.

**Returns**

    apsf - 2D image of the APSF on the image sensor
"""
function apsf(model::DebyeModel, p, theta = Float64[])
    x, y, z = p   # unpack p

    sample_period = model.sim_size_m / model.sim_size_px
    f_objective = model.recipe.f_tubelens / model.recipe.objective_mag          # Objective focal length
    d1 = f_objective                                                            # d1 = f1 ?? (I'm not sure of this...)
    alpha_o = asin(model.recipe.objective_na / model.recipe.medium_index)       # Object side numerical aperture angle
    k = model.recipe.medium_index * 2.0 * pi / model.recipe.wavelength           # Wave number, with medium_index factored in

    # Compute the Fresnel number of the objective at this z depth.
    a = model.recipe.objective_na / model.recipe.medium_index * f_objective
    d10 = f_objective - z
    fresnel_number = pi * a^2 / (model.recipe.wavelength * d10)

    # The PSF is radially symmetric, so we will convert a set of cartesian
    # coordinates to polar coordinates and use those below to interpolate
    # radial coordinates and produce a 2D image.
    x_ticks = range(
        -model.sim_size_m / 2.0 + sample_period / 2.0,
        stop = model.sim_size_m / 2.0 - sample_period / 2.0,
        length = model.sim_size_px,
    )
    X, Y = WaveOptics.meshgrid(x_ticks, x_ticks)
    R = sqrt.(X .^ 2 + Y .^ 2)
    max_r = maximum(R)

    # Create a lookup table of PSF values as a function of radius. Sample at
    # 2x the cartesian sampling rate to make sure we can intepolate back to
    # a cartesian grid without introducing artifacts.
    dr = model.sim_size_m / (2.0 * model.sim_size_px)
    rho = 0:dr:max_r+sample_period/2

    # Compute the APSF for each element of the lookup table.
    apsf_vals = zeros(ComplexF32, size(rho))
    z_image = 0.0
    z_object = z

    # Set intensity scale factor
    scale_factor = 1.0
    if (fresnel_number < 1.0) && model.include_intensity_scalefactor
        scale_factor = M / (d1^2.0 * wavelength^2)
    end

    if (fresnel_number >= 1.0) && model.include_intensity_scalefactor
        scale_factor = 2.0 * pi / wavelength
    end

    for i = 1:length(rho)
        # Normalized optical coordinates, Eqns. 6.2.16
        v = k * rho[i] * sin(alpha_o)           # Radial (transverse) optical coordinate
        u = 4 * k * z_object * sin(alpha_o / 2)^2 # Axial optical coordinate

        if fresnel_number < 1.0
            apsf_vals[i] = scale_factor * fresnel_apsf_fn(u, v, alpha_o)
        else
            apsf_vals[i] = scale_factor * debye_apsf_fn(u, v, alpha_o, k, z_image)
        end
    end

    # Interpolate to populate the image.
    apsf_img_real = Interpolations.scale(
        interpolate(real(apsf_vals), BSpline(Quadratic(Line(Interpolations.OnCell())))),
        rho,
    )
    apsf_img_imag = Interpolations.scale(
        interpolate(imag(apsf_vals), BSpline(Quadratic(Line(Interpolations.OnCell())))),
        rho,
    )


    # Interpolate
    apsf = zeros(ComplexF32, size(R))
    for i in eachindex(R)  # Efficiently iterate over array elements
        apsf[i] = apsf_img_real(R[i]) + 1im * apsf_img_imag(R[i])
    end

    if !(x == 0 && y == 0)
        # Translate and return the result.
        apsf = translate_apsf(model.sim_size_m, model.sim_size_px, apsf, x, y)
    end

    return apsf
end
