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

# import ..WaveOptics

# -------------------------------------------------------------------------
#               WIDE FIELD MICROSCOPE APSF (DEFOCUS MODEL)
# -----------------------------------------------------------------------------

mutable struct DefocusModel <: LateralShiftInvariantModel
    recipe::OpticalRecipe
    sim_size_m::Float64
    sim_size_px::Int64

    padding::Int32
    num_observations::Int32

    # Private model state information
    aperture_plane_coordinates::OpticalCoordinates
    microscope_atf::Array{ComplexF32,2}
    user_intensity_scaling_factor::Float32
end

"""
This code implements compute a microscope APSF by simulating the refocusing of
a microscope with a defocus mask in the back aperture plane [1].  We then
Fourier transforming the back aperture ATF to obtain the image plane APSF.  This
is more efficient than carrying out numerical integration, as in the Debye model.
However, this model does utilize the Debye model to produce a baseline ATF, on top
of which defocus is added.

This approach is faster than the Debye models below, although it is not quite as
accurate and it can be prone to sampling artifacts.  Generally this model produces
results that are within 1% (and often better) of the Debye model.  However you should
always compare the two models to be sure.

References:

[1] Hanser, B. M., Gustafsson, M., & Agard, D. A. (2004). Phase-retrieved pupil
functions in wide-field fluorescence microscopy. Journal of Microscopy. Vol.
216. pp. 32-48. 2004.

**Parameters**

      recipe - an optical recipe containing important simulation parameters
      sim_size_m - linear dimension (used for width & height) in meters (or other units
               of your choice) to be used for the PSF simulation.
      sim_size_px - the number of pixel in each linear dimension.  All simulations are square.
"""
function DefocusModel(recipe::OpticalRecipe, sim_size_m::Float64, sim_size_px::Int64)

    # TODO: Add a check to see if sim_size_m and sim_size_px are sufficient to produce
    # an accurate simulation.


    num_observations = 1

    # We will zero pad the FFT to increase the sampling rate in k-space (i.e. the back aperture)
    # and ensure that there are periodic edge wrapping artifacts in the final PSF.  (Multiplication
    # in the Fourier Domain corresponds to convolution in the primal domain, and in the case of the
    # FFT that means the circular convolution due to periodicity of the discrete fourier transform.)
    padding = 2
    f_objective = recipe.f_tubelens / recipe.objective_mag

    aperture_plane_coordinates = fourier_plane_meshgrid(sim_size_px * padding,
                                                        sim_size_m/sim_size_px,
                                                        recipe.objective_na,
                                                        recipe.wavelength,
                                                        f_objective,
                                                        NormalizedUnits())

    # Use the DebyeModel to accurately compute the ATF for the microscope.  We use this in
    # conjunction with the FFT-based defocus calculation below, making the overall computation
    # of the PSf more accurate.
    debye_model = DebyeModel(recipe, sim_size_m, sim_size_px)
    apsf_z0 = apsf(debye_model, (0., 0., 0.))
    if recipe.normalize_intensity
        apsf_z0 *= sqrt(1.0 / maximum(abs2(apsf_z0)))
    end
    microscope_atf = im_to_ap_xform(WaveOptics.fftpad(apsf_z0))

    # If a static aperture mask has been supplied, include it in the model here.
    if !isa(recipe.static_aperture_mask, NullCodedAperture)
        static_ap_atf = mask(recipe.static_aperture_mask, aperture_plane_coordinates)
        microscope_atf = microscope_atf .* static_ap_atf
    end

    # Intensities can be scaled further by the user by setting this value explicitly.
    user_intensity_scaling_factor = 1.0

    return DefocusModel(recipe, sim_size_m, sim_size_px,
                        padding, num_observations,
                        aperture_plane_coordinates,
                        microscope_atf,
                        user_intensity_scaling_factor)
end

function na_limit!(model::DefocusModel, pupil::AbstractArray{Complex{T}}) where T<:AbstractFloat
    return (model.aperture_plane_coordinates.rho .<= 1.0) .* pupil
end

function na_limit!(rho::AbstractArray{T,2}, pupil::AbstractArray{Complex{T}}) where T<:AbstractFloat
    return (rho .<= 1.0) .* pupil
end


function num_parameters(model::DefocusModel)
    return num_parameters(model.recipe.dynamic_aperture_mask)
end



"""
Set the total number of photons for an in-focus PSF of an equivalent wide field microsocpe
model to be equal to the specified number of total_photons.  This helps to remove the effects
of simulation pixel size and can be used to achive similar intensity scaling across
very different PSF models.  This is especially useful for generating consistent Fisher
information results.
"""
function find_intensity_scaling_factor(model::PointSpreadFunctionModel, total_photons::T) where T<:Real

    # Generate an equivalent widefield model
    recipe = model.recipe
    wf_recipe = WaveOptics.WideFieldOpticalRecipe(recipe.objective_mag, recipe.objective_na, recipe.wavelength, recipe.medium_index, recipe.f_tubelens)
    wf_model = WaveOptics.DefocusModel(wf_recipe, model.sim_size_m, model.sim_size_px);

    # Find the areaof a simulation pixel in um^2
    #sim_pixel_area_um_squared = (model.sim_size_m / model.sim_size_px / 1000)^2
    #area_scaling_factor = 1. / sim_pixel_area_um_squared

    # Find the max intensity without scaling
    WaveOptics.set_intensity_scaling_factor!(wf_model, 1.0)
    psf0 = WaveOptics.psf(wf_model, (0,0,0))

#    alpha = photons_per_um_squared / (maximum(psf0) * area_scaling_factor)
    alpha = total_photons / (sum(psf0))# * area_scaling_factor)

    # Check the result
    WaveOptics.set_intensity_scaling_factor!(wf_model, alpha)
    psf1 = WaveOptics.psf(wf_model, (0,0,0))

    println("Adjusting intensity scaling factor...")
    println("   Widefield PSF sum before: $(sum(psf0))")
    println("   After: $(sum(psf1))")
    println("   Intensity scaling factor: $alpha.")
    return alpha
end

"""
Set a scalar multiplier that is used to adjust the final intensity pattern generate
by the PSF model.  This multiplier can be found using find_intensity_scaling_factor()
above.
"""
function set_intensity_scaling_factor!(model::DefocusModel, f::Float64)
    model.user_intensity_scaling_factor = f
end

function atf(model::DefocusModel, p::NTuple{3,T}, theta = Float64[]; na_limit::Bool = true, apply_tilt::Bool = true) where T<:Real
    x,y,z = p  # Unpack p

    # Generate a defocus ATF at the back aperture plane. This is a more
    # direct and accurate method of computing the defocus ATF than the
    # spherical wavefront and FFT method above.
    defocus_atf = defocus_mask(z,
                               model.aperture_plane_coordinates.rho,
                               model.aperture_plane_coordinates.theta,
                               model.recipe.objective_na,
                               model.recipe.wavelength,
                               model.recipe.medium_index)

    # Compute back aperture simulation size directly
    fourier_size_m = model.sim_size_px/model.sim_size_m * model.recipe.f_tubelens * model.recipe.wavelength

    # Compute the apodization appropriate to account for an objective with the abbe
    # sine correction. This would be appropriate for a plan corrected objective.
    #        microscope_atf = abbe_sine_apodization(model.aperture_plane_coordinates.rho, model.aperture_plane_coordinates.theta, model.recipe.objective_na, model.recipe.medium_index)
    intensity = model.user_intensity_scaling_factor
    microscope_atf = sqrt(intensity) .* model.microscope_atf  # Use Debye model for microscope ATF

    # If the user has supplied an aperture code, we add it to the PSF here.
    if apply_tilt && ((x != 0.) || (y != 0.))
        translation_atf = translation_mask(x, y, model.aperture_plane_coordinates.fx, model.aperture_plane_coordinates.fy, model.recipe.objective_na, model.recipe.wavelength)
        full_atf = microscope_atf .* defocus_atf .* translation_atf
    else
        full_atf = microscope_atf .* defocus_atf
    end

    if !isa(model.recipe.dynamic_aperture_mask, NullCodedAperture)
        ap_atf = mask(model.recipe.dynamic_aperture_mask, model.aperture_plane_coordinates, theta)
        full_atf .= full_atf .* ap_atf
    end

    if na_limit
        return na_limit!(model, full_atf)
    else
        return full_atf
    end
end

function otf_2d(model::DefocusModel, p, theta = Float64[]; return_freqs = false)
    full_atf = atf(model, p, theta, apply_tilt = true, na_limit = true)

    # Compute the normalized autocorrelation using an FFT method.
    A = fftshift(fft(fftshift(fftpad(full_atf))))
    B = fftunpad(fftshift(ifft(fftshift(A.*conj(A)))))

    normalization = sum(full_atf .* conj(full_atf))
    norm_autocorr = B ./ normalization

    if return_freqs
        sample_period_x = model.sim_size_m / (model.sim_size_px * model.padding)
        freqs = fftshift(fftfreq(model.sim_size_px * model.padding, sample_period_x))
        return norm_autocorr, freqs
    else
        return norm_autocorr
    end
end


"""
Compute the amplitude point spread function.

**Parameters**

    p - position of the point source

    theta - the parameters to pass to the aperture_mask.  If theta = 'None', the aperture's default
            parameters will be used.

**Returns**

    apsf - 2D image of the APSF on the image sensor
"""
function apsf(model::DefocusModel, p, theta = Float64[])
    x,y,z = p  # Unpack p

    # Don't apply tilt in the back aperture, as we will opt to instead apply
    # this as a lateral translation in the image plane.
    full_atf = atf(model, p, theta, na_limit = false, apply_tilt = false)

    # Transform back to get the field at the image plane. Restrict to
    # circular aperture using the na_limit() function.
    fourier_size_m = model.sim_size_px/model.sim_size_m * model.recipe.f_tubelens * model.recipe.wavelength
    camera_size_m, apsf = ap_to_im(na_limit!(model, full_atf), fourier_size_m, model.recipe.f_tubelens, model.recipe.wavelength)
    apsf = fftshift(apsf)

    # Unpad the FFT, and return the full result. The translate_apsf() function
    # here translates the point in x and y, returning an interpolated result.
    if x != 0.0 || y != 0.0
        return translate_apsf(model.sim_size_m, model.sim_size_px, WaveOptics.fftunpad(apsf), x, y)
    else
        return WaveOptics.fftunpad(apsf)
    end
end


function psf_3d(model::DefocusModel, zrange::AbstractArray{T,1}) where T<:Real
    # Transform back to get the field at the image plane. Restrict to
    # circular aperture using the na_limit() function.
    fourier_size_m = model.sim_size_px / model.sim_size_m * model.recipe.f_tubelens * model.recipe.wavelength

    c_rho = model.aperture_plane_coordinates.rho
    c_theta = model.aperture_plane_coordinates.theta
    microscope_atf = model.microscope_atf
    c_rho = ThreeDeconv.to_gpu_or_not_to_gpu(c_rho)
    c_theta = ThreeDeconv.to_gpu_or_not_to_gpu(c_theta)
    microscope_atf = ThreeDeconv.to_gpu_or_not_to_gpu(microscope_atf)

    M,N = size(c_rho)

    # Compute back aperture simulation size directly
    fourier_size_m = model.sim_size_px / model.sim_size_m * model.recipe.f_tubelens * model.recipe.wavelength

    # Compute the apodization appropriate to account for an objective with the abbe
    # sine correction. This would be appropriate for a plan corrected objective.
    #        microscope_atf = abbe_sine_apodization(model.aperture_plane_coordinates.rho, model.aperture_plane_coordinates.theta, model.recipe.objective_na, model.recipe.medium_index)
    intensity = model.user_intensity_scaling_factor
    microscope_atf = sqrt(intensity) .* microscope_atf  # Use Debye model for microscope ATF

    psf3d = zeros(Float32, (M, N, length(zrange)))
    for (idx, z) in enumerate(zrange)
        # Don't apply tilt in the back aperture, as we will opt to instead apply
        # this as a lateral translation in the image plane.

        # Generate a defocus ATF at the back aperture plane. This is a more
        # direct and accurate method of computing the defocus ATF than the
        # spherical wavefront and FFT method above.
        defocus_atf = defocus_mask(z,
                                   c_rho,
                                   c_theta,
                                   model.recipe.objective_na,
                                   model.recipe.wavelength,
                                   model.recipe.medium_index)

        # If the user has supplied an aperture code, we add it to the PSF here.
        full_atf = microscope_atf .* defocus_atf
        camera_size_m, apsf = ap_to_im(na_limit!(c_rho, full_atf), fourier_size_m, model.recipe.f_tubelens, model.recipe.wavelength)
        psf3d[:,:,idx] .= collect(abs2.(apsf))
    end

    psf3d = fftshift(psf3d, [1,2])

    # Unpad the FFT, and return the full result. The translate_apsf() function
    # here translates the point in x and y, returning an interpolated result.
    return WaveOptics.fftunpad2(psf3d)
end
