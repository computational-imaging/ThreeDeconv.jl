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
#              IMAGE/OBJECT PLANE TO APERTURE PLANE FUNCTIONS
#
# Using the FFT, these functions model the Fourier transforming properties of
# the objective lens and tube lens in a microscope configured as a 4-f system.
# The FFTs here are accelerated using the FFTW library, if it is available.
# -----------------------------------------------------------------------------

"""
Helper function to quickly transform from the image plane to the aperture
plane via a Fourier Transform.
"""
function im_to_ap(im::S, input_size_m::Float64, focal_len::Float64, wavelength::Float64)::Tuple{Float64,S} where S<:AbstractArray{T,2} where T<:Number

    # A lens Fourier transform may magnify or demagnify the image depending on the focal length.
    sim_size_px = size(im,1)
    delta_x = input_size_m / sim_size_px
    Lf = 1. / delta_x;
    delta_f = 1. / input_size_m;

    output_size_m = focal_len * wavelength * Lf

    # In the FFT, the forward transform is typically unscaled and the inverse
    # tranform is scaled by 1/length(x).  However, for a unitary transform that
    # scales equally in the forward and inverse directions, we can use the
    # unscaled inverse FFT ( bfft() ) and apply the scaling factor of
    # 1/sqrt(length(x)) ourselves.
    unitary_scaling_factor::T = T(1/sqrt(length(im)))
    return (output_size_m, unitary_scaling_factor .* fftshift(fft(ifftshift(im))))
end


"""
Helper function to quickly transform from the aperture plane to the image
plane via a Fourier Transform.
"""
function ap_to_im(ap::S, input_size_m::Float64, focal_len::Float64, wavelength::Float64)::Tuple{Float64,S} where S<:AbstractArray{T,2} where T<:Number

    # A lens Fourier transform may magnify or demagnify the image depending on the focal length.
    sim_size_px = size(ap,1)
    delta_x = input_size_m / sim_size_px
    Lf = 1. / delta_x;
    delta_f = 1. / input_size_m;

    output_size_m = focal_len * wavelength * Lf

    # In the FFT, the forward transform is typically unscaled and the inverse
    # tranform is scaled by 1/length(x).  However, for a unitary transform that
    # scales equally in the forward and inverse directions, we can use the
    # unscaled inverse FFT ( bfft() ) and apply the scaling factor of
    # 1/sqrt(length(x)) ourselves.
    unitary_scaling_factor = 1/sqrt(prod(size(ap)))
    out = unitary_scaling_factor .* bfft(ap)
    return (output_size_m, out)
end

function ap_to_im(ap::S, input_size_m::Float64, focal_len::Float64, wavelength::Float64)::Tuple{Float64,S} where S<:AbstractArray{T,3} where T<:Number

    # A lens Fourier transform may magnify or demagnify the image depending on the focal length.
    sim_size_px = size(ap,1)
    delta_x = input_size_m / sim_size_px
    Lf = 1. / delta_x;
    delta_f = 1. / input_size_m;

    output_size_m = focal_len * wavelength * Lf

    # In the FFT, the forward transform is typically unscaled and the inverse
    # tranform is scaled by 1/length(x).  However, for a unitary transform that
    # scales equally in the forward and inverse directions, we can use the
    # unscaled inverse FFT ( bfft() ) and apply the scaling factor of
    # 1/sqrt(length(x)) ourselves.
    unitary_scaling_factor = 1/sqrt(prod(size(ap)[1:2]))
    out = unitary_scaling_factor .* bfft(ap,[1,2])
    return (output_size_m, out)
end


"""
Simpler version that doesn't return output scaling (useful if you just want to perform an optical FFT)
"""
function im_to_ap_xform(im::AbstractArray{T,2}) where T<:Number
    output_size_m, ap = im_to_ap(im, 1., 1., 1.)
    return ap
end

"""
Simpler version that doesn't return output scaling (useful if you just want to perform an optical FFT)
"""
function ap_to_im_xform(ap::AbstractArray{T,2}) where T<:Number
    output_size_m, im = ap_to_im(ap, 1., 1., 1.)
    return im
end
