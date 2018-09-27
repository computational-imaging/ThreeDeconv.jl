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


"""
In order to avoid aliasing artifacts, the wave optics simulation must be
carried out at a resolution close to the diffraction limit. However, image
sensors sometimes undersample as compared to this limit, thus we may need to
downsample here to reach the resolution of the image sensor.

This utility function performs a downsampling by averaging together oversampling
x oversampling square blocks of pixels. This actually produces some aliasing,
but this aliasing should be present in the real optical system as well.

TODO: someday we could add pixel fill factor as a parameter to this function.
"""
function downsample_to_sensor_resolution(psf_image::AbstractArray{T,2}, oversampling::Int64) where T<:Real
    if (oversampling == 1)
        return psf_image
    end

    # We must ensure that the image has an even number of pixels, so we pad
    # (on the bottom and right) with a row of black pixels if necessary to
    # ensure the image has an even number of pixels in each linear dimension.
    trim_r = size(psf_image, 1) % oversampling
    trim_c = size(psf_image, 2) % oversampling
    trimmed_image = view(psf_image, 1:size(psf_image,1)-trim_r, 1:size(psf_image,2)-trim_c)
    height, width = size(trimmed_image)

    result = zeros(T, (div(height, oversampling), div(width, oversampling)))
    for r in 1:oversampling
        for c in 1:oversampling
            result .= result .+ trimmed_image[r:oversampling:end, c:oversampling:end]
        end
    end

    return result
end

function downsample_to_sensor_resolution(psf_image::AbstractArray{T,3}, oversampling::Int64) where T<:Real
    if (oversampling == 1)
        return psf_image
    end

    # We must ensure that the image has an even number of pixels, so we pad
    # (on the bottom and right) with a row of black pixels if necessary to
    # ensure the image has an even number of pixels in each linear dimension.
    I,J,K = size(psf_image)
    trim_r = I % oversampling
    trim_c = J % oversampling
    trimmed_image = view(psf_image, 1:I-trim_r, 1:J-trim_c, 1:K)
    height, width = size(trimmed_image)

    result = zeros(T, (div(height, oversampling), div(width, oversampling), K))
    @inbounds for r in 1:oversampling
        for c in 1:oversampling
            result .+= trimmed_image[r:oversampling:end, c:oversampling:end, :]
        end
    end

    return result
end
