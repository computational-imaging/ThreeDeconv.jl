# __BEGIN_LICENSE__
#
# ThreeDeconv.jl
#
# Copyright (c) 2018, Stanford University
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


using Printf

include("foi.jl")


"""
noise estimation
"""
function noise_estimation(img::AbstractArray;
                          maxnum_pairs::Int=300,
                          verbose::Bool=false)
    @assert sizeof(img) <= 10^9 * 8 # Prevent input image size more than 1GB

    offset = -minimum(img)
    subimg = img .+ offset
    scale = 1 / (maximum(subimg) * 1.1)
    scaled_img = Float32.(subimg.*scale)

    scaled_a, scaled_b = foi_noiseestimation(scaled_img, maxnum_pairs=maxnum_pairs, verbose=verbose)

    # Rescale back the estimated parameters.
    a = scaled_a / scale
    b = scaled_b / scale^2
    @printf "Estimated noise parameters are a = %4.3f, b = %4.3f.\n" a b

    γ_est = a
    σsq_est = a * offset + b

    if σsq_est < .0
        @printf "Noise estimation might have failed. σ⁠^2 = %4.3f.\n
                 Proceeding by setting σ^2 = .0 to assure the variance's
                 nonnegativity.\n" σsq_est
        σsq_est = .0 + 1e-10
    end
    σ_est = sqrt(σsq_est)
    println("Estimated parameters for Poisson-Gaussian noise model")
    println("γ: $γ_est, σ: $σ_est")
    return γ_est, σ_est
end
