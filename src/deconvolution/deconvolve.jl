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


include("types.jl")
include("optimize.jl")
include("admm.jl")
include("setup_admm.jl")

"""
    deconvolve(img, psf, γ, σ, reg, method, options)

Deconvolve a focal stack image (img). A 3D psf, camera gain and readnoise variance
have to be known.

The detail of the algorithm is available from
"A convex 3D deconvolution algorithm for low photon count fluorescence imaging"
Scientific Reports 8, Article number: 11489 (2018)
Hayato Ikoma, Michael Broxton, Takamasa Kudo, Gordon Wetzstein
"""
function deconvolve(
    img::Array{T,3}, # 3D captured image [width x height x depth]
    psf::Array{Float32,3}, # 3D psf
    γ::Real, # camera gain
    σ::Real, # camera readnoise std
    reg::Real, # regularization parameter
    method::ADMM=ADMM();
    options::DeconvolutionOptions
    ) where T<:Real
    
    @assert all(.!isnan.(img))
    @assert γ > 0
    @assert σ > 0

    I,J,K = size(img)

    # Convert the unit to photoelectron number
    std_img = img ./ γ
    std_σ = σ / γ

    if isnan(method.ρ)
        method.ρ = 6000. * reg / maximum(std_img)
    end

    optimizer = setup_optimizer(method, Float32.(std_img), psf, Float32(σ^2), Float32(reg), Float32(method.ρ))
    result = optimize(optimizer, options)
    result.x = collect(result.x[1:I,1:J,1:K])

    # Release GPU memory
    gc()

    return result
end
