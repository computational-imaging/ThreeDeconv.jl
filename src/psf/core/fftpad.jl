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
Apply zero padding on all sides of an image. This keeps the image relatively
well centered even after zero padding, which is useful if you want to
avoiding a large tilt in the phase of the function after applying the
Fourier transform (i.e. via the Fourier shift property).  Such a tilt is
generally undesirable when using the FFT to model the Fourier transforming
property of an optical system.
"""
function fftpad(u1::AbstractArray{T,1}) where T
    padM = Int(ceil(size(u1,1)/2.))
    temp = zeros(eltype(u1), size(u1,1)*2)
    temp[padM+1:padM+size(u1,1)] .= u1
    return temp
end

function fftpad(u1::AbstractArray{T,2}) where T
    padM = Int(ceil(size(u1,1)/2.))
    padN = Int(ceil(size(u1,2)/2.))
    temp = zeros(eltype(u1), size(u1,1)*2, size(u1,2)*2)
    temp[padM+1:padM+size(u1,1), padN+1:padN+size(u1,2)] .= u1 # Zero pad the input
    return temp
end

function fftpad(u1::AbstractArray{T,3}) where T
    padM = Int(ceil(size(u1,1)/2.))
    padN = Int(ceil(size(u1,2)/2.))
    padK = Int(ceil(size(u1,3)/2.))
    temp = zeros(eltype(u1), size(u1,1)*2, size(u1,2)*2, size(u1,3)*2)
    temp[padM+1:padM+size(u1,1), padN+1:padN+size(u1,2), padK+1:padK+size(u1,3)] .= u1 # Zero pad the input
    return temp
end

"""
Remove symmetric zero padding.
"""
function fftunpad(u1::AbstractArray{T,1}) where T
    M = size(u1,1)
    origM = div(M, 2)
    padM = Int(ceil(origM/2.))
    return view(u1, padM+1:padM + origM)
end

function fftunpad(u1::AbstractArray{T,2}) where T
    (M,N) = size(u1)
    origM = div(M, 2)
    origN = div(N, 2)
    padM = Int(ceil(origM/2.))
    padN = Int(ceil(origN/2.))
    return view(u1, padM+1:padM + origM, padN+1:padN+origN)                 # remove zero padding
end

function fftunpad(u1::AbstractArray{T,3}) where T
    (M,N,K) = size(u1)
    origM = div(M, 2)
    origN = div(N, 2)
    origK = div(K, 2)
    padM = Int(ceil(origM/2.))
    padN = Int(ceil(origN/2.))
    padK = Int(ceil(origK/2.))
    return view(u1, padM+1:padM + origM, padN+1:padN+origN, padK+1:padK + origK)                 # remove zero padding
end

function fftunpad2(u1::AbstractArray{T,3}) where T
    (M,N,K) = size(u1)
    origM = div(M, 2)
    origN = div(N, 2)
    padM = Int(ceil(origM/2.))
    padN = Int(ceil(origN/2.))
    return view(u1, padM+1:padM + origM, padN+1:padN+origN, 1:K)                 # remove zero padding
end

#-------------------------------------------------------------------------------------
# Centered, optimal size fft padding routines

"""
Zero pads the FFT up to at least double the size of the original array, or larger
if a nearby larger dimension will have only small prime least common denominators.
Choosing the dimensions in this way can speed up the FFT algorithm in some cases.
"""
function fftpad_optimal(u1::AbstractArray{T,2}) where T
    optimal_shape = (nextprod([2,3,5,7], size(u1,1)*2),
                     nextprod([2,3,5,7], size(u1,2)*2))
    padM = Int(ceil(optimal_shape[1]/2. - size(u1,1)/2.))
    padN = Int(ceil(optimal_shape[2]/2. - size(u1,2)/2.))
    temp = zeros(eltype(u1), optimal_shape)
    temp[padM+2:padM+size(u1,1)+1, padN+2:padN+size(u1,2)+1] .= u1 # Zero pad the input in a fashion consistent with direct convolution
    return temp
end

"""
Removes optimal zero padding.  You must supply the original (non-zero padded)
array shape.
"""
function fftunpad_optimal(u1::AbstractArray{T,2}, orig_size::Tuple{Int64,Int64}) where T
    optimal_shape = (nextprod([2,3,5,7], orig_size[1]*2),
                     nextprod([2,3,5,7], orig_size[2]*2))
    origM,origN = orig_size
    padM = Int(ceil(optimal_shape[1]/2. - orig_size[1]/2.))
    padN = Int(ceil(optimal_shape[2]/2. - orig_size[2]/2.))
    return view(u1, padM+2:padM + origM+1, padN+2:padN+origN+1)                 # remove zero padding
end
