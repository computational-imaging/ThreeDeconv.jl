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


using FFTW, Statistics
import Base.copy!, LinearAlgebra.norm

function circconv(
    x::AbstractMatrix{T}, h1::AbstractVector{T}, h2::AbstractVector{T})::AbstractMatrix{T} where T<:Real
    N1, N2 = size(x)
    X = rfft(x)
    if h1 == [T(1)]
        h_circ2 = padcircshift(h2, N2)
        H_circ2 = fft(h_circ2)
        out = irfft( X  .* transpose(H_circ2), size(x,1))
    elseif h2 == [T(1)]
        h_circ1 = padcircshift(h1, N1)
        H_circ1 = rfft(h_circ1)
        out = irfft( X .* H_circ1, size(x,1))
    else
        h_circ1 = padcircshift(h1, N1)
        H_circ1 = rfft(h_circ1)
        h_circ2 = padcircshift(h2, N2)
        H_circ2 = fft(h_circ2)
        out = irfft((X .* H_circ1) .* transpose(H_circ2), size(x,1))
    end
    return out
end

function circconv(
    x::AbstractMatrix{T}, h::AbstractMatrix{T}) where T<:Real
    N1, N2 = size(x)
    k1, k2 = size(h)
    p1 = mod(k1,2) == 0 ? div(k1,2) : div(k1-1,2)
    p2 = mod(k2,2) == 0 ? div(k2,2) : div(k2-1,2)
    h_pad = zeros(T, N1, N2)
    h_pad[1:k1, 1:k2] = h
    h_circ = circshift(h_pad, (-p1, -p2))
    return irfft(rfft(x) .* rfft(h_circ), size(x,1))
end

function circconv(x::AbstractMatrix{T}, h::AbstractVector{T}) where T<:Real
    N1, N2 = size(x)
    X = rfft(x)
    h_circ1 = padcircshift(h, N1)
    H_circ1 = rfft(h_circ1)
    h_circ2 = padcircshift(h, N2)
    H_circ2 = fft(h_circ2)
    return irfft((X .* H_circ1) .* transpose(H_circ2), size(x,1))
end

circconv1dx(x::Matrix{T}, h::Vector{T}) where T = circconv(x, [1.0], h)
circconv1dy(x::Matrix{T}, h::Vector{T}) where T = circconv(x, h, [1.0])


function padcircshift(h::AbstractVector{T}, N::Integer) where T<:Real
  k = length(h)
  @assert N > k
  h_pad = zeros(T, N)
  h_pad[1:k] = h
  p = mod(k,2) == 0 ? div(k,2) : div(k-1,2)
  return circshift(h_pad, -p)
end


mad(v::Vector{T}, κ::Float64 = 0.6745) where T= 1.0 / κ * median(abs.(v))


function madBiasFactor(n::Int)::Float64
    lut = [0.798, 0.732, 0.712, 0.702, 0.696, 0.693, 0.690, 0.688, 0.686, 0.685]
    if n <= 20
        out = lut[div(n-1,2) + 1]
    else
        out = 1 / (5.0 * n) + 0.6745
    end
    return out
end


function center_zeropad2!(
    padded_u::AbstractArray{T,2}, u::AbstractArray{T,2}) where T
    M, N = size(u)
    m1 = isodd(M) ? div(M+1,2) : div(M,2)
    m2 = M - m1
    n1 = isodd(N) ? div(N+1,2) : div(N,2)
    n2 = N - n1

    padded_u[1:m1, 1:n1] = u[1:m1, 1:n1]
    padded_u[1:m1, end-n2+1:end] = u[1:m1, end-n2+1:end]
    padded_u[end-m2+1:end, 1:n1] = u[end-m2+1:end, 1:n1]
    padded_u[end-m2+1:end, end-n2+1:end] = u[end-m2+1:end, end-n2+1:end]
end


function center_zeropad2(u::AbstractArray{T,2}, padded_shape::NTuple{2,Int}) where T

    padded_u = zeros(eltype(u), padded_shape)
    center_zeropad2!(padded_u, u)

    return padded_u
end


function center_zeropad2(
    u::AbstractArray{T,3}, padded_shape::NTuple{3,Int}) where T
    @assert size(u,3) == padded_shape[3]
    K = size(u,3)
    padded_u = zeros(eltype(u), padded_shape)

    for k in 1:K
        center_zeropad2!(view(padded_u,:,:,k), view(u,:,:,k))
    end

    return padded_u
end


function center_zeropad3!(
    padded_u::AbstractArray{T,3}, u::AbstractArray{T,3}) where T
    I, J, K = size(u)
    i1 = isodd(I) ? div(I+1,2) : div(I,2)
    i2 = I - i1
    j1 = isodd(J) ? div(J+1,2) : div(J,2)
    j2 = J - j1
    k1 = isodd(K) ? div(K+1,2) : div(K,2)
    k2 = K - k1

    padded_u[1:i1, 1:j1, 1:k1] = u[1:i1, 1:j1, 1:k1]
    padded_u[1:i1, end-j2+1:end, 1:k1] = u[1:i1, end-j2+1:end, 1:k1]
    padded_u[end-i2+1:end, 1:j1, 1:k1] = u[end-i2+1:end, 1:j1, 1:k1]
    padded_u[end-i2+1:end, end-j2+1:end, 1:k1] = u[end-i2+1:end, end-j2+1:end, 1:k1]
    padded_u[1:i1, 1:j1, end-k2+1:end] = u[1:i1, 1:j1, end-k2+1:end]
    padded_u[1:i1, end-j2+1:end, end-k2+1:end] = u[1:i1, end-j2+1:end, end-k2+1:end]
    padded_u[end-i2+1:end, 1:j1, end-k2+1:end] = u[end-i2+1:end, 1:j1, end-k2+1:end]
    padded_u[end-i2+1:end, end-j2+1:end, end-k2+1:end] = u[end-i2+1:end, end-j2+1:end, end-k2+1:end]
end


function center_zeropad3(
    u::AbstractArray{T,3}, padded_shape::NTuple{3,Int}) where T

    padded_u = zeros(eltype(u), padded_shape)
    center_zeropad3!(padded_u, u)

    return padded_u
end


norm(x::Vector{AbstractArray{T,N}}) where T where N = sqrt(sum(sum(abs2, ele) for ele in x))

function copy!(dest::Vector{T}, src::Vector{T}) where T<:AbstractArray{S,N} where S where N
    @assert length(dest) == length(src)
    for (d,s) in zip(dest, src)
        copy!(d, s)
    end
end

function block_soft_threshold(x::T, norm_x::T, κ::T)::T where T<:AbstractFloat
    return x * max(T(1.0) - T(κ) / norm_x, T(0))
end
