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


function setup_optimizer(method::ADMM, y::Array{Float32,3}, psf::Array{Float32,3}, σsq::Float32, ν::Float32, ρ::Float32)
    num_clip = sum(y .< -σsq)
    if num_clip > 0
        ratio = num_clip / length(y) * 100
        println("Clipped voxels: $(ratio)%")
    else
        println("No clipping performed.")
    end

    y[y .< -σsq] .= -σsq
    y_shape = size(y)
    data_shape = size(psf)
    @assert all(y_shape .< data_shape)

    M, N, L = size(y)
    ypad = zeros(Float32, data_shape)
    ypad[1:M, 1:N, 1:L] .= y

    mask_in = zeros(Bool, data_shape)
    mask_in[1:M,1:N,1:L] .= true
    mask_out = .!mask_in

    κ = ν / ρ

    psf = fftshift(psf)

    ypad = to_gpu_or_not_to_gpu(ypad)
    mask_in = to_gpu_or_not_to_gpu(mask_in)
    mask_out = to_gpu_or_not_to_gpu(mask_out)
    psf = to_gpu_or_not_to_gpu(psf)

    H = rfft(psf)
    dft_mat = rfft3_operator(ypad)
    idft_mat = dft_mat'

    # Image formation operator
    K_1 = LinearOperator(data_shape, data_shape,
        x -> idft_mat * (H .* (dft_mat * x)),
        x -> idft_mat * (conj.(H) .* (dft_mat * x)), false, Float32)

    # Identity operator
    K_2 = IdentityOperator(data_shape)

    K_3, Dxx = SecondOrderDifferentialOperator3D(data_shape, (2,2))

    K_4, Dyy = SecondOrderDifferentialOperator3D(data_shape, (1,1))

    K_5, Dzz = SecondOrderDifferentialOperator3D(data_shape, (3,3))

    # Dxy + Dyx = √2 * Dxy = Dxy_2
    K_6, Dxy_2 = SecondOrderDifferentialOperator3D(data_shape, (1,2))

    K_7, Dxz_2 = SecondOrderDifferentialOperator3D(data_shape, (1,3))

    K_8, Dyz_2 = SecondOrderDifferentialOperator3D(data_shape, (2,3))

    K = StackedLinearOperator([K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8])

    update_x! = function(x::T, z::Vector{T}, λ::Vector{T}) where T<:AbstractArray{Float32,3}
        Z = [dft_mat * (z_j .- λ_j) for (z_j, λ_j) in zip(z, λ)]
        tmp = (conj.(H) .* Z[1] .+ Z[2] .+
               conj.(Dxx) .* Z[3] .+ conj.(Dyy) .* Z[4].+
               conj.(Dzz) .* Z[5] .+ conj.(Dxy_2) .* Z[6] .+
               conj.(Dxz_2) .* Z[7] .+ conj.(Dyz_2) .* Z[8]) ./
              (abs2.(H) .+ abs2.(Dxx) .+ abs2.(Dyy) .+ abs2.(Dzz) .+
               abs2.(Dxy_2) .+ abs2.(Dxz_2) .+ abs2.(Dyz_2) .+ 1.0f0)
        x .= idft_mat * tmp
    end

    update_z! = function(Kx::Vector{T}, z::Vector{T}, λ::Vector{T}) where T<:AbstractArray{Float32,3}
        v = [Kx[i] .+ λ[i] for i in 1:length(z)]
        tmp = (-1.0f0 .+ ρ .* (v[1] .- σsq)) ./ 2ρ
        z[1] .= mask_out .* v[1] .+ mask_in .* (tmp .+ sqrt.(max.(tmp.^2 .+ (ypad .+ (ρ * σsq) .* v[1]) ./ ρ, 0.0f0)))
        z[2] .= max.(v[2], 0.0f0)
        v_norm = sqrt.(max.(sum(v[i].^2 for i in 3:8), floatmin(Float32)))
        for i in 1:6
            z[i + 2] .= block_soft_threshold.(v[i + 2], v_norm, κ)
        end
    end

    optimizer = initialize_optimizer(method, K, update_x!, update_z!)
end
