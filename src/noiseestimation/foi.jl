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


using QuadGK
import Optim

const ϵ_reg = 1e-6

struct ParametricNoiseModel
    m_hat::Vector{Float64}
    σ_hat::Vector{Float64}
    model::Function
    model_init::Function
    params::Any
    params_init::Any
end

struct localEstimation{T<:AbstractFloat}
    y::T
    σ::T
    κ::T
    n::Int
end

function foi_noiseestimation(
    z::AbstractArray{Float32};
    τ = 0.2,
    maxnum_pairs::Int,
    verbose::Bool = false,
)
    # τ = 0.2 is good to reject a lot of outliers.

    @assert maximum(z) <= 1.0
    @assert minimum(z) >= 0.0

    # Compute the local sample mean and standard deviation over a smooth region
    println("Computing local noise variance.")
    num_imgs = size(z, 3)
    z_stack = (z[:, :, i] for i = 1:num_imgs)
    est_pairs_array = map(x -> localnoiseestimation(x, τ)[1], z_stack)
    est_pairs = vcat(est_pairs_array...)

    println("Initializing parameters by least-squares.")
    ab_init, y_hat, σ_hat = initialize_parameters(est_pairs)

    num_pairs = length(est_pairs)
    if num_pairs > maxnum_pairs
        idx = rand(1:num_pairs, maxnum_pairs)
        est_pairs = est_pairs[idx]
    end

    println("Initialization done.")
    println("Starting likelihood maximization.")

    likelihood0(x::Vector{Float64}) = nonclipped_negloglikelihood(x, est_pairs)
    result = Optim.optimize(
        likelihood0,
        ab_init,
        Optim.NelderMead(),
        Optim.Options(show_trace = verbose),
    )

    println("Finished the maximization.")

    ab_hat = Optim.minimizer(result)
    @assert Optim.converged(result)

    return ab_hat
end


function localnoiseestimation(z::AbstractArray{Float32,2}, τ)
    # Wavelet and scaling functions used in the original paper are the followings:
    # ψ = Array{Float32}([0.035, 0.085, -0.135, -0.460, 0.807, -0.333])
    # ϕ = Array{Float32}([0.025, -0.060, -0.095, 0.325, 0.571, 0.235])
    ϕ = [
        0.035226291882100656f0,
        -0.08544127388224149f0,
        -0.13501102001039084f0,
        0.4598775021193313f0,
        0.8068915093133388f0,
        0.3326705529509569f0,
    ]
    ϕ ./= sum(ϕ)

    ψ = [
        -0.3326705529509569f0,
        0.8068915093133388f0,
        -0.4598775021193313f0,
        -0.13501102001039084f0,
        0.08544127388224149f0,
        0.035226291882100656f0,
    ]
    ψ ./= norm(ψ)

    σ_gauss = 1.2f0
    gauss = [exp(-x^2 / (2.0f0 * σ_gauss^2)) for x = -10.0f0:10.0f0]
    gauss ./= sum(gauss)

    z_wdet = circconv(z, ψ)[1:2:end, 1:2:end]
    z_wapp = circconv(z, ϕ)[1:2:end, 1:2:end]

    ω = ones(Float32, 7) ./ 7.0f0
    z_smo = circconv(z_wapp, ω, ω)
    s = sqrt(0.5 * π) .* circconv(abs.(z_wdet), ω, ω)

    g = [-0.5f0, 0.0f0, 0.5f0]
    smoothed_zwapp = circconv(z_wapp, gauss, gauss)
    dx_wapp = circconv(smoothed_zwapp, [1.0f0], g)
    dy_wapp = circconv(smoothed_zwapp, g, [1.0f0])

    x_smo = sqrt.(dx_wapp .^ 2 .+ dy_wapp .^ 2) .< τ .* s
    N = length(z_smo)
    num_bins = 300
    histogram_zwapp = [Vector{Float32}() for _ = 1:num_bins]
    histogram_zwdet = [Vector{Float32}() for _ = 1:num_bins]
    min_wapp = sum(ϕ[ϕ.<0.0f0])
    max_wapp = sum(ϕ[ϕ.>=0.0f0])
    Δ = (max_wapp - min_wapp) / num_bins

    for i = 1:N
        if x_smo[i]
            idx = floor(Int, (z_smo[i] - min_wapp) / Δ)
            push!(histogram_zwapp[idx], z_wapp[i])
            push!(histogram_zwdet[idx], z_wdet[i])
        end
    end

    est_pairs = Vector{localEstimation{Float64}}()
    for i = 1:num_bins
        if histogram_zwdet[i] != []
            n = length(histogram_zwdet[i])
            κ = madBiasFactor(n)
            σ = sqrt(max(Float64(mad(histogram_zwdet[i], κ))^2, 0.0))
            y = Float64(mean(histogram_zwapp[i]))
            push!(est_pairs, localEstimation(y, σ, κ, n))
        end
    end

    return est_pairs, x_smo
end


function initialize_parameters(est_pairs::Vector{localEstimation{T}}) where {T}
    num_pairs = length(est_pairs)

    y_hat = zeros(num_pairs)
    σ_hat = zeros(num_pairs)

    Φ_tmp = Vector{Float64}()
    v = Vector{Float64}()

    for i = 1:num_pairs
        tmp = est_pairs[i]
        y_hat[i] = tmp.y
        σ_hat[i] = tmp.σ
        push!(Φ_tmp, tmp.y)
        push!(v, tmp.σ^2)
    end
    Φ = hcat(Φ_tmp, ones(length(Φ_tmp)))
    ab0 = Φ \ v
    return ab0, y_hat, σ_hat
end


function nonclipped_negloglikelihood(
    ab::Vector{Float64},
    est_pairs::Vector{localEstimation{T}},
) where {T}

    Δ = 0.1
    total_val = 0.0

    for l in est_pairs
        c_i = 1.0 / l.n
        d_i = 1.35 / (l.n + 1.5)
        y_i = l.y
        σ_i = l.σ

        # This integrand is sometimes a very sharp peak-like function like a Dirac funciton.
        # Therefore, the direct numerical integration over [0.0, 1.0] is realatively difficult
        # because the algorithm may not evaluate the integrand around the peak.
        # To avoid this issue, the integration interval is separated into multiple intervals.
        # Since the peak is known to be close to y_i, the function value at y_i should be enough large than zero.
        integrand(y::Float64)::Float64 =
            1.0 / σsq_reg(y, ab) * exp(
                -1.0 / (2.0 * σsq_reg(y, ab)) *
                ((y_i - y)^2 / c_i + (σ_i - sqrt(σsq_reg(y, ab)))^2 / d_i),
            )

        val = 0.0
        if y_i - Δ > 0.0
            val += quadgk(integrand, 0.0, y_i - Δ)[1]
        end
        val += quadgk(integrand, max(0.0, y_i - Δ), y_i)[1]
        val += quadgk(integrand, y_i, min(1.0, y_i + Δ))[1]
        if y_i + Δ < 1.0
            val += quadgk(integrand, y_i + Δ, 1.0)[1]
        end

        total_val -= log(1 / (2π * sqrt(c_i * d_i)) * abs(val) + ϵ_reg)
    end

    return total_val
end


σsq_reg(y::Float64, ab::Vector{Float64}) = max(ϵ_reg^2, σsq(y, ab))
σsq(y::Float64, ab::Vector{Float64}) = ab[1] * y + ab[2]
