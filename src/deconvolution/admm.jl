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

mutable struct ADMM <: Solver
    ρ::Float64
    ϵ_rel_diff::Float64
end

function ADMM(;
        ρ::Real = NaN,
        ϵ_rel_diff::Real = 1e-3) # before: 1e-3, for accuracy: 1e-5
    return ADMM(ρ, ϵ_rel_diff)
end

struct ADMMmetric <: SolverMetric
    iteration::Int
    rel_norm_diff::Float64
    elapsed_time::Float64
end

mutable struct ADMMstate{T<:AbstractArray{Float32,3}} <: SolverState
    x::T
    z::Vector{T}
    λ::Vector{T}
    Kx::Vector{T}
    x_prev::T
end

function ADMMstate(input_shape::NTuple, output_shape::Vector{Tuple})
    if CUDA.functional()
        dtype = CuArray{Float32}
    else
        dtype = Float32
    end
    x = zeros(dtype, input_shape)
    z = [zeros(dtype, sh) for sh in output_shape]
    λ = [zeros(dtype, sh) for sh in output_shape]
    Kx = [zeros(dtype, sh) for sh in output_shape]

    # Allocate an array for convergence check
    x_prev = zeros(dtype, input_shape)
    return ADMMstate(x, z, λ, Kx, x_prev)
end

function print_header(method::ADMM)
    @printf "Iter     rel_diff      time  \n"
    @printf "------   ---------   --------\n"
end


function Base.show(io::IO, s::ADMMmetric)
    @printf io "%6d   %9.3g   %8.2f\n" s.iteration s.rel_norm_diff s.elapsed_time
end

function initialize_optimizer(method::ADMM,
                              K::StackedLinearOperator,
                              update_x!::Function,
                              update_z!::Function)

    initial_state = ADMMstate(K.input_shape, K.output_shape)

    function update_λ!( Kx::Vector{T}, z::Vector{T}, λ::Vector{T}) where T<:AbstractArray{Float32,3}
        for (Kx_j, z_j, λ_j) in zip(Kx, z, λ)
            λ_j .+= Kx_j .- z_j
        end
    end

    function update_state!(state::ADMMstate)
        state.x_prev .= state.x
        update_x!(state.x, state.z, state.λ)
        K.forward!(state.x, state.Kx)
        update_z!(state.Kx, state.z, state.λ)
        update_λ!(state.Kx, state.z, state.λ)
    end


    function check_state(state::ADMMstate, iter::Int, tic::Float64)
        rel_norm_diff = norm(state.x - state.x_prev) / norm(state.x_prev)
        toc = time()
        return ADMMmetric(iter, rel_norm_diff, toc - tic)
    end

    function check_convergence(metric::ADMMmetric)
        return metric.rel_norm_diff < method.ϵ_rel_diff
    end

    return Optimizer(method, initial_state, update_state!, check_state, check_convergence)
end
