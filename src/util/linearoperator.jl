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


import Base.*, Base.eltype, Base.size
import LinearAlgebra.issymmetric, LinearAlgebra.adjoint
using LinearMaps, Arpack

abstract type AbstractLinearOperator end

function *(op::AbstractLinearOperator, v::AbstractArray)
    # @assert size(v) == op.input_shape
    return op.forward(v)
end

function opnorm(A::AbstractLinearOperator)
    f(x) = A.adjoint(A.forward(reshape(x, A.input_shape)))[:]
    AtA = LinearMap(f, f, A.input_size, A.input_size, issymmetric = true, isposdef = true)
    λ, _ = Arpack.eigs(AtA; nev = 1, which = :LM)
    return sqrt(λ[1])
end

function size(op::AbstractLinearOperator)
    return (op.output_size, op.input_size)
end

function size(op::AbstractLinearOperator, i::Integer)
    if i == 1
        return op.output_size
    elseif i == 2
        return op.input_size
    else
        error("Not implemented.")
    end
end

function eltype(op::AbstractLinearOperator)
    return op.eltype
end

function issymmetric(op::AbstractLinearOperator)
    return op.symmetric
end

mutable struct LinearOperator <: AbstractLinearOperator
    input_shape::Tuple
    output_shape::Tuple
    input_size::Int
    output_size::Int
    forward::Function
    adjoint::Function
    symmetric::Bool
    eltype::DataType
end

function LinearOperator(
    input_shape::Tuple,
    output_shape::Tuple,
    forward::Function,
    adjoint::Function,
    symmetric::Bool = false,
    eltype::DataType = Float32,
)
    LinearOperator(
        input_shape,
        output_shape,
        prod(s for s in input_shape),
        prod(s for s in output_shape),
        forward,
        adjoint,
        symmetric,
        eltype,
    )
end

function adjoint(op::LinearOperator)
    return LinearOperator(
        op.output_shape,
        op.input_shape,
        op.output_size,
        op.input_size,
        op.adjoint,
        op.forward,
        op.symmetric,
        op.eltype,
    )
end





mutable struct StackedLinearOperator <: AbstractLinearOperator
    linops::Vector{LinearOperator}
    input_shape::Tuple
    output_shape::Vector{Tuple}
    input_size::Int
    output_size::Int
    forward!::Function
    adjoint!::Function
    forward::Function
    adjoint::Function
    symmetric::Bool
end


function StackedLinearOperator(linops::AbstractVector{LinearOperator})

    @assert length(unique(linop.input_shape for linop in linops)) == 1
    input_shape = linops[1].input_shape
    output_shape = [linop.output_shape for linop in linops]

    input_size = linops[1].input_size
    output_size = sum(linop.input_size for linop in linops)

    forward! = function (input, output::Vector)
        # Might be better to change it to pmap
        for j = 1:length(output)
            output[j] .= linops[j].forward(input)
        end
    end

    adjoint! = function (input::Vector, output)
        copy!(
            output,
            sum(
                reshape(linops[j].adjoint(input[j]), size(output)) for j = 1:length(linops)
            ),
        )
    end

    forward_stack = function (input)
        output = [similar(input, sh) for sh in output_shape]
        forward!(input, output)
        return output
    end

    adjoint_stack = function (input)
        output = similar(input[1], input_shape)
        adjoint!(input, output)
        return output
    end

    return StackedLinearOperator(
        linops,
        input_shape,
        output_shape,
        input_size,
        output_size,
        forward!,
        adjoint!,
        forward_stack,
        adjoint_stack,
        false,
    )
end


function IdentityOperator(data_shape::Tuple)
    return LinearOperator(data_shape, data_shape, copy, copy, true)
end


# 3D Second-order differential operator
function SecondOrderDifferentialOperator3D(data_shape::NTuple{3,Int}, dir::NTuple{2,Int})
    I, J, K = data_shape
    diff = zeros(Float32, I, J, K)
    if dir == (1, 1)
        diff[end-1, 1, 1] = 1.0f0
        diff[end, 1, 1] = -2.0f0
        diff[1, 1, 1] = 1.0f0
    elseif dir == (2, 2)
        diff[1, end-1, 1] = 1.0f0
        diff[1, end, 1] = -2.0f0
        diff[1, 1, 1] = 1.0f0
    elseif dir == (3, 3)
        diff[1, 1, end-1] = 1.0f0
        diff[1, 1, end] = -2.0f0
        diff[1, 1, 1] = 1.0f0
    elseif dir == (1, 2) || dir == (2, 1)
        # Absorbing the factor √2 to this operator
        diff[end-1, end-1, 1] = sqrt(2.0f0)
        diff[end, 1, 1] = -sqrt(2.0f0)
        diff[1, end, 1] = -sqrt(2.0f0)
        diff[1, 1, 1] = sqrt(2.0f0)
    elseif dir == (1, 3) || dir == (3, 1)
        diff[end-1, 1, end-1] = sqrt(2.0f0)
        diff[end, 1, 1] = -sqrt(2.0f0)
        diff[1, 1, end] = -sqrt(2.0f0)
        diff[1, 1, 1] = sqrt(2.0f0)
    elseif dir == (2, 3) || dir == (3, 2)
        diff[1, end-1, end-1] = sqrt(2.0f0)
        diff[1, end, 1] = -sqrt(2.0f0)
        diff[1, 1, end] = -sqrt(2.0f0)
        diff[1, 1, 1] = sqrt(2.0f0)
    else
        error("Direction is wrong.")
    end

    diff = to_gpu_or_not_to_gpu(diff)

    dft_mat = rfft3_operator(diff)
    D = dft_mat * diff
    idft_mat = dft_mat'

    D_operator = LinearOperator(
        data_shape,
        data_shape,
        x -> idft_mat * (D .* (dft_mat * x)),
        x -> idft_mat * (conj.(D) .* (dft_mat * x)),
    )

    return D_operator, D
end
