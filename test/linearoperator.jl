using Test, LinearAlgebra
using ThreeDeconv

function adjointnessCheck(A::ThreeDeconv.LinearOperator)
    x = randn(eltype(A), A.input_shape)
    forward_output = A.forward(x)

    y = randn(eltype(A), A.output_shape)
    adjoint_output = A.adjoint(y)

    return dot(forward_output, y) ≈ dot(x, adjoint_output)
end

function adjointnessCheck(A::ThreeDeconv.StackedLinearOperator)
    x = randn(A.input_shape)
    forward_output = vcat((v[:] for v in A.forward(x))...)

    y = [randn(output_shape) for output_shape in A.output_shape]
    adjoint_output = A.adjoint(y)

    return dot(forward_output, vcat(y...)) ≈ dot(x, adjoint_output)
end

@testset "linearoperator" begin
    A = randn(12, 10)
    LinOpA = ThreeDeconv.LinearOperator(
        (10, 1),
        (12, 1),
        10,
        12,
        x -> A * x,
        y -> A' * y,
        false,
        Float64,
    )
    @test adjointnessCheck(LinOpA)
    @test opnorm(A) ≈ ThreeDeconv.opnorm(LinOpA)

    B = randn(12, 10)
    LinOpB = ThreeDeconv.LinearOperator(
        (10, 1),
        (12, 1),
        10,
        12,
        x -> B * x,
        y -> B' * y,
        false,
        Float64,
    )

    LinOpK = ThreeDeconv.StackedLinearOperator([LinOpA, LinOpB])
    K = vcat(A, B)
    @test adjointnessCheck(LinOpK)
    @test opnorm(A) ≈ ThreeDeconv.opnorm(LinOpA)
    @test opnorm(B) ≈ ThreeDeconv.opnorm(LinOpB)
    @test opnorm(K) ≈ ThreeDeconv.opnorm(LinOpK)

end
