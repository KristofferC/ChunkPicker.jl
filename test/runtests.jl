using ChunkPicker
using ForwardDiff
using HyperHessians
using Test

const SECS = 0.01  # keep the suite fast; we assert on results, not on timings

f = x -> sum(abs2, x) + exp(sum(x))          # R^n -> R
g = x -> [sum(sin, x), prod(x), sum(x .^ 3)]  # R^n -> R^m

@testset "ChunkPicker" begin
    @testset "ForwardDiff $op" for op in (:gradient, :jacobian, :hessian)
        h = op === :jacobian ? g : f
        n = 6
        x = rand(n)
        res = pick_chunk(ForwardDiffBackend(), h, x; op, seconds = SECS, verbose = false)
        @test res isa ChunkPicker.ChunkPickResult
        @test res.kind === :chunk
        @test res.chunk in 1:n
        @test length(res.timings) == n
        @test all(t -> t.kind === :chunk, res.timings)
        @test occursin("Chunk{$(res.chunk)}", res.recommendation)

        # the winning chunk must give the correct derivative
        if op === :gradient
            @test ForwardDiff.gradient(h, x, ForwardDiff.GradientConfig(h, x, ForwardDiff.Chunk{res.chunk}())) ≈
                ForwardDiff.gradient(h, x)
        elseif op === :jacobian
            @test ForwardDiff.jacobian(h, x, ForwardDiff.JacobianConfig(h, x, ForwardDiff.Chunk{res.chunk}())) ≈
                ForwardDiff.jacobian(h, x)
        else
            @test ForwardDiff.hessian(h, x, ForwardDiff.HessianConfig(h, x, ForwardDiff.Chunk{res.chunk}())) ≈
                ForwardDiff.hessian(h, x)
        end
    end

    @testset "ForwardDiff bad op" begin
        @test_throws ArgumentError pick_chunk(ForwardDiffBackend(), f, rand(3); op = :nope, verbose = false)
    end

    @testset "custom chunks" begin
        res = pick_chunk(ForwardDiffBackend(), f, rand(8); op = :gradient, chunks = [2, 4, 8], seconds = SECS, verbose = false)
        @test [t.chunk for t in res.timings] == [2, 4, 8]
        @test res.chunk in (2, 4, 8)
    end

    @testset "HyperHessians hessian" begin
        n = 6
        x = rand(n)
        res = pick_chunk(HyperHessiansBackend(), f, x; seconds = SECS, verbose = false)
        @test res isa ChunkPicker.ChunkPickResult
        @test res.op === :hessian
        chunk_timings = filter(t -> t.kind === :chunk, res.timings)
        @test length(chunk_timings) == n
        @test res.chunk in 1:n
        # winning config computes the correct Hessian
        @test HyperHessians.hessian(f, x) ≈ ForwardDiff.hessian(f, x)

        # Jet variant is present iff the loaded HyperHessians provides it
        if isdefined(HyperHessians, :Jet)
            @test any(t -> t.kind === :jet, res.timings)
        else
            @test all(t -> t.kind === :chunk, res.timings)
        end
    end

    @testset "HyperHessians rejects non-hessian ops" begin
        @test_throws ArgumentError pick_chunk(HyperHessiansBackend(), f, rand(3); op = :gradient, verbose = false)
    end
end
