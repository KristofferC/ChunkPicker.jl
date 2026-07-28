using ChunkPicker
using DifferentiationInterface: AutoForwardDiff
import DifferentiationInterface as DI
using ForwardDiff
using HyperHessians
using Test

const SECS = 0.01  # keep the suite fast; we assert on results, not on timings

f = x -> sum(abs2, x) + exp(sum(x))          # R^n -> R
g = x -> [sum(sin, x), prod(x), sum(x .^ 3)]  # R^n -> R^m

@testset "ChunkPicker" begin
    @testset "AutoForwardDiff $op" for op in (:gradient, :jacobian, :hessian, :hvp)
        h = op === :jacobian ? g : f
        n = 6
        x = rand(n)
        res = pick_chunk(AutoForwardDiff(), h, x; op, seconds = SECS, verbose = false)
        @test res isa ChunkPicker.ChunkPickResult
        @test res.kind === :chunk
        @test !res.simd
        @test res.chunk in 1:n
        @test length(res.timings) == n
        @test all(t -> t.kind === :chunk && !t.simd, res.timings)
        @test occursin("AutoForwardDiff(chunksize = $(res.chunk))", res.recommendation)

        # the winning chunk must give the correct derivative through DI
        b = AutoForwardDiff(chunksize = res.chunk)
        if op === :gradient
            @test DI.gradient(h, b, x) ≈ ForwardDiff.gradient(h, x)
        elseif op === :jacobian
            @test DI.jacobian(h, b, x) ≈ ForwardDiff.jacobian(h, x)
        elseif op === :hessian
            @test DI.hessian(h, b, x) ≈ ForwardDiff.hessian(h, x)
        else
            v = ones(n)
            @test DI.hvp(h, b, x, (v,))[1] ≈ ForwardDiff.hessian(h, x) * v
        end
    end

    @testset "AutoForwardDiff hvp custom tangents" begin
        n = 6
        x = rand(n)
        v = rand(n)
        res = pick_chunk(AutoForwardDiff(), f, x; op = :hvp, tangents = v, seconds = SECS, verbose = false)
        @test res.chunk in 1:n
        res2 = pick_chunk(AutoForwardDiff(), f, x; op = :hvp, tangents = (v, 2 .* v), seconds = SECS, verbose = false)
        @test res2.chunk in 1:n
    end

    @testset "AutoForwardDiff bad op" begin
        @test_throws ArgumentError pick_chunk(AutoForwardDiff(), f, rand(3); op = :nope, verbose = false)
    end

    @testset "custom chunks" begin
        res = pick_chunk(AutoForwardDiff(), f, rand(8); op = :gradient, chunks = [2, 4, 8], seconds = SECS, verbose = false)
        @test [t.chunk for t in res.timings] == [2, 4, 8]
        @test res.chunk in (2, 4, 8)
    end

    has_simd = hasmethod(
        HyperHessians.HessianConfig,
        Tuple{Vector{Float64}, HyperHessians.Chunk{1}}, (:simd,),
    )

    @testset "HyperHessians hessian" begin
        n = 6
        x = rand(n)
        # the ext uses the same hasmethod probe; assert the probe agrees with
        # actually constructing a simd config so a broken probe fails loudly
        probe_works = try
            HyperHessians.HessianConfig(x, HyperHessians.Chunk{2}(); simd = true)
            true
        catch
            false
        end
        @test has_simd == probe_works
        res = pick_chunk(HyperHessiansBackend(), f, x; seconds = SECS, verbose = false)
        @test res isa ChunkPicker.ChunkPickResult
        @test res.op === :hessian
        chunk_timings = filter(t -> t.kind === :chunk, res.timings)
        @test length(chunk_timings) == (has_simd ? 2n : n)
        @test res.chunk in 1:n
        # winning config computes the correct Hessian
        @test HyperHessians.hessian(f, x) ≈ ForwardDiff.hessian(f, x)

        # Jet variant is present iff the loaded HyperHessians provides it
        if isdefined(HyperHessians, :Jet)
            @test any(t -> t.kind === :jet, res.timings)
        else
            @test all(t -> t.kind === :chunk, res.timings)
        end

        # SIMD variants are present iff the loaded HyperHessians supports them
        if has_simd
            @test count(t -> t.simd, res.timings) == n
            @test res.simd == occursin("simd = true", res.recommendation)
            # simd configs compute the correct Hessian regardless of which
            # variant happened to win the (noisy) benchmark
            for T in (Float64, Float32)
                xt = T.(x)
                cfg = HyperHessians.HessianConfig(xt, HyperHessians.Chunk{2}(); simd = true)
                @test HyperHessians.hessian(f, xt, cfg) ≈ ForwardDiff.hessian(f, xt) rtol = sqrt(eps(T))
            end
            # simd variants can be turned off without dropping the plain ones
            res2 = pick_chunk(HyperHessiansBackend(), f, x; seconds = SECS, simd = false, verbose = false)
            @test all(t -> !t.simd, res2.timings)
            @test count(t -> t.kind === :chunk, res2.timings) == n
        else
            @test all(t -> !t.simd, res.timings)
        end

        # non-float eltypes never get simd variants
        resint = pick_chunk(HyperHessiansBackend(), f, rand(1:5, 4); seconds = SECS, verbose = false)
        @test all(t -> !t.simd, resint.timings)
    end

    @testset "HyperHessians hvp" begin
        n = 6
        x = rand(n)
        v = rand(n)
        has_simd_hvp = hasmethod(
            HyperHessians.HVPConfig,
            Tuple{Vector{Float64}, Vector{Float64}, HyperHessians.Chunk{1}}, (:simd,),
        )
        res = pick_chunk(HyperHessiansBackend(), f, x; op = :hvp, tangents = v, seconds = SECS, verbose = false)
        @test res.op === :hvp
        @test all(t -> t.kind === :chunk, res.timings) # no Jet for hvp
        @test length(res.timings) == (has_simd_hvp ? 2n : n)
        @test occursin("HVPConfig", res.recommendation)
        if has_simd_hvp
            for simd in (false, true)
                cfg = HyperHessians.HVPConfig(x, v, HyperHessians.Chunk{res.chunk}(); simd)
                @test HyperHessians.hvp(f, x, v, cfg) ≈ ForwardDiff.hessian(f, x) * v
            end
        end
        # default tangents and bundles work
        res_def = pick_chunk(HyperHessiansBackend(), f, x; op = :hvp, seconds = SECS, verbose = false)
        @test res_def.op === :hvp
        res_bundle = pick_chunk(
            HyperHessiansBackend(), f, x;
            op = :hvp, tangents = (v, 2 .* v), seconds = SECS, verbose = false,
        )
        @test res_bundle.op === :hvp
    end

    @testset "HyperHessians rejects unsupported ops" begin
        @test_throws ArgumentError pick_chunk(HyperHessiansBackend(), f, rand(3); op = :gradient, verbose = false)
    end
end
