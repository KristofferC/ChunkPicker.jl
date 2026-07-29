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
        expected = op === :hessian ? ChunkPicker._smart_chunks_fd_hess(n) :
            op === :hvp ? ChunkPicker._smart_chunks_fd_hvp(n) :
            ChunkPicker._smart_chunks_fd_grad(n)
        @test [t.chunk for t in res.timings] == expected
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

    @testset "AutoHyperHessians $op" for op in (:hessian, :hvp)
        n = 6
        x = rand(n)
        res = pick_chunk(AutoHyperHessians(), f, x; op, seconds = SECS, verbose = false)
        @test res.chunk in 1:n
        @test [t.chunk for t in res.timings] == ChunkPicker.smart_chunks(n)
        @test all(t -> t.kind === :chunk && !t.simd, res.timings)
        @test occursin("AutoHyperHessians(chunksize = $(res.chunk))", res.recommendation)
        b = AutoHyperHessians(chunksize = res.chunk)
        if op === :hessian
            @test DI.hessian(f, b, x) ≈ ForwardDiff.hessian(f, x)
        else
            v = ones(n)
            @test DI.hvp(f, b, x, (v,))[1] ≈ ForwardDiff.hessian(f, x) * v
        end
    end

    @testset "AutoHyperHessians rejects first-order ops" begin
        @test_throws ArgumentError pick_chunk(AutoHyperHessians(), f, rand(3); op = :gradient, verbose = false)
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
    HHExt = Base.get_extension(ChunkPicker, :ChunkPickerHyperHessiansExt)
    has_jet = HHExt.HAS_JET
    has_jet_simd = HHExt.HAS_JET_SIMD

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
        nc = length(ChunkPicker.smart_chunks(n))
        @test length(chunk_timings) == (has_simd ? 2nc : nc)
        @test sort!(unique([t.chunk for t in chunk_timings])) == ChunkPicker.smart_chunks(n)
        @test res.chunk in 1:n
        # winning config computes the correct Hessian
        @test HyperHessians.hessian(f, x) ≈ ForwardDiff.hessian(f, x)

        # Jet variants are present iff the loaded HyperHessians provides them,
        # with a simd flavor when the jet config accepts the flag
        jet_timings = filter(t -> t.kind === :jet, res.timings)
        if has_jet
            @test length(jet_timings) == (has_jet_simd && has_simd ? 2 : 1)
            @test count(t -> t.simd, jet_timings) == (has_jet_simd && has_simd ? 1 : 0)
            for simd in (has_jet_simd ? (false, true) : (false,))
                cfg = simd ? HyperHessians.HessianConfig(x, HyperHessians.Jet; simd = true) :
                    HyperHessians.HessianConfig(x, HyperHessians.Jet)
                @test HyperHessians.hessian(f, x, cfg) ≈ ForwardDiff.hessian(f, x)
            end
            res_nojet = pick_chunk(HyperHessiansBackend(), f, x; seconds = SECS, jet = false, verbose = false)
            @test all(t -> t.kind === :chunk, res_nojet.timings)
            # the integer form caps the input length for the Jet candidate
            res_cap = pick_chunk(HyperHessiansBackend(), f, x; seconds = SECS, jet = n - 1, verbose = false)
            @test all(t -> t.kind === :chunk, res_cap.timings)
            res_forced = pick_chunk(
                HyperHessiansBackend(), f, x;
                seconds = SECS, jet = true, simd = false, verbose = false,
            )
            @test any(t -> t.kind === :jet, res_forced.timings)
        else
            @test isempty(jet_timings)
        end

        # SIMD variants are present iff the loaded HyperHessians supports them
        if has_simd
            @test count(t -> t.simd, res.timings) == nc + (has_jet_simd ? 1 : 0)
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
            @test count(t -> t.kind === :chunk, res2.timings) == nc
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
        nc = length(ChunkPicker.smart_chunks(n))
        @test length(res.timings) == (has_simd_hvp ? 2nc : nc)
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

    @testset "ForwardDiff smart sets" begin
        @test ChunkPicker._smart_chunks_fd_grad(3) == [1, 2, 3]
        @test ChunkPicker._smart_chunks_fd_grad(6) == [3, 4, 5, 6]
        @test ChunkPicker._smart_chunks_fd_grad(20) == [3, 4, 5, 8, 12, 16, 20]
        @test ChunkPicker._smart_chunks_fd_grad(100) == [3, 4, 5, 8, 12, 16, 24, 32]
        @test ChunkPicker._smart_chunks_fd_hess(6) == [2, 3, 4, 6]
        @test ChunkPicker._smart_chunks_fd_hess(48) == [4, 6, 7, 8, 10, 12, 16]
        @test ChunkPicker._smart_chunks_fd_hess(100) == [4, 5, 6, 8, 10, 12, 15, 16]
        @test ChunkPicker._smart_chunks_fd_hvp(32) == [2, 4, 5, 6, 7, 8, 11, 16, 32]
        @test ChunkPicker._smart_chunks_fd_hvp(100) == [2, 4, 8, 15, 16, 17, 20, 25]
        for gen in (ChunkPicker._smart_chunks_fd_grad, ChunkPicker._smart_chunks_fd_hess, ChunkPicker._smart_chunks_fd_hvp), n in 5:200
            cs = gen(n)
            @test length(cs) <= 12 && issorted(cs) && allunique(cs)
        end
    end

    @testset "smart_chunks" begin
        @test ChunkPicker.smart_chunks(1) == [1]
        @test ChunkPicker.smart_chunks(4) == [1, 2, 3, 4]
        @test ChunkPicker.smart_chunks(6) == [2, 3, 4, 6]
        @test ChunkPicker.smart_chunks(9) == [2, 3, 4, 5, 6, 8, 9]
        @test ChunkPicker.smart_chunks(16) == [2, 3, 4, 6, 8, 12, 16]
        @test ChunkPicker.smart_chunks(20) == [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]
        @test ChunkPicker.smart_chunks(100) == [2, 3, 4, 5, 6, 8, 10, 12, 16]
        # every set stays small even for awkward sizes
        for n in 5:200
            cs = ChunkPicker.smart_chunks(n)
            @test length(cs) <= 12 && all(c -> 1 <= c <= min(n, 16) || c == n, cs)
        end
        # Float32 raises the caps to 24 and drops 2 from the base set
        @test ChunkPicker.smart_chunks(9, Float32) == [3, 4, 5, 6, 8, 9]
        @test ChunkPicker.smart_chunks(24, Float32) == [3, 4, 6, 8, 12, 16, 24]
        @test ChunkPicker.smart_chunks(100, Float32) == [3, 4, 5, 6, 8, 10, 12, 16, 20]
        @test ChunkPicker.smart_chunks(6, Float64) == ChunkPicker.smart_chunks(6)
        for n in 5:200
            cs = ChunkPicker.smart_chunks(n, Float32)
            @test length(cs) <= 12 && all(c -> 1 <= c <= min(n, 24) || c == n, cs)
        end
    end

    @testset "Float32 smart sweep" begin
        n = 9
        x = rand(Float32, n)
        res = pick_chunk(HyperHessiansBackend(), f, x; seconds = SECS, verbose = false)
        chunk_timings = filter(t -> t.kind === :chunk, res.timings)
        @test sort!(unique([t.chunk for t in chunk_timings])) == ChunkPicker.smart_chunks(n, Float32)
    end

    @testset "brute force behind chunks = :all" begin
        n = 6
        x = rand(n)
        res = pick_chunk(HyperHessiansBackend(), f, x; chunks = :all, simd = false, jet = false, seconds = SECS, verbose = false)
        @test [t.chunk for t in res.timings] == collect(1:n)
        @test_throws ArgumentError pick_chunk(HyperHessiansBackend(), f, x; chunks = :bogus, verbose = false)
        # under :all the jet simd variant is not capped at n = 7
        if has_jet && has_jet_simd && has_simd
            x8 = rand(8)
            res8 = pick_chunk(HyperHessiansBackend(), f, x8; seconds = SECS, verbose = false)
            @test count(t -> t.kind === :jet, res8.timings) == 1 # smart: no jet simd at n = 8
            res8_all = pick_chunk(HyperHessiansBackend(), f, x8; chunks = :all, seconds = SECS, verbose = false)
            @test count(t -> t.kind === :jet, res8_all.timings) == 2
        end
    end

    @testset "HyperHessians rejects unsupported ops" begin
        @test_throws ArgumentError pick_chunk(HyperHessiansBackend(), f, rand(3); op = :gradient, verbose = false)
    end
end
