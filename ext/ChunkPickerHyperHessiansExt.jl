module ChunkPickerHyperHessiansExt

using ChunkPicker: ChunkPicker, HyperHessiansBackend
using HyperHessians: HyperHessians, Chunk, HessianConfig
using BenchmarkTools: @belapsed

const HAS_JET = isdefined(HyperHessians, :Jet)
const HAS_SIMD = hasmethod(HessianConfig, Tuple{Vector{Float64}, Chunk{1}}, (:simd,))

function ChunkPicker.pick_chunk(
        backend::HyperHessiansBackend, f, x::AbstractArray;
        op::Symbol = :hessian,
        chunks = 1:min(length(x), 12),
        seconds::Real = 0.5,
        jet::Bool = true,
        simd::Bool = true,
        verbose::Bool = true,
    )
    op === :hessian ||
        throw(ArgumentError("HyperHessiansBackend only supports `op = :hessian`, got :$op"))
    simd_ok = HAS_SIMD && eltype(x) <: Union{Float32, Float64}
    if simd && !simd_ok && verbose && !HAS_SIMD
        @info "This HyperHessians has no `simd` config option; skipping the SIMD variants."
    end
    candidates = Tuple{Int, Symbol, Bool}[]
    for N in chunks
        push!(candidates, (N, :chunk, false))
        simd && simd_ok && push!(candidates, (N, :chunk, true))
    end
    if jet
        if HAS_JET
            push!(candidates, (length(x), :jet, false))
        elseif verbose
            @info "This HyperHessians has no `Jet` (needs PR #55 / a newer release); skipping the Jet variant."
        end
    end
    bench = (N, kind, s) -> kind === :jet ? _bench_jet(f, x, seconds) : _bench_dual(f, x, N, s, seconds)
    recommend = (N, kind, s) -> kind === :jet ?
        "HyperHessians.HessianConfig(x, HyperHessians.Jet)" :
        "HyperHessians.HessianConfig(x, HyperHessians.Chunk{$N}()$(s ? "; simd = true" : ""))"
    return ChunkPicker._run(bench, recommend, backend, op, candidates; verbose)
end

function _bench_dual(f, x, N::Int, simd::Bool, seconds)
    cfg = simd ? HessianConfig(x, Chunk{N}(); simd = true) : HessianConfig(x, Chunk{N}())
    H = similar(x, float(eltype(x)), length(x), length(x))
    return @belapsed HyperHessians.hessian!($H, $f, $x, $cfg) seconds = seconds
end

function _bench_jet(f, x, seconds)
    cfg = HessianConfig(x, HyperHessians.Jet)
    H = similar(x, float(eltype(x)), length(x), length(x))
    return @belapsed HyperHessians.hessian!($H, $f, $x, $cfg) seconds = seconds
end

end # module
