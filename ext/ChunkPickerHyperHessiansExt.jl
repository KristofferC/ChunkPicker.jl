module ChunkPickerHyperHessiansExt

using ChunkPicker: ChunkPicker, HyperHessiansBackend
using HyperHessians: HyperHessians, Chunk, HessianConfig
using BenchmarkTools: @belapsed

const HAS_JET = isdefined(HyperHessians, :Jet)

function ChunkPicker.pick_chunk(
        backend::HyperHessiansBackend, f, x::AbstractArray;
        op::Symbol = :hessian,
        chunks = 1:min(length(x), 12),
        seconds::Real = 0.5,
        jet::Bool = true,
        verbose::Bool = true,
    )
    op === :hessian ||
        throw(ArgumentError("HyperHessiansBackend only supports `op = :hessian`, got :$op"))
    candidates = Tuple{Int, Symbol}[(N, :chunk) for N in chunks]
    if jet
        if HAS_JET
            push!(candidates, (length(x), :jet))
        elseif verbose
            @info "This HyperHessians has no `Jet` (needs PR #55 / a newer release); skipping the Jet variant."
        end
    end
    bench = (N, kind) -> kind === :jet ? _bench_jet(f, x, seconds) : _bench_dual(f, x, N, seconds)
    recommend = (N, kind) -> kind === :jet ?
        "HyperHessians.HessianConfig(x, HyperHessians.Jet)" :
        "HyperHessians.HessianConfig(x, HyperHessians.Chunk{$N}())"
    return ChunkPicker._run(bench, recommend, backend, op, candidates; verbose)
end

function _bench_dual(f, x, N::Int, seconds)
    cfg = HessianConfig(x, Chunk{N}())
    H = similar(x, float(eltype(x)), length(x), length(x))
    return @belapsed HyperHessians.hessian!($H, $f, $x, $cfg) seconds = seconds
end

function _bench_jet(f, x, seconds)
    cfg = HessianConfig(x, HyperHessians.Jet)
    H = similar(x, float(eltype(x)), length(x), length(x))
    return @belapsed HyperHessians.hessian!($H, $f, $x, $cfg) seconds = seconds
end

end # module
