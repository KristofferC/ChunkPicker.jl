module ChunkPickerHyperHessiansExt

using ChunkPicker: ChunkPicker, HyperHessiansBackend
using HyperHessians: HyperHessians, Chunk, HessianConfig, HVPConfig
using BenchmarkTools: @belapsed

const HAS_JET = isdefined(HyperHessians, :Jet)
const HAS_SIMD = hasmethod(HessianConfig, Tuple{Vector{Float64}, Chunk{1}}, (:simd,))
const HAS_SIMD_HVP = hasmethod(HVPConfig, Tuple{Vector{Float64}, Vector{Float64}, Chunk{1}}, (:simd,))

function ChunkPicker.pick_chunk(
        backend::HyperHessiansBackend, f, x::AbstractArray;
        op::Symbol = :hessian,
        chunks = 1:min(length(x), 12),
        tangents = nothing,
        seconds::Real = 0.5,
        jet::Bool = true,
        simd::Bool = true,
        verbose::Bool = true,
    )
    op in (:hessian, :hvp) ||
        throw(ArgumentError("HyperHessiansBackend supports `op = :hessian` or `op = :hvp`, got :$op"))
    has_simd_op = op === :hvp ? HAS_SIMD_HVP : HAS_SIMD
    simd_ok = has_simd_op && eltype(x) in (Float32, Float64)
    if simd && !simd_ok && verbose
        if !has_simd_op
            @info "This HyperHessians has no `simd` config option for :$op; skipping the SIMD variants."
        else
            @info "SIMD variants only apply to Float32/Float64 inputs (got eltype $(eltype(x))); skipping them."
        end
    end
    candidates = Tuple{Int, Symbol, Bool}[]
    for N in chunks
        push!(candidates, (N, :chunk, false))
        simd && simd_ok && push!(candidates, (N, :chunk, true))
    end
    if op === :hessian && jet
        if HAS_JET
            push!(candidates, (length(x), :jet, false))
        elseif verbose
            @info "This HyperHessians has no `Jet` (needs PR #55 / a newer release); skipping the Jet variant."
        end
    end
    tx = op === :hvp ? ChunkPicker._tangent_tuple(x, tangents) : nothing
    v = tx === nothing ? nothing : (length(tx) == 1 ? tx[1] : tx)
    bench = (N, kind, s) -> kind === :jet ? _bench_jet(f, x, seconds) :
        op === :hvp ? _bench_hvp(f, x, v, N, s, seconds) : _bench_dual(f, x, N, s, seconds)
    recommend = (N, kind, s) -> kind === :jet ?
        "HyperHessians.HessianConfig(x, HyperHessians.Jet)" :
        op === :hvp ?
        "HyperHessians.HVPConfig(x, v, HyperHessians.Chunk{$N}()$(s ? "; simd = true" : ""))" :
        "HyperHessians.HessianConfig(x, HyperHessians.Chunk{$N}()$(s ? "; simd = true" : ""))"
    return ChunkPicker._run(bench, recommend, backend, op, candidates; verbose)
end

function _bench_dual(f, x, N::Int, simd::Bool, seconds)
    cfg = simd ? HessianConfig(x, Chunk{N}(); simd = true) : HessianConfig(x, Chunk{N}())
    H = similar(x, float(eltype(x)), length(x), length(x))
    return @belapsed HyperHessians.hessian!($H, $f, $x, $cfg) seconds = seconds
end

function _bench_hvp(f, x, v, N::Int, simd::Bool, seconds)
    cfg = simd ? HVPConfig(x, v, Chunk{N}(); simd = true) : HVPConfig(x, v, Chunk{N}())
    hv = v isa Tuple ? map(t -> similar(t, float(eltype(t))), v) : similar(v, float(eltype(v)))
    return @belapsed HyperHessians.hvp!($hv, $f, $x, $v, $cfg) seconds = seconds
end

function _bench_jet(f, x, seconds)
    cfg = HessianConfig(x, HyperHessians.Jet)
    H = similar(x, float(eltype(x)), length(x), length(x))
    return @belapsed HyperHessians.hessian!($H, $f, $x, $cfg) seconds = seconds
end

end # module
