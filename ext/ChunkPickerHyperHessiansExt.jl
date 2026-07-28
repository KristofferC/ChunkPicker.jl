module ChunkPickerHyperHessiansExt

using ChunkPicker: ChunkPicker, HyperHessiansBackend
using HyperHessians: HyperHessians, Chunk, HessianConfig, HVPConfig
using BenchmarkTools: @belapsed

const HAS_SIMD = hasmethod(HessianConfig, Tuple{Vector{Float64}, Chunk{1}}, (:simd,))
const HAS_SIMD_HVP = hasmethod(HVPConfig, Tuple{Vector{Float64}, Vector{Float64}, Chunk{1}}, (:simd,))

# The chunk slot of `HessianConfig(x, chunk = ...; simd)` is untyped, so a
# plain hasmethod probe matches it for *any* second argument; resolve the
# method and check it really is the Jet-specific one before trusting it.
function _jet_config_method()
    isdefined(HyperHessians, :Jet) || return nothing
    m = which(HessianConfig, Tuple{Vector{Float64}, Type{HyperHessians.Jet}})
    return Base.unwrap_unionall(m.sig).parameters[3] === Type{HyperHessians.Jet} ? m : nothing
end
const JET_CONFIG_METHOD = _jet_config_method()
const HAS_JET = JET_CONFIG_METHOD !== nothing
const HAS_JET_SIMD = HAS_JET && :simd in Base.kwarg_decl(JET_CONFIG_METHOD)

# The jet helpers are fully unrolled over the n(n+1)/2 triangle, so compile
# time grows steeply with n; cap the candidate by default (see `jet` kwarg).
const JET_DEFAULT_MAX_N = 32

function ChunkPicker.pick_chunk(
        backend::HyperHessiansBackend, f, x::AbstractArray;
        op::Symbol = :hessian,
        chunks = :smart,
        tangents = nothing,
        seconds::Real = 0.5,
        jet::Union{Bool, Integer} = JET_DEFAULT_MAX_N,
        simd::Bool = true,
        verbose::Bool = true,
    )
    op in (:hessian, :hvp) ||
        throw(ArgumentError("HyperHessiansBackend supports `op = :hessian` or `op = :hvp`, got :$op"))
    has_simd_op = op === :hvp ? HAS_SIMD_HVP : HAS_SIMD
    elt_ok = eltype(x) in (Float32, Float64)
    simd_ok = has_simd_op && elt_ok
    if simd && !simd_ok && verbose
        if !has_simd_op
            @info "This HyperHessians has no `simd` config option for :$op; skipping the SIMD variants."
        else
            @info "SIMD variants only apply to Float32/Float64 inputs (got eltype $(eltype(x))); skipping them."
        end
    end
    smart = chunks === :smart
    candidates = Tuple{Int, Symbol, Bool}[]
    for N in ChunkPicker._resolve_chunks(chunks, length(x), op, eltype(x))
        push!(candidates, (N, :chunk, false))
        simd && simd_ok && push!(candidates, (N, :chunk, true))
    end
    jet_max = jet === true ? typemax(Int) : jet === false ? -1 : Int(jet)
    # In the measured grid (benchmark/RESULTS.md) the jet never wins past
    # n = 16, and jet-with-simd never past n = 7 (its triangle becomes one
    # long Vec), so the smart set stops there. An explicit `jet` cap wins.
    smart && jet === JET_DEFAULT_MAX_N && (jet_max = 16)
    jet_simd_max = smart ? 7 : typemax(Int)
    if op === :hessian && jet !== false
        if !HAS_JET
            verbose && @info "This HyperHessians has no `Jet` config (needs PR #55 / a newer release); skipping the Jet variants."
        elseif length(x) > jet_max
            verbose && !smart && @info "Input length $(length(x)) exceeds the Jet candidate cap of $jet_max (compile time for the unrolled triangle grows steeply with n); pass e.g. `jet = $(length(x))` or `jet = true` to include it anyway."
        else
            push!(candidates, (length(x), :jet, false))
            simd && simd_ok && HAS_JET_SIMD && length(x) <= jet_simd_max &&
                push!(candidates, (length(x), :jet, true))
        end
    end
    tx = op === :hvp ? ChunkPicker._tangent_tuple(x, tangents) : nothing
    v = tx === nothing ? nothing : (length(tx) == 1 ? tx[1] : tx)
    bench = (N, kind, s) -> kind === :jet ? _bench_jet(f, x, s, seconds) :
        op === :hvp ? _bench_hvp(f, x, v, N, s, seconds) : _bench_dual(f, x, N, s, seconds)
    recommend = (N, kind, s) -> kind === :jet ?
        "HyperHessians.HessianConfig(x, HyperHessians.Jet$(s ? "; simd = true" : ""))" :
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

function _bench_jet(f, x, simd::Bool, seconds)
    cfg = simd ? HessianConfig(x, HyperHessians.Jet; simd = true) :
        HessianConfig(x, HyperHessians.Jet)
    H = similar(x, float(eltype(x)), length(x), length(x))
    return @belapsed HyperHessians.hessian!($H, $f, $x, $cfg) seconds = seconds
end

end # module
