module ChunkPicker

using Printf: @sprintf
import DifferentiationInterface as DI
using DifferentiationInterface: AutoForwardDiff, AutoHyperHessians
using BenchmarkTools: @belapsed

export pick_chunk, HyperHessiansBackend, AutoForwardDiff, AutoHyperHessians

"""
    AbstractBackend

Supertype for ChunkPicker's native backends. Besides these, `pick_chunk`
accepts ADTypes backends driven through DifferentiationInterface —
`AutoForwardDiff` and `AutoHyperHessians` (load the corresponding AD package).
"""
abstract type AbstractBackend end

"""
    HyperHessiansBackend()

Select HyperHessians through its native config API. Supports `op = :hessian`
and `op = :hvp` and benchmarks the axes DifferentiationInterface cannot
express: the SIMD.Vec variants and, when the loaded version provides it, the
`Jet` representation. For a plain chunk sweep use `AutoHyperHessians()`
instead. Requires `using HyperHessians`.
"""
struct HyperHessiansBackend <: AbstractBackend end

backendname(::HyperHessiansBackend) = "HyperHessians"
backendname(::AutoForwardDiff) = "AutoForwardDiff"
backendname(::AutoHyperHessians) = "AutoHyperHessians"

"""
    ChunkTiming(chunk, kind, simd, time)

One measurement. `kind` is `:chunk` (a chunked config of size `chunk`) or `:jet`
(the HyperHessians symmetric `Jet`, which computes the whole Hessian at once).
`simd` marks HyperHessians' SIMD.Vec-forced arithmetic (`simd = true` configs).
`time` is the minimum measured time in seconds.
"""
struct ChunkTiming
    chunk::Int
    kind::Symbol
    simd::Bool
    time::Float64
end

_label(chunk::Integer, kind::Symbol, simd::Bool) =
    kind === :jet ? (simd ? "Jet simd" : "Jet") :
    simd ? "chunk $(chunk) simd" : "chunk $(chunk)"
_label(t::ChunkTiming) = _label(t.chunk, t.kind, t.simd)

"""
    ChunkPickResult

Result of [`pick_chunk`](@ref). Fields:

- `chunk`          : chunk size of the fastest variant (for a `Jet` this is `length(x)`).
- `kind`           : `:chunk` or `:jet` — which representation was fastest.
- `simd`           : whether the fastest variant uses SIMD.Vec-forced arithmetic
                     (HyperHessians `simd = true` configs).
- `timings`        : `Vector{ChunkTiming}`, one per candidate.
- `backend`, `op`  : what was benchmarked.
- `recommendation` : ready-to-use backend/config constructor for the fastest variant.
"""
struct ChunkPickResult
    chunk::Int
    kind::Symbol
    simd::Bool
    timings::Vector{ChunkTiming}
    backend::Any
    op::Symbol
    recommendation::String
end

"""
    pick_chunk(backend, f, x; op, chunks, tangents, seconds, verbose) -> ChunkPickResult

Benchmark `f` differentiated at `x` with `backend` across candidate chunk sizes and
return the fastest variant.

# Arguments
- `backend` : an ADTypes backend driven through DifferentiationInterface —
              `AutoForwardDiff()` (requires `using ForwardDiff`) or
              `AutoHyperHessians()` (requires `using HyperHessians`) — or the
              native [`HyperHessiansBackend`](@ref).
- `f`       : the function to differentiate. For `:gradient`/`:hessian`/`:hvp`,
              `f(x)::Real`; for `:jacobian`, `f(x)::AbstractArray`.
- `x`       : the input point (`AbstractArray`).

# Keywords
- `op`       : `:gradient`, `:jacobian`, `:hessian` or `:hvp` (`AutoForwardDiff`);
               `:hessian` or `:hvp` (`AutoHyperHessians` and HyperHessians native).
- `chunks`   : candidate chunk sizes. Defaults are capped since the useful range is
               small: `1:min(length(x), 32)` for `:gradient`/`:jacobian`,
               `1:min(length(x), 12)` for `:hessian`/`:hvp`. Pass an explicit range to
               sweep further.
- `tangents` : for `op = :hvp`, the tangent vector (or tuple of vectors for bundled
               directions). Defaults to a vector of ones.
- `seconds`  : per-candidate benchmark budget passed to BenchmarkTools (default `0.5`).
- `verbose`  : print progress while benchmarking (default `true`).

The native HyperHessians backend additionally accepts `jet` to include the `Jet`
variants (HyperHessians never selects `Jet` on its own, so this sweep is how you
find out whether it wins). `jet` is an `Integer` cap on the input length for which
the candidate is included (default `32` — the jet's unrolled triangle makes compile
time grow steeply with `length(x)`), or `true`/`false` to force/disable it; it is
skipped, with a note, when the loaded HyperHessians has no `Jet`. `simd::Bool =
true` additionally benchmarks each chunk size — and the `Jet` — with
SIMD.Vec-forced arithmetic (`...; simd = true` configs; skipped when the loaded
HyperHessians has no `simd` option or the eltype is not Float32/Float64).

# Example
```julia
using ChunkPicker, ForwardDiff
res = pick_chunk(AutoForwardDiff(), x -> sum(abs2, x), rand(50); op = :gradient)
res.chunk
```
"""
function pick_chunk end

# Fallback for a native backend whose package has not been loaded. The extension
# adds a strictly more specific method on the concrete backend type, so this
# only fires when the extension is inactive.
function pick_chunk(b::AbstractBackend, f, x::AbstractArray; kwargs...)
    error(
        "$(backendname(b)) must be loaded to use `$(nameof(typeof(b)))`. " *
            "Run `using $(backendname(b))` first.",
    )
end

# Shared driver, also used by the HyperHessians extension.
# - `candidates` : iterable of `(chunk::Int, kind::Symbol, simd::Bool)`.
# - `bench(chunk, kind, simd)` : builds the config/output and returns the minimum time (seconds).
# - `recommend(chunk, kind, simd)` : constructor snippet for the given variant.
function _run(bench, recommend, backend, op::Symbol, candidates; verbose::Bool)
    isempty(candidates) && throw(ArgumentError("no candidates to benchmark"))
    verbose && printstyled(
        "Benchmarking $(backendname(backend)) :$op over $(length(candidates)) candidate(s)\n";
        bold = true,
    )
    timings = ChunkTiming[]
    for (chunk, kind, simd) in candidates
        t = bench(chunk, kind, simd)
        push!(timings, ChunkTiming(chunk, kind, simd, t))
        verbose && println("  ", rpad(_label(chunk, kind, simd), 14), " ", _fmt(t))
    end
    best = argmin(t -> t.time, timings)
    verbose && printstyled(
        "  → fastest: $(_label(best))  ($(_fmt(best.time)))\n";
        bold = true, color = :green,
    )
    return ChunkPickResult(best.chunk, best.kind, best.simd, timings, backend, op, recommend(best.chunk, best.kind, best.simd))
end

function _fmt(t::Real) # t in seconds
    t < 1.0e-6 && return @sprintf("%.1f ns", t * 1.0e9)
    t < 1.0e-3 && return @sprintf("%.3f μs", t * 1.0e6)
    t < 1.0 && return @sprintf("%.3f ms", t * 1.0e3)
    return @sprintf("%.3f s", t)
end

# Default tangent(s) for op = :hvp.
_tangent_tuple(x, ::Nothing) = (ones(float(eltype(x)), size(x)...),)
_tangent_tuple(x, v::AbstractArray) = (v,)
_tangent_tuple(x, v::Tuple) = v

function Base.show(io::IO, ::MIME"text/plain", r::ChunkPickResult)
    println(io, "ChunkPickResult ($(backendname(r.backend)), :$(r.op))")
    tmin = minimum(t.time for t in r.timings)
    for t in r.timings
        best = t.chunk == r.chunk && t.kind == r.kind && t.simd == r.simd
        rel = t.time / tmin
        println(io, best ? "* " : "  ", rpad(_label(t), 14), " ", lpad(_fmt(t.time), 10), "  ", @sprintf("%.2fx", rel))
    end
    print(io, "→ ", r.recommendation)
    return
end

## ADTypes backends, driven through DifferentiationInterface. The chunk size is
## a parameter of the backend type, so the sweep rebuilds the backend per
## candidate; DI's preparation is rebuilt with it. The AD package itself
## (ForwardDiff, HyperHessians) must be loaded for DI to drive it.
const ChunkedADType = Union{AutoForwardDiff, AutoHyperHessians}

_with_chunk(b::AutoForwardDiff, N::Int) = AutoForwardDiff(; chunksize = N, tag = b.tag)
_with_chunk(::AutoHyperHessians, N::Int) = AutoHyperHessians(; chunksize = N)

_supported_ops(::AutoForwardDiff) = (:gradient, :jacobian, :hessian, :hvp)
_supported_ops(::AutoHyperHessians) = (:hessian, :hvp)

function pick_chunk(
        backend::ChunkedADType, f, x::AbstractArray;
        op::Symbol = first(_supported_ops(backend)),
        chunks = 1:min(length(x), op === :hessian || op === :hvp ? 12 : 32),
        tangents = nothing,
        seconds::Real = 0.5,
        verbose::Bool = true,
    )
    op in _supported_ops(backend) || throw(
        ArgumentError(
            "$(backendname(backend)): `op` must be one of " *
                "$(_supported_ops(backend)), got :$op"
        )
    )
    tx = op === :hvp ? _tangent_tuple(x, tangents) : nothing
    candidates = [(N, :chunk, false) for N in chunks]
    bench = (N, _kind, _simd) -> _bench_di(Val(op), f, _with_chunk(backend, N), x, tx, seconds)
    name = backendname(backend)
    recommend = (N, _kind, _simd) -> "$name(chunksize = $N)"
    return _run(bench, recommend, backend, op, candidates; verbose)
end

function _bench_di(::Val{:gradient}, f, b, x, tx, seconds)
    out = similar(x, float(eltype(x)))
    prep = DI.prepare_gradient(f, b, x)
    return @belapsed DI.gradient!($f, $out, $prep, $b, $x) seconds = seconds
end

function _bench_di(::Val{:jacobian}, f, b, x, tx, seconds)
    y = f(x)
    out = similar(y, float(eltype(y)), length(y), length(x))
    prep = DI.prepare_jacobian(f, b, x)
    return @belapsed DI.jacobian!($f, $out, $prep, $b, $x) seconds = seconds
end

function _bench_di(::Val{:hessian}, f, b, x, tx, seconds)
    out = similar(x, float(eltype(x)), length(x), length(x))
    prep = DI.prepare_hessian(f, b, x)
    return @belapsed DI.hessian!($f, $out, $prep, $b, $x) seconds = seconds
end

function _bench_di(::Val{:hvp}, f, b, x, tx, seconds)
    ty = map(v -> similar(v, float(eltype(v))), tx)
    prep = DI.prepare_hvp(f, b, x, tx)
    return @belapsed DI.hvp!($f, $ty, $prep, $b, $x, $tx) seconds = seconds
end

end # module ChunkPicker
