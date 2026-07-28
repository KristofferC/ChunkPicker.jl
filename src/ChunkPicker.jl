module ChunkPicker

using Printf: @sprintf

export pick_chunk, HyperHessiansBackend

"""
    AbstractBackend

Supertype for ChunkPicker's native backends. Besides these, `pick_chunk`
accepts ADTypes backends driven through DifferentiationInterface — currently
`AutoForwardDiff` (load DifferentiationInterface and ForwardDiff).
"""
abstract type AbstractBackend end

"""
    HyperHessiansBackend()

Select HyperHessians. Supports `op = :hessian` and `op = :hvp`. Benchmarks the
`HyperDual` chunk sweep, the SIMD.Vec variants, and, when the loaded version
provides it, the `Jet` representation. Requires `using HyperHessians`.
"""
struct HyperHessiansBackend <: AbstractBackend end

backendname(::HyperHessiansBackend) = "HyperHessians"

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
    kind === :jet ? "Jet" : simd ? "chunk $(chunk) simd" : "chunk $(chunk)"
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
- `backend` : an ADTypes backend driven through DifferentiationInterface (currently
              `AutoForwardDiff()`; requires DifferentiationInterface and ForwardDiff to
              be loaded), or the native [`HyperHessiansBackend`](@ref).
- `f`       : the function to differentiate. For `:gradient`/`:hessian`/`:hvp`,
              `f(x)::Real`; for `:jacobian`, `f(x)::AbstractArray`.
- `x`       : the input point (`AbstractArray`).

# Keywords
- `op`       : `:gradient`, `:jacobian`, `:hessian` or `:hvp` (`AutoForwardDiff`);
               `:hessian` or `:hvp` (HyperHessians).
- `chunks`   : candidate chunk sizes. Defaults are capped since the useful range is
               small: `1:min(length(x), 32)` for `:gradient`/`:jacobian`,
               `1:min(length(x), 12)` for `:hessian`/`:hvp`. Pass an explicit range to
               sweep further.
- `tangents` : for `op = :hvp`, the tangent vector (or tuple of vectors for bundled
               directions). Defaults to a vector of ones.
- `seconds`  : per-candidate benchmark budget passed to BenchmarkTools (default `0.5`).
- `verbose`  : print progress while benchmarking (default `true`).

HyperHessians additionally accepts `jet::Bool = true` to include the `Jet` variant
(ignored, with a note, when the loaded HyperHessians has no `Jet`), and
`simd::Bool = true` to also benchmark each chunk size with SIMD.Vec-forced
arithmetic (`HessianConfig(...; simd = true)`; skipped when the loaded
HyperHessians has no `simd` option or the eltype is not Float32/Float64).

# Example
```julia
using ChunkPicker, DifferentiationInterface, ForwardDiff
res = pick_chunk(AutoForwardDiff(), x -> sum(abs2, x), rand(50); op = :gradient)
res.chunk
```
"""
function pick_chunk end

# Fallback for a backend whose package has not been loaded. The extension adds a
# strictly more specific method on the concrete backend type, so this only fires
# when the extension is inactive.
function pick_chunk(b::AbstractBackend, f, x::AbstractArray; kwargs...)
    error(
        "$(backendname(b)) must be loaded to use `$(nameof(typeof(b)))`. " *
            "Run `using $(backendname(b))` first.",
    )
end

# Shared driver used by the extensions.
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

# Default tangent(s) for op = :hvp; extensions normalize user input with this.
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

end # module ChunkPicker
