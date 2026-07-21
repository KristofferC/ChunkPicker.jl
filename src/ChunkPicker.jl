module ChunkPicker

using Printf: @sprintf

export pick_chunk, ForwardDiffBackend, HyperHessiansBackend

"""
    AbstractBackend

Supertype for the AD backends `ChunkPicker` can benchmark. Load the corresponding
package (`ForwardDiff` or `HyperHessians`) to enable a backend.
"""
abstract type AbstractBackend end

"""
    ForwardDiffBackend()

Select ForwardDiff. Supports `op = :gradient`, `:jacobian` and `:hessian`.
Requires `using ForwardDiff`.
"""
struct ForwardDiffBackend <: AbstractBackend end

"""
    HyperHessiansBackend()

Select HyperHessians. Supports `op = :hessian` only. Benchmarks the `HyperDual`
chunk sweep and, when the loaded version provides it, the `Jet` representation.
Requires `using HyperHessians`.
"""
struct HyperHessiansBackend <: AbstractBackend end

backendname(::ForwardDiffBackend) = "ForwardDiff"
backendname(::HyperHessiansBackend) = "HyperHessians"

"""
    ChunkTiming(chunk, kind, time)

One measurement. `kind` is `:chunk` (a chunked config of size `chunk`) or `:jet`
(the HyperHessians symmetric `Jet`, which computes the whole Hessian at once).
`time` is the minimum measured time in seconds.
"""
struct ChunkTiming
    chunk::Int
    kind::Symbol
    time::Float64
end

_label(chunk::Integer, kind::Symbol) = kind === :jet ? "Jet" : "chunk $(chunk)"
_label(t::ChunkTiming) = _label(t.chunk, t.kind)

"""
    ChunkPickResult

Result of [`pick_chunk`](@ref). Fields:

- `chunk`          : chunk size of the fastest variant (for a `Jet` this is `length(x)`).
- `kind`           : `:chunk` or `:jet` — which representation was fastest.
- `timings`        : `Vector{ChunkTiming}`, one per candidate.
- `backend`, `op`  : what was benchmarked.
- `recommendation` : ready-to-use config constructor for the fastest variant.
"""
struct ChunkPickResult
    chunk::Int
    kind::Symbol
    timings::Vector{ChunkTiming}
    backend::AbstractBackend
    op::Symbol
    recommendation::String
end

"""
    pick_chunk(backend, f, x; op, chunks, seconds, verbose) -> ChunkPickResult

Benchmark `f` differentiated at `x` with `backend` across candidate chunk sizes (and,
for HyperHessians, the `Jet` representation) and return the fastest variant.

# Arguments
- `backend` : [`ForwardDiffBackend`](@ref) or [`HyperHessiansBackend`](@ref).
- `f`       : the function to differentiate. For `:gradient`/`:hessian`, `f(x)::Real`;
              for `:jacobian`, `f(x)::AbstractArray`.
- `x`       : the input point (`AbstractArray`).

# Keywords
- `op`      : `:gradient`, `:jacobian` or `:hessian` (default `:gradient` for ForwardDiff,
              `:hessian` for HyperHessians).
- `chunks`  : candidate chunk sizes. Defaults are capped since the useful range is small:
              `1:min(length(x), 32)` for `:gradient`/`:jacobian`, `1:min(length(x), 12)`
              for `:hessian`. Pass an explicit range to sweep further.
- `seconds` : per-candidate benchmark budget passed to BenchmarkTools (default `0.5`).
- `verbose` : print progress while benchmarking (default `true`).

HyperHessians additionally accepts `jet::Bool = true` to include the `Jet` variant
(ignored, with a note, when the loaded HyperHessians has no `Jet`).

# Example
```julia
using ChunkPicker, ForwardDiff
res = pick_chunk(ForwardDiffBackend(), x -> sum(abs2, x), rand(50); op = :gradient)
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
# - `candidates` : iterable of `(chunk::Int, kind::Symbol)`.
# - `bench(chunk, kind)` : builds the config/output and returns the minimum time (seconds).
# - `recommend(chunk, kind)` : constructor snippet for the given variant.
function _run(bench, recommend, backend::AbstractBackend, op::Symbol, candidates; verbose::Bool)
    isempty(candidates) && throw(ArgumentError("no candidates to benchmark"))
    verbose && printstyled(
        "Benchmarking $(backendname(backend)) :$op over $(length(candidates)) candidate(s)\n";
        bold = true,
    )
    timings = ChunkTiming[]
    for (chunk, kind) in candidates
        t = bench(chunk, kind)
        push!(timings, ChunkTiming(chunk, kind, t))
        verbose && println("  ", rpad(_label(chunk, kind), 10), " ", _fmt(t))
    end
    best = argmin(t -> t.time, timings)
    verbose && printstyled(
        "  → fastest: $(_label(best))  ($(_fmt(best.time)))\n";
        bold = true, color = :green,
    )
    return ChunkPickResult(best.chunk, best.kind, timings, backend, op, recommend(best.chunk, best.kind))
end

function _fmt(t::Real) # t in seconds
    t < 1e-6 && return @sprintf("%.1f ns", t * 1e9)
    t < 1e-3 && return @sprintf("%.3f μs", t * 1e6)
    t < 1.0  && return @sprintf("%.3f ms", t * 1e3)
    return @sprintf("%.3f s", t)
end

function Base.show(io::IO, ::MIME"text/plain", r::ChunkPickResult)
    println(io, "ChunkPickResult ($(backendname(r.backend)), :$(r.op))")
    tmin = minimum(t.time for t in r.timings)
    for t in r.timings
        best = t.chunk == r.chunk && t.kind == r.kind
        rel = t.time / tmin
        println(io, best ? "* " : "  ", rpad(_label(t), 10), " ", lpad(_fmt(t.time), 10), "  ", @sprintf("%.2fx", rel))
    end
    print(io, "→ ", r.recommendation)
end

end # module ChunkPicker
