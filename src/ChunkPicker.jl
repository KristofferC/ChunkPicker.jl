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
- `chunks`   : candidate chunk sizes. The default `:smart` benchmarks only the
               measured candidate set for the op and backend (HyperHessians:
               [`smart_chunks`](@ref); ForwardDiff: op-specific sets from the
               ForwardDiff grid — e.g. the gradient sweep tests ~6 sizes instead
               of 32). Pass `:all` for the exhaustive brute-force sweep, or any
               explicit iterable of sizes.
- `tangents` : for `op = :hvp`, the tangent vector (or tuple of vectors for bundled
               directions). Defaults to a vector of ones.
- `seconds`  : per-candidate benchmark budget passed to BenchmarkTools (default `0.5`).
- `verbose`  : print progress while benchmarking (default `true`).

The native HyperHessians backend defaults to `chunks = :smart` and additionally
accepts `jet` to include the `Jet` variants (HyperHessians never selects `Jet` on
its own, so this sweep is how you find out whether it wins). `jet` is an `Integer`
cap on the input length for which the candidate is included (default `32`; under
`:smart` the default cap is 16, past which the jet never won in the measured
grid), or `true`/`false` to force/disable it; it is skipped, with a note, when the
loaded HyperHessians has no `Jet`. `simd::Bool = true` additionally benchmarks
each candidate with SIMD.Vec-forced arithmetic (`...; simd = true` configs;
skipped when the loaded HyperHessians has no `simd` option or the eltype is not
Float32/Float64; under `:smart` the Jet-with-simd variant stops at `length(x) = 7`
where its measured cliff begins).

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
"""
    smart_chunks(n::Integer[, T::Type]) -> Vector{Int}

The default chunk-size candidates for an input of length `n` with eltype `T`:
the sizes that ever win, measured on a benchmark grid of 6 function families
x 20 input sizes x chunks 1:16 (1:24 for Float32) x simd on/off, on AVX2
(Arrow Lake), AVX-512 (Zen 4) and NEON (Apple M4) — see
`benchmark/RESULTS.md`. For `n <= 4` everything is included; above that:

- a base set `{2, 3, 4, 6, 8, 12, 16}`: small chunks win for cheap
  functions (dual size dominates), SIMD-width multiples for expensive ones;
- `n` itself while `n <= cap`: the whole Hessian in one evaluation;
- `cld(n, 2)` and `cld(n, 3)`: fewest evaluations per dual size;
- divisors of `n` in `4:cap`: no padded trailing chunk.

`cap` is 16, except for `Float32` where the doubled SIMD lane count moves
the winners up: `cap` is 24 and the base set drops 2. Across the measured
grid the best candidate in this set is within 2% of the exhaustive optimum
in 99% of Float64 cases (worst 9%) and exactly optimal in every Float32
case, with ~40% fewer candidates than the previous dense sweep. Pass
`chunks = :all` to `pick_chunk` for the exhaustive sweep instead.
"""
smart_chunks(n::Integer) = _smart_chunks(n, (2, 3, 4, 6, 8, 12, 16), 16)
smart_chunks(n::Integer, ::Type{T}) where {T} = smart_chunks(n)
smart_chunks(n::Integer, ::Type{Float32}) = _smart_chunks(n, (3, 4, 6, 8, 12, 16), 24)

function _smart_chunks(n::Integer, base, cap::Int; frontier_k = 2:3, divisors = 4:cap, single_max = cap)
    n <= 0 && return Int[]
    n <= 4 && return collect(1:n)
    chunks = Int[c for c in base if c < n]
    n <= single_max && push!(chunks, n)
    for k in frontier_k
        c = cld(n, k)
        1 < c < n && c <= cap && push!(chunks, c)
    end
    for d in divisors
        d < n && n % d == 0 && push!(chunks, d)
    end
    return sort!(unique!(chunks))
end

# ForwardDiff sweeps use their own sets, measured on the ForwardDiff grid
# (fd_grid.jl in benchmark/): gradient cost per evaluation is only O(chunk),
# so plateaus are wide and winners sit high (full-vector keeps winning to
# n = 32 for cheap functions); the Dual-of-Dual hessian amortizes its
# per-evaluation overhead hardest, favoring the largest chunks plus the
# cld(n, k) frontier and divisors; hvp sits between them.
_smart_chunks_fd_grad(n::Integer) =
    _smart_chunks(n, (3, 4, 5, 8, 12, 16, 24, 32), 32; frontier_k = 1:0, divisors = 1:0)
_smart_chunks_fd_hess(n::Integer) =
    _smart_chunks(n, (4, 8, 12, 16), 16; frontier_k = 2:7)
_smart_chunks_fd_hvp(n::Integer) =
    _smart_chunks(n, (2, 4, 8, 16), 32; frontier_k = 2:7, divisors = 12:4:24)

# `chunks` kwarg: `:smart` (the measured candidate set for the given op and
# backend family), `:all` (exhaustive brute force), or any iterable of sizes.
function _resolve_chunks(chunks, n::Int, op::Symbol, ::Type{T} = Float64, family::Symbol = :hyperhessians) where {T}
    if chunks === :smart
        family === :forwarddiff || return smart_chunks(n, T)
        op === :hessian && return _smart_chunks_fd_hess(n)
        op === :hvp && return _smart_chunks_fd_hvp(n)
        return _smart_chunks_fd_grad(n) # :gradient / :jacobian
    end
    if chunks === :all
        op === :hessian || op === :hvp || return collect(1:min(n, 32))
        cap = T === Float32 ? 24 : 16
        full = collect(1:min(n, cap))
        cap < n <= 20 && push!(full, n)
        return full
    end
    chunks isa Symbol && throw(ArgumentError("`chunks` must be `:smart`, `:all`, or an iterable of sizes, got :$chunks"))
    return collect(Int, chunks)
end

const ChunkedADType = Union{AutoForwardDiff, AutoHyperHessians}

_with_chunk(b::AutoForwardDiff, N::Int) = AutoForwardDiff(; chunksize = N, tag = b.tag)
_with_chunk(::AutoHyperHessians, N::Int) = AutoHyperHessians(; chunksize = N)

_supported_ops(::AutoForwardDiff) = (:gradient, :jacobian, :hessian, :hvp)
_supported_ops(::AutoHyperHessians) = (:hessian, :hvp)

function pick_chunk(
        backend::ChunkedADType, f, x::AbstractArray;
        op::Symbol = first(_supported_ops(backend)),
        chunks = :smart,
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
    family = backend isa AutoForwardDiff ? :forwarddiff : :hyperhessians
    candidates = [(N, :chunk, false) for N in _resolve_chunks(chunks, length(x), op, eltype(x), family)]
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
