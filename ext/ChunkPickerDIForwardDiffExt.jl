module ChunkPickerDIForwardDiffExt

using ChunkPicker: ChunkPicker
import DifferentiationInterface as DI
using DifferentiationInterface: AutoForwardDiff
using ForwardDiff: ForwardDiff
using BenchmarkTools: @belapsed

ChunkPicker.backendname(::AutoForwardDiff) = "AutoForwardDiff"

_with_chunk(b::AutoForwardDiff, N::Int) = AutoForwardDiff(; chunksize = N, tag = b.tag)

function ChunkPicker.pick_chunk(
        backend::AutoForwardDiff, f, x::AbstractArray;
        op::Symbol = :gradient,
        chunks = 1:min(length(x), op === :hessian || op === :hvp ? 12 : 32),
        tangents = nothing,
        seconds::Real = 0.5,
        verbose::Bool = true,
    )
    op in (:gradient, :jacobian, :hessian, :hvp) ||
        throw(ArgumentError("AutoForwardDiff: `op` must be :gradient, :jacobian, :hessian or :hvp, got :$op"))
    tx = op === :hvp ? ChunkPicker._tangent_tuple(x, tangents) : nothing
    candidates = [(N, :chunk, false) for N in chunks]
    bench = (N, _kind, _simd) -> _bench(Val(op), f, _with_chunk(backend, N), x, tx, seconds)
    recommend = (N, _kind, _simd) -> "AutoForwardDiff(chunksize = $N)"
    return ChunkPicker._run(bench, recommend, backend, op, candidates; verbose)
end

function _bench(::Val{:gradient}, f, b, x, tx, seconds)
    out = similar(x, float(eltype(x)))
    prep = DI.prepare_gradient(f, b, x)
    return @belapsed DI.gradient!($f, $out, $prep, $b, $x) seconds = seconds
end

function _bench(::Val{:jacobian}, f, b, x, tx, seconds)
    y = f(x)
    out = similar(y, float(eltype(y)), length(y), length(x))
    prep = DI.prepare_jacobian(f, b, x)
    return @belapsed DI.jacobian!($f, $out, $prep, $b, $x) seconds = seconds
end

function _bench(::Val{:hessian}, f, b, x, tx, seconds)
    out = similar(x, float(eltype(x)), length(x), length(x))
    prep = DI.prepare_hessian(f, b, x)
    return @belapsed DI.hessian!($f, $out, $prep, $b, $x) seconds = seconds
end

function _bench(::Val{:hvp}, f, b, x, tx, seconds)
    ty = map(v -> similar(v, float(eltype(v))), tx)
    prep = DI.prepare_hvp(f, b, x, tx)
    return @belapsed DI.hvp!($f, $ty, $prep, $b, $x, $tx) seconds = seconds
end

end # module
