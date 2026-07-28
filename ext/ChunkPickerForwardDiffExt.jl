module ChunkPickerForwardDiffExt

using ChunkPicker: ChunkPicker, ForwardDiffBackend
using ForwardDiff: ForwardDiff, Chunk, GradientConfig, JacobianConfig, HessianConfig
using BenchmarkTools: @belapsed

function ChunkPicker.pick_chunk(
        backend::ForwardDiffBackend, f, x::AbstractArray;
        op::Symbol = :gradient,
        chunks = 1:min(length(x), op === :hessian ? 12 : 32),
        seconds::Real = 0.5,
        verbose::Bool = true,
    )
    op in (:gradient, :jacobian, :hessian) ||
        throw(ArgumentError("ForwardDiffBackend: `op` must be :gradient, :jacobian or :hessian, got :$op"))
    candidates = [(N, :chunk, false) for N in chunks]
    bench = (N, _kind, _simd) -> _bench(Val(op), f, x, N, seconds)
    cfgname = op === :gradient ? "GradientConfig" : op === :jacobian ? "JacobianConfig" : "HessianConfig"
    recommend = (N, _kind, _simd) -> "ForwardDiff.$cfgname(f, x, ForwardDiff.Chunk{$N}())"
    return ChunkPicker._run(bench, recommend, backend, op, candidates; verbose)
end

function _bench(::Val{:gradient}, f, x, N::Int, seconds)
    cfg = GradientConfig(f, x, Chunk{N}())
    out = similar(x, float(eltype(x)))
    return @belapsed ForwardDiff.gradient!($out, $f, $x, $cfg) seconds = seconds
end

function _bench(::Val{:jacobian}, f, x, N::Int, seconds)
    y = f(x)
    cfg = JacobianConfig(f, x, Chunk{N}())
    out = similar(y, float(eltype(y)), length(y), length(x))
    return @belapsed ForwardDiff.jacobian!($out, $f, $x, $cfg) seconds = seconds
end

function _bench(::Val{:hessian}, f, x, N::Int, seconds)
    cfg = HessianConfig(f, x, Chunk{N}())
    out = similar(x, float(eltype(x)), length(x), length(x))
    return @belapsed ForwardDiff.hessian!($out, $f, $x, $cfg) seconds = seconds
end

end # module
