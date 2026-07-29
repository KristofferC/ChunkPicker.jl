# Self-contained version of fd_grid.jl: bootstraps a temp environment, so it
# runs on any machine with just Julia.
#
#   julia fd_grid_standalone.jl
#
# Writes ~/fd_grid_<host>[_f32].tsv and prints progress. GRID_SECONDS
# (default 0.2) per-candidate budget; GRID_ELTYPE=Float32 for the reduced
# Float32 grid.

import Pkg
Pkg.activate(mktempdir())
Pkg.add(["ForwardDiff", "BenchmarkTools", "DiffTests"])

using BenchmarkTools
import ForwardDiff, DiffTests

BenchmarkTools.DEFAULT_PARAMETERS.seconds = parse(Float64, get(ENV, "GRID_SECONDS", "0.2"))

const ELT = get(ENV, "GRID_ELTYPE", "Float64") == "Float32" ? Float32 : Float64
const MINI = ELT === Float32

sumexp(x) = sum(abs2, x) + exp(sum(x))
logbarrier(x) = -sum(log, x) + 0.5 * sum(abs2, x)
function chaincouple(x)
    s = zero(eltype(x))
    @inbounds for i in 1:(length(x) - 1)
        s += tanh(x[i] * x[i + 1])
    end
    return s + sum(abs2, x)
end

const FUNCS = [
    "rosenbrock" => DiffTests.rosenbrock_1,
    "ackley" => DiffTests.ackley,
    "logit" => DiffTests.self_weighted_logit,
    "sumexp" => sumexp,
    "logbarrier" => logbarrier,
    "chaincouple" => chaincouple,
]

# Gradient: per-eval cost ~ O(chunk), winners known to sit higher; sweep to 32.
const SIZES_G = MINI ? [8, 16, 32, 64, 200] : [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 100, 200]
const CAP_G = 32
# Hessian: Dual-of-Dual, (c+1)^2 values per number; same shape as the
# HyperHessians grid for comparability.
const SIZES_H = MINI ? [4, 8, 12, 16, 24, 32, 48, 100] : [1:12; 14; 16; 20; 24; 32; 48; 64; 100]
const CAP_H = 16

function bench_grad(f, x, c::Int)
    cfg = ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{c}())
    out = similar(x)
    return @belapsed ForwardDiff.gradient!($out, $f, $x, $cfg)
end

function bench_hess(f, x, c::Int)
    cfg = ForwardDiff.HessianConfig(f, x, ForwardDiff.Chunk{c}())
    H = similar(x, length(x), length(x))
    return @belapsed ForwardDiff.hessian!($H, $f, $x, $cfg)
end

function main(out::String)
    open(out, "w") do io
        println(io, "# host=$(gethostname()) elt=$(ELT) fd=$(pkgversion(ForwardDiff)) julia=$(VERSION) cpu=$(Sys.cpu_info()[1].model) date=$(time())")
        println(io, "func\tn\tkind\tchunk\tsimd\ttime")
        for (name, f) in FUNCS
            for n in SIZES_G
                x = collect(range(ELT(0.1), ELT(1.0); length = n))
                for c in 1:min(n, CAP_G)
                    t = bench_grad(f, x, c)
                    println(io, "$name\t$n\tgrad\t$c\tfalse\t$t")
                end
                flush(io)
                println(stderr, "done grad $name n=$n")
            end
            for n in SIZES_H
                x = n == 1 ? ELT[0.55] : collect(range(ELT(0.1), ELT(1.0); length = n))
                for c in 1:min(n, CAP_H)
                    t = bench_hess(f, x, c)
                    println(io, "$name\t$n\thess\t$c\tfalse\t$t")
                end
                flush(io)
                println(stderr, "done hess $name n=$n")
            end
        end
    end
    return
end

main(joinpath(homedir(), "fd_grid_$(gethostname())$(ELT === Float32 ? "_f32" : "").tsv"))
