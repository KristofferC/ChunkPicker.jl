# Self-contained version of chunk_grid.jl: bootstraps a temp environment with
# HyperHessians#kc/jet from GitHub, so it runs on any machine with just Julia.
#
#   julia chunk_grid_standalone.jl
#
# Writes ~/chunk_grid_<host>.tsv and prints progress. GRID_SECONDS (default
# 0.15) is the per-candidate BenchmarkTools budget.

import Pkg
Pkg.activate(mktempdir())
Pkg.add(
    [
        Pkg.PackageSpec(url = "https://github.com/KristofferC/HyperHessians.jl", rev = "kc/jet"),
        Pkg.PackageSpec(name = "BenchmarkTools"),
        Pkg.PackageSpec(name = "DiffTests"),
    ]
)

using HyperHessians, BenchmarkTools
using HyperHessians: HessianConfig, Chunk, Jet
import DiffTests

BenchmarkTools.DEFAULT_PARAMETERS.seconds = parse(Float64, get(ENV, "GRID_SECONDS", "0.15"))

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

const SIZES = [1:12; 14; 16; 20; 24; 32; 48; 64; 100]
const CHUNK_CAP = 16
const JET_CAP = 16

function bench_chunk(f, x, c::Int, simd::Bool)
    cfg = HessianConfig(x, Chunk{c}(); simd)
    H = zeros(length(x), length(x))
    return @belapsed HyperHessians.hessian!($H, $f, $x, $cfg)
end

function bench_jet(f, x, simd::Bool)
    cfg = HessianConfig(x, Jet; simd)
    H = zeros(length(x), length(x))
    return @belapsed HyperHessians.hessian!($H, $f, $x, $cfg)
end

function main(out::String)
    ncases = length(FUNCS) * length(SIZES)
    done = 0
    open(out, "w") do io
        println(io, "# host=$(gethostname()) julia=$(VERSION) cpu=$(Sys.cpu_info()[1].model) date=$(time())")
        println(io, "func\tn\tkind\tchunk\tsimd\ttime")
        for (name, f) in FUNCS
            for n in SIZES
                x = n == 1 ? [0.55] : collect(range(0.1, 1.0; length = n))
                chunks = collect(1:min(n, CHUNK_CAP))
                CHUNK_CAP < n <= 20 && push!(chunks, n)
                for c in chunks, simd in (false, true)
                    t = bench_chunk(f, x, c, simd)
                    println(io, "$name\t$n\tchunk\t$c\t$simd\t$t")
                end
                if n <= JET_CAP
                    for simd in (false, true)
                        t = bench_jet(f, x, simd)
                        println(io, "$name\t$n\tjet\t$n\t$simd\t$t")
                    end
                end
                flush(io)
                done += 1
                println(stderr, "[$done/$ncases] $name n=$n")
            end
        end
    end
    println("\nDONE -> $out")
    println("send it back with e.g.:  gzip -c9 $out | base64")
    return
end

main(joinpath(homedir(), "chunk_grid_$(gethostname()).tsv"))
