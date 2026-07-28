# Brute-force ground truth for the smart candidate selection in pick_chunk:
# benchmark hessian! for every (chunk, simd) — plus the Jet variants — over a
# grid of functions and input sizes, and dump one TSV row per measurement.
# The analysis (analyze_grid.jl) evaluates candidate-set strategies offline
# against this data.
#
# Run (from a HyperHessians checkout, ChunkPicker nested or dev'd):
#   julia --project=benchmark ChunkPicker.jl/benchmark/chunk_grid.jl out.tsv
# GRID_SECONDS (default 0.15) is the per-candidate BenchmarkTools budget.

using HyperHessians, BenchmarkTools
using HyperHessians: HessianConfig, Chunk, Jet
import DiffTests

BenchmarkTools.DEFAULT_PARAMETERS.seconds = parse(Float64, get(ENV, "GRID_SECONDS", "0.15"))

# GRID_ELTYPE=Float32 runs a reduced Float32 grid with chunks up to 24, to
# check whether the doubled SIMD lane count shifts the winning chunk sizes up.
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

const SIZES = MINI ? [4, 8, 12, 16, 24, 32, 48, 64, 100] : [1:12; 14; 16; 20; 24; 32; 48; 64; 100]
const CHUNK_CAP = MINI ? 24 : 16 # beyond this HyperDual tuples spill hopelessly
const JET_CAP = 16   # jet compile time grows steeply past this

function bench_chunk(f, x, c::Int, simd::Bool)
    cfg = HessianConfig(x, Chunk{c}(); simd)
    H = zeros(eltype(x), length(x), length(x))
    return @belapsed HyperHessians.hessian!($H, $f, $x, $cfg)
end

function bench_jet(f, x, simd::Bool)
    cfg = HessianConfig(x, Jet; simd)
    H = zeros(eltype(x), length(x), length(x))
    return @belapsed HyperHessians.hessian!($H, $f, $x, $cfg)
end

function main(out::String)
    open(out, "w") do io
        println(io, "# host=$(gethostname()) elt=$(ELT) julia=$(VERSION) cpu=$(Sys.cpu_info()[1].model) date=$(time())")
        println(io, "func\tn\tkind\tchunk\tsimd\ttime")
        for (name, f) in FUNCS
            for n in SIZES
                x = n == 1 ? ELT[0.55] : collect(range(ELT(0.1), ELT(1.0); length = n))
                chunks = collect(1:min(n, CHUNK_CAP))
                CHUNK_CAP < n <= 20 && push!(chunks, n) # full-vector check
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
                println(stderr, "done $name n=$n")
            end
        end
    end
    return
end

main(length(ARGS) >= 1 ? ARGS[1] : "grid_results.tsv")
