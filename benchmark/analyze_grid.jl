# Offline analysis of chunk_grid.jl TSVs: rank the true winners per
# (host, func, n) and score candidate-set strategies by regret
# (best-in-set time / true best time) and candidate count.
#
#   julia ChunkPicker.jl/benchmark/analyze_grid.jl results/*.tsv

using Printf

struct Row
    host::String
    func::String
    n::Int
    kind::Symbol
    chunk::Int
    simd::Bool
    time::Float64
end

function load(files)
    rows = Row[]
    for file in files
        host = "?"
        for line in eachline(file)
            if startswith(line, "#")
                m = match(r"host=(\S+)", line)
                m !== nothing && (host = m.captures[1])
                continue
            end
            startswith(line, "func\t") && continue
            f, n, kind, c, s, t = split(line, '\t')
            push!(rows, Row(host, f, parse(Int, n), Symbol(kind), parse(Int, c), parse(Bool, s), parse(Float64, t)))
        end
    end
    return rows
end

cases(rows) = sort!(unique((r.host, r.func, r.n) for r in rows))

function report_winners(rows; top = 3)
    for (host, func, n) in cases(rows)
        rs = sort!([r for r in rows if r.host == host && r.func == func && r.n == n]; by = r -> r.time)
        best = rs[1]
        line = @sprintf("%-16s %-12s n=%-4d", host, func, n)
        for r in rs[1:min(top, length(rs))]
            label = r.kind === :jet ? "Jet" : "c$(r.chunk)"
            label *= r.simd ? "+s" : ""
            line *= @sprintf("  %-8s %6.2fx", label, r.time / best.time)
        end
        println(line)
    end
    return
end

# A strategy maps n -> Vector{(kind, chunk, simd)}. Regret for a case is
# time(best candidate present in the grid) / time(true best in the grid);
# candidates missing from the grid are ignored.
function score(rows, strategy; verbose = false)
    regrets = Float64[]
    counts = Int[]
    worst = ("", "", 0, 1.0)
    for (host, func, n) in cases(rows)
        rs = [r for r in rows if r.host == host && r.func == func && r.n == n]
        best = minimum(r.time for r in rs)
        cand = strategy(n)
        push!(counts, length(cand))
        avail = [r for r in rs if (r.kind, r.chunk, r.simd) in cand || (r.kind === :jet && (:jet, n, r.simd) in cand)]
        isempty(avail) && error("strategy produced no measurable candidate for n=$n")
        t = minimum(r.time for r in avail)
        reg = t / best
        push!(regrets, reg)
        if reg > worst[4]
            worst = (host, func, n, reg)
        end
        verbose && reg > 1.05 && @printf(
            "  regret %.2fx  %s %s n=%d (best %s)\n", reg, host, func, n,
            let b = rs[argmin([r.time for r in rs])]
                "$(b.kind) c=$(b.chunk) simd=$(b.simd)"
            end
        )
    end
    sort!(regrets)
    p(q) = regrets[clamp(ceil(Int, q * length(regrets)), 1, length(regrets))]
    @printf(
        "cases=%d  candidates: mean=%.1f max=%d  regret: mean=%.3f p90=%.3f p99=%.3f max=%.3f",
        length(regrets), sum(counts) / length(counts), maximum(counts),
        sum(regrets) / length(regrets), p(0.9), p(0.99), regrets[end]
    )
    @printf("  worst: %s %s n=%d\n", worst[1], worst[2], worst[3])
    return
end

## Strategies

# What pick_chunk does today: every chunk 1:min(n, 12), x2 for simd, + Jet x2.
function current(n)
    cand = Tuple{Symbol, Int, Bool}[]
    for c in 1:min(n, 12), s in (false, true)
        push!(cand, (:chunk, c, s))
    end
    n >= 1 && push!(cand, (:jet, n, false), (:jet, n, true))
    return cand
end

# Exhaustive over everything the grid measured (regret 1.0 by construction).
function exhaustive(n)
    cand = Tuple{Symbol, Int, Bool}[]
    for c in 1:min(n, 16), s in (false, true)
        push!(cand, (:chunk, c, s))
    end
    16 < n <= 20 && for s in (false, true)
        push!(cand, (:chunk, n, s))
    end
    push!(cand, (:jet, n, false), (:jet, n, true))
    return cand
end

if abspath(PROGRAM_FILE) == @__FILE__
    rows = load(ARGS)
    println("== top-3 per case ==")
    report_winners(rows)
    println("\n== strategies ==")
    print("current:    ")
    score(rows, current)
    print("exhaustive: ")
    score(rows, exhaustive)
end
