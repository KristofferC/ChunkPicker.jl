# ChunkPicker

Benchmark-driven selection of the optimal *chunk size* (and, for HyperHessians, the
`Jet` vs `HyperDual` representation) for forward-mode automatic differentiation.

Given a function, an input, and an operation, ChunkPicker times every candidate
configuration with [BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl) and
tells you which one is fastest for *your* function.

The AD backends are loaded through package extensions, so ChunkPicker itself only
depends on BenchmarkTools. A backend becomes available when you load its package:

| Backend                 | Load           | Operations                        |
| ----------------------- | -------------- | --------------------------------- |
| `ForwardDiffBackend()`  | `using ForwardDiff`   | `:gradient`, `:jacobian`, `:hessian` |
| `HyperHessiansBackend()`| `using HyperHessians` | `:hessian` (HyperDual chunks + `Jet`) |

## Usage

```julia
using ChunkPicker, ForwardDiff

f = x -> sum(abs2, x) + exp(sum(x))
x = rand(50)

res = pick_chunk(ForwardDiffBackend(), f, x; op = :gradient)
res.chunk           # fastest chunk size, e.g. 12
res.recommendation  # "ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{12}())"
```

`pick_chunk` prints progress as it benchmarks (disable with `verbose = false`) and
returns a `ChunkPickResult` that shows the full table:

```
ChunkPickResult (ForwardDiff, :gradient)
  chunk 1      1.204 μs  2.10x
  ...
* chunk 12     573.0 ns  1.00x
→ ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{12}())
```

### Jacobian / Hessian

```julia
g = x -> [sum(sin, x), prod(x)]          # R^n -> R^m
pick_chunk(ForwardDiffBackend(), g, x; op = :jacobian)
pick_chunk(ForwardDiffBackend(), f, x; op = :hessian)
```

### HyperHessians: HyperDual chunks vs Jet

For HyperHessians the sweep also includes the symmetric `Jet` representation (a single
evaluation of the whole Hessian) when the loaded version provides it, so you can see
whether a `Jet` beats the best `HyperDual` chunk:

```julia
using ChunkPicker, HyperHessians

res = pick_chunk(HyperHessiansBackend(), f, rand(4))
# ...
# * chunk 4       87.4 ns  1.00x
#   Jet          98.3 ns  1.12x
# → HyperHessians.HessianConfig(x, HyperHessians.Chunk{4}())

res.kind  # :chunk or :jet — which representation won
```

`Jet` requires a HyperHessians version that defines it
([PR #55](https://github.com/KristofferC/HyperHessians.jl/pull/55) or later); on older
versions the Jet candidate is skipped. Disable it explicitly with `jet = false`.

### HyperHessians: SIMD variants

When the loaded HyperHessians supports the `simd` config option and the eltype is
`Float32`/`Float64`, each chunk size is additionally benchmarked with SIMD.Vec-forced
arithmetic and the recommendation includes the flag when it wins:

```julia
res = pick_chunk(HyperHessiansBackend(), f, rand(48))
# ...
#   chunk 8         45.8 μs  1.24x
# * chunk 8 simd    36.9 μs  1.00x
# → HyperHessians.HessianConfig(x, HyperHessians.Chunk{8}(); simd = true)

res.simd  # whether the winning variant uses simd = true
```

Disable with `simd = false`.

## Keywords

- `op`      — operation (default `:gradient` for ForwardDiff, `:hessian` for HyperHessians).
- `chunks`  — candidate chunk sizes. Defaults are capped since the useful range is
  small: `1:min(length(x), 32)` for `:gradient`/`:jacobian`, `1:min(length(x), 12)` for
  `:hessian`. Pass an explicit range to sweep further.
- `seconds` — per-candidate benchmark budget (default `0.5`).
- `verbose` — print progress (default `true`).
- `jet`     — HyperHessians only: include the `Jet` variant (default `true`).
- `simd`    — HyperHessians only: also benchmark SIMD.Vec-forced variants (default `true`).

## Result

`ChunkPickResult` fields: `chunk`, `kind` (`:chunk`/`:jet`), `simd`, `timings`
(`Vector{ChunkTiming}` with `chunk`, `kind`, `simd`, `time` in seconds), `backend`, `op`,
and a ready-to-use `recommendation` string.
