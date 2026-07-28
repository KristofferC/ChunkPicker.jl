# ChunkPicker

Benchmark-driven selection of the optimal *chunk size* (and, for HyperHessians, the
`Jet` vs `HyperDual` representation) for forward-mode automatic differentiation.

Given a function, an input, and an operation, ChunkPicker times every candidate
configuration with [BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl) and
tells you which one is fastest for *your* function.

ADTypes backends are driven through
[DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl)
(a direct dependency; the backend types are re-exported). Load the AD package
itself to make its backend usable:

| Backend                 | Load                  | Operations                                     |
| ----------------------- | --------------------- | ---------------------------------------------- |
| `AutoForwardDiff()`     | `using ForwardDiff`   | `:gradient`, `:jacobian`, `:hessian`, `:hvp`   |
| `AutoHyperHessians()`   | `using HyperHessians` | `:hessian`, `:hvp`                             |
| `HyperHessiansBackend()`| `using HyperHessians` | `:hessian` ((chunks + `Jet`) × simd), `:hvp` (chunks × simd) |

`AutoHyperHessians` sweeps the plain chunk axis through DifferentiationInterface;
the native `HyperHessiansBackend` additionally benchmarks the axes ADTypes cannot
express yet (the `simd` variants and the `Jet` representation) and recommends a
`HessianConfig`/`HVPConfig` instead of a backend.

## Usage

```julia
using ChunkPicker, ForwardDiff

f = x -> sum(abs2, x) + exp(sum(x))
x = rand(50)

res = pick_chunk(AutoForwardDiff(), f, x; op = :gradient)
res.chunk           # fastest chunk size, e.g. 12
res.recommendation  # "AutoForwardDiff(chunksize = 12)"
```

The recommended backend plugs straight into DifferentiationInterface
(`gradient(f, prep, AutoForwardDiff(chunksize = 12), x)` etc.). Chunk size is a
parameter of the ADTypes backend type, which is what makes the sweep possible —
DifferentiationInterface builds its own preparation internally, so there is no
separate config object to pass.

`pick_chunk` prints progress as it benchmarks (disable with `verbose = false`) and
returns a `ChunkPickResult` that shows the full table:

```
ChunkPickResult (AutoForwardDiff, :gradient)
  chunk 1      1.204 μs  2.10x
  ...
* chunk 12     573.0 ns  1.00x
→ AutoForwardDiff(chunksize = 12)
```

### Hessian–vector products

`op = :hvp` benchmarks Hessian–vector products; pass the tangent (or a tuple of
tangents for bundled directions) with `tangents`:

```julia
pick_chunk(AutoForwardDiff(), f, x; op = :hvp, tangents = rand(50))
pick_chunk(HyperHessiansBackend(), f, x; op = :hvp, tangents = (v1, v2))
```

### Jacobian / Hessian

```julia
g = x -> [sum(sin, x), prod(x)]          # R^n -> R^m
pick_chunk(AutoForwardDiff(), g, x; op = :jacobian)
pick_chunk(AutoForwardDiff(), f, x; op = :hessian)
```

### HyperHessians: HyperDual chunks vs Jet

For HyperHessians the sweep also includes the symmetric `Jet` representation (a single
evaluation of the whole Hessian, storing the gradient once and only the upper
triangle) when the loaded version provides it. HyperHessians never selects `Jet` on
its own — whether it beats the best `HyperDual` chunk depends on the function, input
length, and CPU — so this sweep is how you decide:

```julia
using ChunkPicker, HyperHessians

res = pick_chunk(HyperHessiansBackend(), f, rand(4))
# ...
#   chunk 4           36.8 ns  1.26x
#   chunk 4 simd     118.8 ns  4.06x
#   Jet               36.4 ns  1.24x
# * Jet simd          29.2 ns  1.00x
# → HyperHessians.HessianConfig(x, HyperHessians.Jet; simd = true)

res.kind  # :chunk or :jet — which representation won
```

The `Jet` is benchmarked with and without SIMD.Vec-forced arithmetic (like the
chunked configs, see below). Because the jet's unrolled triangle makes compile time
grow steeply with `length(x)`, the candidate is only included up to `length(x) <= 32`
by default; pass `jet = <n>` (or `jet = true` for no cap) to raise it. `Jet` requires
a HyperHessians version that defines it
([PR #55](https://github.com/KristofferC/HyperHessians.jl/pull/55) or later); on older
versions the Jet candidates are skipped. Disable them explicitly with `jet = false`.

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

## Smart candidate selection

For `:hessian`/`:hvp` the default `chunks = :smart` benchmarks only the chunk sizes
that ever win, instead of a dense `1:12` sweep. The set was derived from a
brute-force grid (6 function families × 20 input sizes × chunks 1:16 × simd on/off,
on AVX2, AVX-512 and NEON — see `benchmark/RESULTS.md`): a small base set
`{2, 3, 4, 6, 8, 12, 16}`, the full-vector `n` while `n ≤ 16`, `⌈n/2⌉`/`⌈n/3⌉`
(fewest evaluations per dual size), and divisors of `n` in `4:16` (no padded
trailing chunk); everything for `n ≤ 4`. Across the measured grid the best
candidate in this set is within 2% of the exhaustive optimum in 99% of cases
(worst 9%), with ~40% fewer benchmarks than the dense sweep — which itself misses
some of the true winners (e.g. full-vector at `n = 13..16`).

Pass `chunks = :all` for the exhaustive brute-force sweep, or an explicit iterable
of sizes.

## Keywords

- `op`       — operation (default `:gradient` for `AutoForwardDiff`, `:hessian` for HyperHessians).
- `chunks`   — candidate chunk sizes: `:smart` (default for `:hessian`/`:hvp`, see above),
  `:all` (exhaustive), or an explicit iterable. `:gradient`/`:jacobian` default to
  `1:min(length(x), 32)`.
- `tangents` — for `op = :hvp`: the tangent vector or tuple of tangents (default: ones).
- `seconds`  — per-candidate benchmark budget (default `0.5`).
- `verbose`  — print progress (default `true`).
- `jet`      — HyperHessians only: max input length for which the `Jet` variants are
  included (default `32`), or `true`/`false` to force/disable them.
- `simd`     — HyperHessians only: also benchmark SIMD.Vec-forced variants (default `true`).

## Result

`ChunkPickResult` fields: `chunk`, `kind` (`:chunk`/`:jet`), `simd`, `timings`
(`Vector{ChunkTiming}` with `chunk`, `kind`, `simd`, `time` in seconds), `backend`, `op`,
and a ready-to-use `recommendation` string.
