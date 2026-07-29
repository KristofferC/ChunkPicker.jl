# Smart chunk-candidate selection (2026-07-28)

Question: `pick_chunk`'s dense sweep (`1:min(n, 12)` × simd + Jet) spends most
of its time on candidates that are never optimal — e.g. chunk 9 or 11 for
`n = 20`. Which candidates can be skipped without giving up the optimum?

## Method

`chunk_grid.jl` brute-forces `hessian!` over:

- 6 function families: rosenbrock (cheap, coupled), ackley (transcendental
  reductions), self_weighted_logit (expensive per element), sumexp (very
  cheap), logbarrier, chaincouple (neighbor-coupled tanh)
- input sizes `n ∈ 1:12, 14, 16, 20, 24, 32, 48, 64, 100`
- chunks `1:16` (+ full-vector `n` for `n ≤ 20`), each with `simd` off/on
- `Jet` off/on `simd` for `n ≤ 16`

on three machines: Intel Ultra 7 265K "corem-uppsala" (Arrow Lake, AVX2),
AMD EPYC 9354 "demeter6" (Zen 4, AVX-512), Apple M4 Pro (NEON). 2628 rows
per machine; `analyze_grid.jl` scores candidate-set strategies offline
against the per-case true optimum (`chunk_grid_standalone.jl` is a
self-bootstrapping copy for machines without a checkout).

## Winner structure (Float64, 360 cases)

- `n ≤ 4`: `Jet`/`Jet+simd` and tiny chunks; everything is cheap to test.
- `5 ≤ n ≤ 16`: the winner is one of `Jet` (expensive f — wins up to n = 16
  for logit on M4 and Zen 4), full-vector `c = n` (often `+simd`; the dense
  `1:12` sweep *misses* these entirely for n = 13..16), or a small chunk
  `{2, 3, 4}` (cheap f).
- `n ≥ 20`: `{4, 8, 12, 16}`(±simd) for expensive f — bigger on wider
  registers — `{3, 4}` for cheap f, plus two structural families:
  `⌈n/2⌉`/`⌈n/3⌉` (fewest evaluations per dual width; e.g. `c7` at `n = 20`
  is within 0–6% of optimal, same eval count as `c8` with smaller duals) and
  divisors of `n` (no padded trailing chunk; `c10` at `n = 100`).
- `Jet+simd` never wins past `n = 7` (the triangle becomes one long `Vec`).
- Padding vs alignment at equal eval count is arch-dependent (`[3,3,3]` beats
  `[4,4,1]` on M4, ties/flips on Zen 4), so both shapes stay as candidates.

## The rule (`smart_chunks(n)`)

`n ≤ 4`: all of `1:n`. Otherwise the union of `{2, 3, 4, 6, 8, 12, 16}`,
`n` while `n ≤ 16`, `⌈n/2⌉` and `⌈n/3⌉` (≤ 16), and divisors of `n` in
`4:16` — each ± simd; `Jet` for `n ≤ 16`, `Jet+simd` for `n ≤ 7`.

Scores over the 360 measured cases (regret = best-in-set / true best):

| strategy        | mean candidates | regret mean | p99   | max   |
| --------------- | --------------- | ----------- | ----- | ----- |
| dense (old)     | 19.4            | 1.002       | 1.062 | 1.086 |
| smart           | 12.4            | 1.000       | 1.020 | 1.086 |
| exhaustive      | 22.5            | 1.000       | 1.000 | 1.000 |

Both non-exhaustive worst cases are the same single point (full-vector+simd
at `n = 20` on Zen 4, where `⌈n/2⌉+simd` is 9% behind). The smart set
strictly dominates the old dense sweep: fewer candidates, lower regret.
`chunks = :all` selects the exhaustive sweep.

## Float32

Measured with `GRID_ELTYPE=Float32` (reduced grid: 9 sizes, chunks to 24)
on the same three machines. The doubled SIMD lane count moves winners up,
most on wide registers: `c16+simd` dominates large n on Zen 4, `c20` wins
where it divides `n = 100`, `c24` appears as runner-up (full-vector at 24),
and full-vector wins extend to `n = 16`; NEON stays close to the Float64
picture. `smart_chunks(n, Float32)` therefore raises the caps (full-vector,
frontier, divisors) from 16 to 24 and drops 2 from the base set: exactly
optimal in all 162 measured Float32 cases, mean 12.9 candidates. The
Float64 rule applied to Float32 data has max regret 1.16 (the `c20`
divisor case).

## HyperHessians' default chunk

The same data scores HyperHessians' function-agnostic default
(`pickchunksize`; its action space is a plain non-simd chunk), against the
best non-simd chunk per case:

| eltype  | rule                       | geo   | p90  | max  |
| ------- | -------------------------- | ----- | ---- | ---- |
| Float64 | `min(n, 8)` (old)          | 1.444 | 3.10 | 5.72 |
| Float64 | `n≤10: n, n≤32: 4, else 6` | 1.137 | 1.64 | 2.51 |
| Float32 | `min(n, 8)` (old)          | 1.359 | 2.67 | 4.04 |
| Float32 | `n≤12: n, else 6`          | 1.231 | 1.79 | 2.32 |

Chunk 8 is disastrous around `n = 9..12` (geomean regret 2.3–2.9x there:
e.g. `n = 9` does three evaluations with 8-wide duals where full-vector
does one). The remaining ~1.2x is irreducible without knowing the function
— cheap functions want tiny chunks, expensive ones want wide — which is
what `pick_chunk` is for.

Arch-awareness was evaluated and rejected: per machine, the arch-blind rule
is within 1% of each machine's own best rule for Float64 (AVX2 and AVX-512
are indistinguishable in the default's non-simd action space; arch
differences concentrate in the simd variants and the Jet). The only real
split is Float32 on NEON, which prefers a tail of 4 over 6 (~13% geomean)
— judged not worth an arch branch, and measured on a single ARM chip.

## ForwardDiff (2026-07-29)

`fd_grid.jl` repeats the exercise for `ForwardDiff.gradient!` (chunks 1:32,
sizes to 200) and `ForwardDiff.hessian!` (chunks 1:16), Float64 + a Float32
mini, on all three machines. The economics differ from HyperHessians:

- gradient: per-evaluation cost is only O(chunk), so plateaus are wide and
  winners sit high — per-function sweet spots at `{4, 5}` / `{8, 9, 12}` /
  `{24..32}` (cheap rosenbrock), and full-vector wins through `n = 32` for
  cheap functions. The set `{3, 4, 5, 8, 12, 16, 24, 32}` + full-vector
  `n ≤ 32` (~6 candidates vs the old dense 32) scores geo 1.010 / max 1.18
  across all three machines (Float32: 1.007 / 1.15).
- hessian: Dual-of-Dual pays its per-evaluation overhead hardest, so the
  *largest* chunks win broadly (`c16` even at `n = 48..64`), with the odd
  winners (`c11`, `c13`, `c15`) being exactly the `⌈n/k⌉` frontier — except
  cheap functions on Zen 4, where `c6` wins outright at every large `n`.
  The set `{4, 6, 8, 12, 16}` + full-vector `n ≤ 16` + frontier `k = 2..7`
  + divisors (~6 candidates) scores geo 1.001 / max 1.06 across all three
  machines and both eltypes. The HyperHessians-derived set does *not*
  transfer (max 1.29).
- hvp (mini-measured through DifferentiationInterface): between the two —
  full-vector wins to `n = 32`, frontier/divisor chunks at large `n`, and a
  tiny-chunk tail (`c2`) for very cheap functions.

ForwardDiff's own default (`pickchunksize`: equal split ≤ 12) scored against
the best chunk per case: hessian geo 1.06 / max 1.47 (decent), gradient geo
1.32 / max 3.26 and Float32 gradient geo 1.62 (weak — the per-function
spread from `{4, 5}` to `{24..32}` is unbridgeable function-agnostically).

## Data

`results/grid_<host>[_f32].tsv` and `results/fd_grid_<host>[_f32].tsv`, one
row per measurement (`func n kind chunk simd time`; for the ForwardDiff
grids `kind` is the op, grad/hess).
