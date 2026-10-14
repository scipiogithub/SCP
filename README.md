# Exact decomposition algorithms for the Student Clustering Problem

Code and data for:

> T. Cura. *Exact decomposition algorithms for the student clustering problem.*
> European Journal of Operational Research (accepted, pending final revision).

The repository accompanies the paper under the EJOR policy on computational
experiments. It contains the two exact decomposition algorithms, the
heterogeneous-cost experiment of Section 6.4, the monolithic MILP used as a
baseline, and the opening-cost vectors generated for that experiment.

## Contents

| File | Contents |
|---|---|
| `code/nogood_singletree.jl` | Combinatorial (no-good) method, EXACT_NG; the monolithic MILP baseline (`opt_all`, `main_milp`); instance reader; pricing phase |
| `code/benders_singletree.jl` | Branch-and-Benders-cut method, EXACT_B |
| `code/benders_general.jl` | Heterogeneous-cost experiment of Section 6.4, and `export_costs` |
| `instances/heterogeneous_opening_costs.txt` | The opening-cost vectors actually used in Section 6.4 |

Each Julia file is self-contained: all three define their own instance reader and
pricing phase, so any one of them can be `include`d on its own.

---

## Requirements

| | |
|---|---|
| Julia | 1.12.5 |
| Julia packages | `JuMP`, `CPLEX` (CPLEX.jl), `Random`, `Printf` |
| Solver | IBM ILOG CPLEX 22.1.1 |

**CPLEX is commercial software and is not distributed here.** A licence is
required; academic licences are available from IBM. `CPLEX.jl` must be built
against the same installation.

All reported runs used a **single thread** and a **zero relative MIP gap**.

---

## Instance data

The 121 benchmark instances are **not redistributed here**: they were introduced
by Sá & Prata (2026) and are available from

> https://github.com/NayzaMamede/students-clustering-problem

Put them in a folder named `SCP` **beside the Julia files** — that is, in the same
directory as `nogood_singletree.jl` — so that instance *i* is at
`SCP/SCP_inst<i>.txt` relative to that directory. If you keep the layout of this
repository, that means `code/SCP/SCP_inst<i>.txt`.

The reader resolves instance paths with `@__DIR__`, the directory of the Julia
source file, rather than against the working directory. So what matters is where
the `.jl` file sits, not where you start Julia: the same call works from the
repository root or from inside `code/`. The flip side is that if you copy a
single `.jl` file somewhere on its own, the `SCP` folder has to follow it.

Instance 121 is the real municipal network of Fortaleza.

The **opening-cost vectors** for the heterogeneous-cost experiment of Section 6.4
are new data produced by this study and are included here as
`instances/heterogeneous_opening_costs.txt`. Each release of this repository is archived on
Zenodo with its own DOI; the paper cites the DOI of the release that produced the
reported results.

The vectors can be regenerated with `export_costs()`. They are published rather
than the seed alone, because the drawn values depend on the random stream of the
Julia version used *and* on Δ, which is itself obtained by solving the subproblem.

---

## Reproducing the reported tables

With the benchmark instances in place, `include` the file you need and call the
listed function. From the repository root:

```julia
include("code/nogood_singletree.jl")
main()
```

| Table | File | Command |
|---|---|---|
| 2 — Version 1 vs Version 2 | `code/nogood_singletree.jl` | `main()` for Version 1; `main(use_weights = false)` for Version 2 |
| 3, S2 — EXACT_NG column | `code/nogood_singletree.jl` | `main()` |
| 3, S2 — EXACT_B column | `code/benders_singletree.jl` | `main(instances = 1:121)` |
| 3, S2 — MILP column | `code/nogood_singletree.jl` | `main_milp()` |
| 4, 5, S3 | — | derived from the objective values of the three runs above and the published metaheuristic results |
| 6, 7, S4 | `code/benders_general.jl` | `main_general(cost_scale = 1.0)`, then `2.0`, then `5.0` |
| Cost-vector file | `code/benders_general.jl` | `export_costs()` |

Notes on the commands:

* `benders_singletree.jl`'s `main` defaults to a reduced instance list (`OPEN`);
  pass `instances = 1:121` for the full reported run.
* `export_costs` writes its output file to the **working directory**, not to the
  code directory; the copy committed here sits in `instances/`.
* `export_costs` deliberately lives inside `benders_general.jl` rather than in a
  script of its own, so that it consumes the random stream in exactly the same
  order as `main_general`. Moving it would silently change the exported values.

### The MILP baseline

The MILP column of Table 3 and Table S2 comes from `opt_all` in
`nogood_singletree.jl` — the Julia formulation of the monolithic model of
Eqs. (1)–(6), solved directly by CPLEX under the same settings as everything else
(zero gap, 105 s, single thread). `opt_all` returns only the objective value and
the optimality flag, so the timing was taken around the call; the wrapper
`main_milp()` performs that run and prints `i obj total_t proven`.

Two details matter for reproduction:

* The large constant follows the same convention as the exact methods:
  `f = 1e6` for the benchmark instances and `f = 1e4` for instance 121. The
  reported `z*` includes the `f·|Y|` term — for instance 121, 57,480 is five
  schools at 10,000 plus 7,480 of distance.
* `total_t` is wall-clock around the whole call and so includes model
  construction; it can slightly exceed the 105 s solver limit, which is why
  instance 121 is listed at 105.36 s.

### Table 1

The figures in **Table 1** — the LP relaxation bound, the integrality rate and the
Proposition 1 check on twelve instances — were obtained by querying the solver
directly rather than through a script, and no script for them is included here.
The quantities are defined in Section 6.1 of the paper.

---

## Solver settings as run

* Single thread (`CPX_PARAM_THREADS = 1`) everywhere, including the MILP baseline.
* Zero relative MIP gap (`CPX_PARAM_EPGAP = 0`) on every master and on the MILP.
* Overall wall-clock limit 105 s per instance (`CPX_PARAM_TILIM`).
* Grade-wise transportation subproblems: dual simplex (`CPX_PARAM_LPMETHOD = 2`),
  60 s internal safety limit each.
* Attractiveness-guided warm start of the branch-and-Benders-cut method: 30 s.
* **Presolve differs between the two methods, by design.** The no-good master sets
  `CPX_PARAM_REDUCE = 1` (primal-only reductions), because its lazy cuts exclude
  configurations that are feasible for the original problem, so dual reductions
  are not safe. The branch-and-Benders-cut master retains the default presolve:
  its optimality cuts are valid inequalities and exclude no feasible solution.
  This is the only difference between the two configurations besides the cut
  itself, and it is reported in the paper.

---

## Replication notes

* **Instance 120 is borderline.** It certifies optimality at ≈104.6 s of the
  105 s budget, so on slower hardware it may come back uncertified, changing the
  certification count from 91 to 90.
* **Time-to-best is hardware-dependent** by nature. Total solution time and the
  certification counts are the measures to compare across machines.
* Objective values in the deposited heterogeneous-cost output are printed to one
  decimal. At `c = 1`, instance 28 therefore prints as 1787.8 against the MILP's
  1787.7, although the run recorded the two as equal within tolerance; this is
  noted in the paper (Table 7 footnote).
* Reference machine: Apple M2 Pro (12-core, 3.48 GHz), 16 GB RAM, macOS Ventura.

---

## Citation

> T. Cura. *Exact decomposition algorithms for the student clustering problem.*
> European Journal of Operational Research (forthcoming).

The volume, pages and article DOI will be added once the paper is published.

## Licence

The code in this repository, and the accompanying file of opening-cost vectors,
are released under the MIT Licence; see `LICENSE`.

The 121 benchmark instances are **not** covered by that licence. They are
third-party data, are not redistributed here, and remain governed by the terms of
their original source.
