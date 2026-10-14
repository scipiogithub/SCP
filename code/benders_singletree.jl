# benders_singletree.jl
# -----------------------------------------------------------------------------
# SINGLE-TREE branch-and-Benders-cut for the SCP, warm-started by the
# attractiveness-guided (no-good) incumbent, plus a calibration harness for the
# warm-start budget.
#
# PERFORMANCE-CRITICAL: the transportation subproblem is built ONCE per instance
# and only its right-hand sides (capacities = w_jk * yhat_j) are updated on each
# lazy-callback evaluation, then re-solved with warm-started dual simplex. This
# avoids rebuilding a large LP (up to m*n variables) at every integer node.
#
# NOTE: cannot be run here (no Julia/CPLEX). Run on your machine.
#   Validate:   main(crosscheck=true, instances=1:8)
#   Calibrate:  calibrate()                     # seeded random training set + sweep
#   Report:     main(instances=OPEN)            # with the chosen warm_budget
# -----------------------------------------------------------------------------

using JuMP
using CPLEX
using Printf
using Random

struct SCPInstance
    m::Int; n::Int; p::Int
    d::Matrix{Float64}; q::Matrix{Bool}; w::Matrix{Int}
end

function read_scp_instance(filepath::String)::SCPInstance
    filepath = joinpath(@__DIR__, filepath)
    lines = readlines(filepath); idx = 1
    skip_empty() = while idx <= length(lines) && strip(lines[idx]) == ""; idx += 1; end
    skip_to_section(h) = while idx <= length(lines) && !startswith(strip(lines[idx]), h); idx += 1; end
    m = parse(Int, split(lines[1], ":")[2]); n = parse(Int, split(lines[2], ":")[2]); p = parse(Int, split(lines[3], ":")[2]); idx = 4
    skip_to_section("Distancias"); idx += 1
    distances = Matrix{Float64}(undef, m, n)
    for i in 1:m; skip_empty(); distances[i, :] = parse.(Float64, split(strip(lines[idx]))); idx += 1; end
    skip_to_section("Demandas"); idx += 1
    assignments = Matrix{Bool}(undef, m, p)
    for i in 1:m; skip_empty(); assignments[i, :] = parse.(Bool, split(strip(lines[idx]))); idx += 1; end
    skip_to_section("Capacidades"); idx += 1
    capacities = Matrix{Int}(undef, n, p)
    for j in 1:n; skip_empty(); capacities[j, :] = parse.(Int, split(strip(lines[idx]))); idx += 1; end
    return SCPInstance(m, n, p, distances, assignments, capacities)
end

# --- Persistent transportation subproblem, DECOMPOSED BY GRADE -----------------
# The SSAP separates across grades (each student is in exactly one grade), so we
# build one small transportation LP per grade (|M_k| x n) instead of a single
# m x n LP. Each is built once; only capacity RHS = w_jk*yhat_j is updated per
# call, then re-solved with warm-started dual simplex. Much faster and more
# stable on the largest instances (avoids per-node freezes inside the callback).
struct Sub
    models::Vector{Model}
    xs::Vector{Matrix{VariableRef}}
    assigns::Vector{Vector{ConstraintRef}}
    caps::Vector{Vector{ConstraintRef}}     # caps[k][j]: capacity of school j for grade k
    Mk::Vector{Vector{Int}}                 # students in each grade
end

function make_sub(inst::SCPInstance)::Sub
    m, n, p, d, q, w = inst.m, inst.n, inst.p, inst.d, inst.q, inst.w
    models = Model[]; xs = Matrix{VariableRef}[]
    assigns = Vector{ConstraintRef}[]; caps = Vector{ConstraintRef}[]; Mk = Vector{Int}[]
    for k in 1:p
        idx = [i for i in 1:m if q[i, k]]         # students in grade k
        push!(Mk, idx); mk = length(idx)
        sp = Model(CPLEX.Optimizer)
        set_optimizer_attribute(sp, "CPX_PARAM_THREADS", 1)
        set_optimizer_attribute(sp, "CPX_PARAM_LPMETHOD", 2)
        set_optimizer_attribute(sp, "CPX_PARAM_TILIM", 60)    # safety net; TU LP solves fast
        set_silent(sp)
        @variable(sp, x[1:mk, 1:n] >= 0)
        @objective(sp, Min, sum(d[idx[r], j] * x[r, j] for r in 1:mk, j in 1:n))
        assign = @constraint(sp, [r = 1:mk], sum(x[r, j] for j in 1:n) == 1)
        cap = @constraint(sp, [j = 1:n], sum(x[r, j] for r in 1:mk) <= 0.0)
        push!(models, sp); push!(xs, x)
        push!(assigns, Vector{ConstraintRef}(assign)); push!(caps, Vector{ConstraintRef}(cap))
    end
    return Sub(models, xs, assigns, caps, Mk)
end

# Evaluate all grade subproblems for a fixed yhat; returns (dist, feasible,
# const_term, coef) with the Benders optimality cut  eta >= const_term + sum_j coef[j]*Y[j].
function eval_sub!(S::Sub, inst::SCPInstance, yhat::Vector{Float64})
    n, p, w = inst.n, inst.p, inst.w
    dist = 0.0; ct = 0.0; coef = zeros(n)
    for k in 1:p
        sp = S.models[k]; cap = S.caps[k]; assign = S.assigns[k]
        for j in 1:n
            set_normalized_rhs(cap[j], w[j, k] * yhat[j])
        end
        optimize!(sp)
        termination_status(sp) == MOI.OPTIMAL || return (Inf, false, 0.0, zeros(n))
        dist += objective_value(sp)
        α = dual.(assign); β = dual.(cap)
        ct += sum(α)
        for j in 1:n
            coef[j] += β[j] * w[j, k]
        end
    end
    return (dist, true, ct, coef)
end

# --- Warm start: best incumbent from the no-good scheme within `budget` seconds -
function warmstart_nogood(inst::SCPInstance, f::Float64, tstart::Float64, budget::Float64, S::Sub)
    m, n, p, w, q = inst.m, inst.n, inst.p, inst.w, inst.q
    τ = sum(q, dims = 1)[:]
    # pricing: all schools open -> per-school counts -> attractiveness weights
    eval_sub!(S, inst, ones(n))
    a = zeros(n)
    for k in 1:p
        xk = value.(S.xs[k])
        for j in 1:n
            a[j] += sum(@view xk[:, j])
        end
    end
    ω = maximum(a) .- a
    # attractiveness master + no-good enumeration bounded by `budget`
    mp = Model(CPLEX.Optimizer); set_silent(mp); set_optimizer_attribute(mp, "CPX_PARAM_THREADS", 1)
    set_optimizer_attribute(mp, "CPX_PARAM_TILIM", 30)   # safety: bound each master solve
    @variable(mp, Y[1:n], Bin)
    @objective(mp, Min, f * sum(Y) + sum(ω[j] * Y[j] for j in 1:n))
    @constraint(mp, [k = 1:p], sum(w[j, k] * Y[j] for j in 1:n) >= τ[k])
    optimize!(mp)
    yb = round.(Int, value.(Y)); κ = sum(yb)
    d0, _, _, _ = eval_sub!(S, inst, Float64.(yb)); bestdist = d0; bestY = copy(yb)
    while time() - tstart <= budget
        @constraint(mp, sum((yb[j] == 1 ? 1 - Y[j] : Y[j]) for j in 1:n) >= 1)
        optimize!(mp)
        (termination_status(mp) == MOI.OPTIMAL && sum(round.(Int, value.(Y))) == κ) || break
        yb = round.(Int, value.(Y))
        dist, _, _, _ = eval_sub!(S, inst, Float64.(yb))
        if dist < bestdist; bestdist = dist; bestY = copy(yb); end
    end
    return bestY, bestdist
end

# --- Single-tree branch-and-Benders-cut -------------------------------------
function benders_singletree(inst::SCPInstance, tstart::Float64, f::Float64; tlim = 105.0, warm_budget = 30.0, tol = 1e-6)
    m, n, p, w = inst.m, inst.n, inst.p, inst.w
    τ = sum(inst.q, dims = 1)[:]

    S = make_sub(inst)                              # built ONCE, reused everywhere
    Y0, dist0 = warmstart_nogood(inst, f, tstart, min(warm_budget, tlim), S)

    mp = Model(CPLEX.Optimizer)
    set_optimizer_attribute(mp, "CPX_PARAM_THREADS", 1)
    set_optimizer_attribute(mp, "CPX_PARAM_EPGAP", 0.0)
    set_silent(mp)
    @variable(mp, Y[1:n], Bin)
    @variable(mp, η >= 0)
    @objective(mp, Min, f * sum(Y) + η)
    @constraint(mp, [k = 1:p], sum(w[j, k] * Y[j] for j in 1:n) >= τ[k])
    set_start_value.(Y, Float64.(Y0)); set_start_value(η, dist0)

    ncuts = Ref(0)
    best_ub = Ref(f * sum(Y0) + dist0)      # manual, always-valid incumbent (warm start)
    best_t  = Ref(time() - tstart)          # time-to-best, comparable with Algorithm 2
    function lazy_cb(cb_data)
        callback_node_status(cb_data, mp) == MOI.CALLBACK_NODE_STATUS_INTEGER || return
        yv = callback_value.(cb_data, Y); ev = callback_value(cb_data, η)
        yb = round.(Int, yv)
        dist, feas, ct, coef = eval_sub!(S, inst, Float64.(yb))
        if !feas
            con = @build_constraint(sum((yb[j] == 1 ? 1 - Y[j] : Y[j]) for j in 1:n) >= 1)
            MOI.submit(mp, MOI.LazyConstraint(cb_data), con); return
        end
        tc = f * sum(yb) + dist              # true cost of this FEASIBLE candidate
        if tc < best_ub[]; best_ub[] = tc; best_t[] = time() - tstart; end
        if ev < dist - tol
            con = @build_constraint(η >= ct + sum(coef[j] * Y[j] for j in 1:n))
            MOI.submit(mp, MOI.LazyConstraint(cb_data), con); ncuts[] += 1
        end
    end
    set_attribute(mp, MOI.LazyConstraintCallback(), lazy_cb)

    set_optimizer_attribute(mp, "CPX_PARAM_TILIM", max(1.0, tlim - (time() - tstart)))
    optimize!(mp)

    proven = termination_status(mp) == MOI.OPTIMAL
    # report the manually tracked incumbent (valid); equals the optimum when proven
    obj = best_ub[]
    return obj, proven, ncuts[], best_t[]
end

# --- Monolithic MILP for cross-checking -------------------------------------
function opt_all(instance::SCPInstance, f::Float64)
    model = Model(CPLEX.Optimizer)
    set_optimizer_attribute(model, "CPX_PARAM_EPGAP", 0.0)
    set_optimizer_attribute(model, "CPX_PARAM_TILIM", 105)
    set_optimizer_attribute(model, "CPX_PARAM_THREADS", 1)
    set_silent(model)
    m, n, p, q, w, d = instance.m, instance.n, instance.p, instance.q, instance.w, instance.d
    @variable(model, X[1:m, 1:n], Bin); @variable(model, Y[1:n], Bin)
    @objective(model, Min, sum(X[i, j] * d[i, j] for i in 1:m, j in 1:n) + f * sum(Y))
    @constraint(model, [i = 1:m], sum(X[i, j] for j in 1:n) == 1)
    @constraint(model, [j = 1:n, k = 1:p], sum(q[i, k] * X[i, j] for i in 1:m) <= w[j, k] * Y[j])
    @constraint(model, [i = 1:m, j = 1:n], X[i, j] - Y[j] <= 0)
    optimize!(model)
    return objective_value(model), termination_status(model) == MOI.OPTIMAL
end

const OPEN = [61,66,67,68,72,73,75,77,79,80,84,86,87,88,89,90,91,93,95,100,
              104,105,109,112,113,114,115,116,117,120]

# --- Calibration harness ------------------------------------------------------
function calibrate(; pool = 45:120, ntrain = 20, seed = 1234,
                   budgets = [0.0, 15.0, 30.0, 45.0, 60.0])
    Random.seed!(seed)
    train = sort(shuffle(collect(pool))[1:ntrain])
    hard = count(i -> i in OPEN, train)
    println("Calibration training set (pool=", first(pool), "..", last(pool),
            ", n=", ntrain, ", seed=", seed, "):")
    println("  ", train)
    println("  of which hard (open under no-good): ", hard, "/", ntrain)
    println("\nwarm_budget   proven   total_obj")
    for wb in budgets
        _, nproven, totobj = run_budget(train, wb)   # run_budget prints per-instance progress
        @printf("SUMMARY  warm_budget=%.0f  proven=%d/%d  total_obj=%d\n\n",
                wb, nproven, ntrain, round(Int, totobj))
        flush(stdout)
    end
end

# Runs one warm_budget over a fixed instance list (serial within the call);
# used as the unit of work for process-parallel calibration.
function run_budget(train::Vector{Int}, wb::Float64)
    nproven = 0; totobj = 0.0
    for i in train
        inst = read_scp_instance("SCP/SCP_inst$i.txt")
        t = time()
        obj, proven, _ = benders_singletree(inst, t, 1e6; warm_budget = wb)
        nproven += proven ? 1 : 0; totobj += obj
        println("  [budget=", Int(wb), "] inst ", i, ": obj=", round(Int, obj),
                " proven=", proven, " time=", round(time() - t, digits = 1))
        flush(stdout)
    end
    return (wb, nproven, totobj)
end

# --- Driver ------------------------------------------------------------------
function main(; crosscheck::Bool = false, instances = OPEN, warm_budget = 30.0)
    nproven = 0
    for i in instances
        inst = read_scp_instance("SCP/SCP_inst$i.txt")
        f = i == 121 ? 1e4 : 1e6
        t = time()
        obj, proven, ncuts, best_t = benders_singletree(inst, t, f; warm_budget = warm_budget)
        total_t = time() - t                     # TOTAL wall-clock, comparable with the MILP
        nproven += proven ? 1 : 0
        if crosscheck
            obj2, _ = opt_all(inst, f)
            println(i, "  benders=", round(Int, obj), "  milp=", round(Int, obj2),
                    "  match=", round(Int, obj) == round(Int, obj2), "  proven=", proven)
        else
            # i  obj  ncuts  best_t  total_t  proven   (same layout as nogood_singletree.jl)
            println(i, " ", round(Int, obj), " ", ncuts, " ", round(best_t, digits = 2), " ",
                    round(total_t, digits = 2), " ", proven)
        end
        flush(stdout)
    end
    println("TOTAL proven optimal: ", nproven, "/", length(collect(instances)))
    flush(stdout)
end

# This file is now functions-only (safe to `include` from a parallel driver).
# Usage:
#   include("benders_singletree.jl")
#   main(crosscheck = true, instances = 1:8)   # validate
#   calibrate()                                # SERIAL calibration (fair timing)
#   main(instances = OPEN)                      # report with the chosen warm_budget
# For PROCESS-PARALLEL calibration (faster, exploratory), run: run_calibration.jl
