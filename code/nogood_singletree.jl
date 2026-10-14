# nogood_singletree.jl
# -----------------------------------------------------------------------------
# VARIANT A (ABLATION): SINGLE-TREE branch-and-check with COMBINATORIAL (no-good)
# cuts only.
#
# DESIGN. The subproblem machinery (Sub, make_sub, eval_sub!) is IDENTICAL to
# benders_singletree.jl, and the master/coverage/pricing are the same. The ONLY
# difference from EXACT_B is the cut submitted in the callback:
#     EXACT_B    ->  dual Benders optimality cut   eta >= ct + sum_j coef_j Y_j
#     VARIANT A  ->  combinatorial no-good cut     Hamming(Y, yhat) >= 1
# NO dual information is used anywhere below (eval_dist! never calls dual()).
#
# Therefore:  VARIANT A vs EXACT_B  = pure cut-type effect  (same tree, same code)
#             VARIANT A vs EXACT_NG = pure tree-structure effect
#
#   Validate:  main(crosscheck=true, instances=1:10)
#   Report:    main()                       # all 121, prints no_good.txt format
# -----------------------------------------------------------------------------

using JuMP
using CPLEX
using Printf

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
struct Sub
    models::Vector{Model}
    xs::Vector{Matrix{VariableRef}}
    assigns::Vector{Vector{ConstraintRef}}
    caps::Vector{Vector{ConstraintRef}}
    Mk::Vector{Vector{Int}}
end

function make_sub(inst::SCPInstance)::Sub
    m, n, p, d, q, w = inst.m, inst.n, inst.p, inst.d, inst.q, inst.w
    models = Model[]; xs = Matrix{VariableRef}[]
    assigns = Vector{ConstraintRef}[]; caps = Vector{ConstraintRef}[]; Mk = Vector{Int}[]
    for k in 1:p
        idx = [i for i in 1:m if q[i, k]]
        push!(Mk, idx); mk = length(idx)
        sp = Model(CPLEX.Optimizer)
        set_optimizer_attribute(sp, "CPX_PARAM_THREADS", 1)
        set_optimizer_attribute(sp, "CPX_PARAM_LPMETHOD", 2)
        set_optimizer_attribute(sp, "CPX_PARAM_TILIM", 60)
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

# Exact evaluation ONLY -- deliberately returns no dual information, so this
# variant cannot accidentally benefit from the Benders bound.
function eval_dist!(S::Sub, inst::SCPInstance, yhat::Vector{Float64})
    n, p, w = inst.n, inst.p, inst.w
    dist = 0.0
    for k in 1:p
        sp = S.models[k]; cap = S.caps[k]
        for j in 1:n
            set_normalized_rhs(cap[j], w[j, k] * yhat[j])
        end
        optimize!(sp)
        termination_status(sp) == MOI.OPTIMAL || return (Inf, false)
        dist += objective_value(sp)
    end
    return (dist, true)
end

# --- Pricing phase: attractiveness weights ------
function attractiveness(S::Sub, inst::SCPInstance)
    n, p = inst.n, inst.p
    eval_dist!(S, inst, ones(n))            # all schools open
    a = zeros(n)
    for k in 1:p
        xk = value.(S.xs[k])
        for j in 1:n
            a[j] += sum(@view xk[:, j])
        end
    end
    return maximum(a) .- a                  # popular school -> small penalty
end

# --- Single-tree branch-and-check (no-good cuts only) -------------------------
# `use_weights = false` reproduces Version 2 of Section 6.2 (uniform weights,
# omega_j = 1); the default `true` is Version 1 and is the configuration behind
# every other result in the paper.
function nogood_singletree(inst::SCPInstance, tstart::Float64, f::Float64;
                           tlim = 105.0, use_weights::Bool = true)
    n, p, w = inst.n, inst.p, inst.w
    τ = sum(inst.q, dims = 1)[:]

    S = make_sub(inst)                      # built ONCE, reused everywhere
    ω = use_weights ? attractiveness(S, inst) : ones(Float64, n)
    t_price = time() - tstart               # cost of make_sub + pricing

    # --- Certify the minimum cardinality κ -----------------------------
    # Do NOT use the lexicographic surrogate f*sum(Y) + ω'Y here. With f = 1e6
    # every objective coefficient is ~1e6 while the discriminating ω term is O(1),
    # so a zero-gap proof forces CPLEX to separate solutions that differ by ~1e-8
    # in relative terms -- numerically hard, and it simply times out. κ only needs
    # minimum cardinality, so ask exactly that: unit costs, well scaled, fast.
    card = Model(CPLEX.Optimizer)
    set_optimizer_attribute(card, "CPX_PARAM_THREADS", 1)
    set_optimizer_attribute(card, "CPX_PARAM_TILIM", max(1.0, tlim - (time() - tstart)))
    set_silent(card)
    @variable(card, Yc[1:n], Bin)
    @objective(card, Min, sum(Yc))
    @constraint(card, [k = 1:p], sum(w[j, k] * Yc[j] for j in 1:n) >= τ[k])
    optimize!(card)
    stc = termination_status(card)

    if primal_status(card) != MOI.FEASIBLE_POINT
        return (Inf, false, 0, 0.0,
                "no_cardinality_solution($(stc), price=$(round(t_price, digits=1))s)")
    end

    y0 = round.(Int, value.(Yc)); κ = sum(y0)
    d0, feas0 = eval_dist!(S, inst, Float64.(y0))
    best_ub = Ref(feas0 ? f * κ + d0 : Inf)     # manual, always-valid incumbent
    best_t  = Ref(time() - tstart)
    ncand   = Ref(1)

    # CORRECTNESS GUARD: if κ was not PROVEN minimal, restricting to sum(Y)==κ
    # would search the wrong stratum and "INFEASIBLE => proven" would be a false
    # optimality claim.
    if stc != MOI.OPTIMAL
        return (best_ub[], false, ncand[], best_t[],
                "kappa_not_proven($(stc), price=$(round(t_price, digits=1))s)")
    end

    # --- Enumerate the minimum-cardinality stratum ----------------------
    mp = Model(CPLEX.Optimizer)
    set_optimizer_attribute(mp, "CPX_PARAM_THREADS", 1)
    # Lazy constraints invalidate DUAL presolve reductions: keep primal-only, else
    # CPLEX may discard integer solutions we still have to enumerate.
    set_optimizer_attribute(mp, "CPX_PARAM_REDUCE", 1)
    set_silent(mp)
    @variable(mp, Y[1:n], Bin)
    # sum(Y) == κ makes f*sum(Y) the constant f*κ, so the master objective reduces
    # to the attractiveness guide alone -- well scaled, and used only for ordering.
    # No gap parameter is needed: every candidate is rejected by a no-good cut, so
    # an incumbent never exists and the tree can only end INFEASIBLE or TIME_LIMIT.
    @objective(mp, Min, sum(ω[j] * Y[j] for j in 1:n))
    @constraint(mp, [k = 1:p], sum(w[j, k] * Y[j] for j in 1:n) >= τ[k])
    @constraint(mp, sum(Y) == κ)

    function lazy_cb(cb_data)
        callback_node_status(cb_data, mp) == MOI.CALLBACK_NODE_STATUS_INTEGER || return
        yv = callback_value.(cb_data, Y)
        yb = round.(Int, yv)
        dist, feas = eval_dist!(S, inst, Float64.(yb))
        ncand[] += 1
        if feas
            tc = f * sum(yb) + dist          # true cost of this feasible candidate
            if tc < best_ub[]
                best_ub[] = tc; best_t[] = time() - tstart
            end
        end
        # The no-good cut is the ONLY mechanism: every candidate is excluded, so the
        # tree is forced to enumerate the whole minimum-cardinality stratum.
        con = @build_constraint(sum((yb[j] == 1 ? 1 - Y[j] : Y[j]) for j in 1:n) >= 1)
        MOI.submit(mp, MOI.LazyConstraint(cb_data), con)
    end
    set_attribute(mp, MOI.LazyConstraintCallback(), lazy_cb)

    set_optimizer_attribute(mp, "CPX_PARAM_TILIM", max(1.0, tlim - (time() - tstart)))
    optimize!(mp)

    st = termination_status(mp)
    # Since every candidate is rejected, the tree can only close by running out of
    # integer solutions -> the stratum has been fully enumerated -> proven optimal.
    proven = (st == MOI.INFEASIBLE || st == MOI.INFEASIBLE_OR_UNBOUNDED)
    return best_ub[], proven, ncand[], best_t[], string(st)
end

# --- Monolithic MILP for cross-checking ---------------------------------------
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

# --- Driver -------------------------------------------------------------------
function main(; crosscheck::Bool = false, instances = 1:121, use_weights::Bool = true)
    nproven = 0
    for i in instances
        inst = read_scp_instance("SCP/SCP_inst$i.txt")
        f = i == 121 ? 1e4 : 1e6
        t = time()
        obj, proven, ncand, best_t, status = nogood_singletree(inst, t, f; use_weights = use_weights)
        total_t = time() - t                       
        nproven += proven ? 1 : 0
        objstr = isfinite(obj) ? string(round(Int, obj)) : "Inf"   
        note = proven ? "" : "   [" * status * "]"
        if crosscheck
            obj2, milp_ok = opt_all(inst, f)
            matched = isfinite(obj) && round(Int, obj) == round(Int, obj2)
            println(i, "  nogood_st=", objstr, " proven=", proven,
                    "   milp=", round(Int, obj2), " proven=", milp_ok,
                    "   match=", matched, note)
        else
            # i  obj  ncand  best_t  total_t  proven
            println(i, " ", objstr, " ", ncand, " ", round(best_t, digits = 2), " ",
                    round(total_t, digits = 2), " ", proven, note)
        end
        flush(stdout)
    end
    println("TOTAL proven optimal: ", nproven, "/", length(collect(instances)))
    flush(stdout)
end

# Functions-only (safe to `include`). Usage:
#   include("nogood_singletree.jl")
#   main(crosscheck = true, instances = 1:10)   # validate first
#   main()                                      # full run, serial

# --- monolithic MILP baseline: reporting run -----------------------------------
# `opt_all` above is the Julia reimplementation of the monolithic formulation of
# Eqs. (1)-(6) that produced the MILP column of Table 3 and Table S2. It returns
# the objective value and the optimality flag but not the solution time, which
# those tables also report, so the timing was taken around the call. This wrapper
# makes that run reproducible from the published code.
#
# Output format matches the other result files:  i  obj  total_t  proven
# The large constant follows the same convention as `main`: f = 1e6 for the
# benchmark instances and f = 1e4 for instance 121, the real municipal network.
# Note that total_t is wall-clock around the whole call, so it includes model
# construction and can slightly exceed the 105 s solver limit.
#
# Usage:  include("nogood_singletree.jl");  main_milp()
function main_milp(; instances = 1:121)
    nproven = 0
    for i in instances
        inst = read_scp_instance("SCP/SCP_inst$i.txt")
        f = i == 121 ? 1e4 : 1e6
        t = time()
        obj, proven = opt_all(inst, f)
        total_t = time() - t
        nproven += proven ? 1 : 0
        println(i, " ", round(Int, obj), " ", round(total_t, digits = 2), " ", proven)
        flush(stdout)
    end
    println("TOTAL proven optimal: ", nproven, "/", length(collect(instances)))
    flush(stdout)
end
