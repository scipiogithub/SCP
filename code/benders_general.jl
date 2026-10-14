# benders_general.jl
# -----------------------------------------------------------------------------
# GENERALITY EXPERIMENT (Reviewer #2, point 1).
#
# The referee objects that the proposed algorithms rely on the parameter setting
# f_j = f with f large enough to induce a lexicographic objective, and that they
# therefore do not address the general SCP of Section 2. This holds for the
# combinatorial scheme, which confines the search to the minimum-cardinality
# stratum. It does NOT hold for the branch-and-Benders-cut method: its master is
#       min  sum_j f_j Y_j + eta   s.t. (II), optimality cuts,
# which never restricts cardinality and is indifferent to the cost structure.
# The Benders cuts come from the transportation dual and do not involve f at all.
#
# This script tests that claim on instances with HETEROGENEOUS, NON-DOMINATING
# opening costs, where the optimum genuinely trades schools against distance.
#
# Cost design (instance-adaptive, so the trade-off is real by construction):
#   dist_all = assignment cost with every school open      (lower bound)
#   kappa    = minimum number of schools meeting coverage (II)
#   dist_k   = assignment cost at the best kappa-school configuration
#   Delta    = (dist_k - dist_all) / (n - kappa)      ~ value of one extra school
#   f_j      = U[0.5, 1.5] * Delta                    heterogeneous across schools
# With f_j on the scale of Delta neither term dominates: opening a further school
# costs about what it saves, so the optimal cardinality is not pinned to kappa.
#
# Reported per instance: kappa, the number of schools opened in the certified
# optimum (n_opt), and whether n_opt > kappa — i.e. whether the minimum-cardinality
# restriction would have excluded the true optimum. Every run is cross-checked
# against the monolithic MILP with the same f_j.
#
# Run (serial):  include("benders_general.jl");  main_general()
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

# --- persistent transportation subproblem (identical to benders_singletree.jl)
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

# --- minimum cardinality kappa (unit costs: well scaled, solves instantly) ----
function min_cardinality(inst::SCPInstance)
    n, p, w = inst.n, inst.p, inst.w
    τ = sum(inst.q, dims = 1)[:]
    mc = Model(CPLEX.Optimizer); set_silent(mc)
    set_optimizer_attribute(mc, "CPX_PARAM_THREADS", 1)
    @variable(mc, Y[1:n], Bin)
    @objective(mc, Min, sum(Y))
    @constraint(mc, [k = 1:p], sum(w[j, k] * Y[j] for j in 1:n) >= τ[k])
    optimize!(mc)
    termination_status(mc) == MOI.OPTIMAL || return (0, Int[], false)
    yb = round.(Int, value.(Y))
    return (sum(yb), yb, true)
end

# --- heterogeneous, non-dominating opening costs ------------------------------
# `cost_scale` multiplies Delta, so the same design can be swept across cost
# regimes: scale 1 gives f_j on the order of the average marginal saving (a sharp
# trade-off); larger values move the optimum back towards kappa. Default 1.0
# reproduces the run reported in benders_general_results.txt.
function make_costs(inst::SCPInstance, S::Sub, rng::AbstractRNG; cost_scale = 1.0)
    n = inst.n
    dist_all, ok1, _, _ = eval_sub!(S, inst, ones(n))
    κ, y0, ok2 = min_cardinality(inst)
    (ok1 && ok2) || return (Float64[], 0, false)
    dist_κ, ok3, _, _ = eval_sub!(S, inst, Float64.(y0))
    ok3 || return (Float64[], κ, false)
    Δ = n > κ ? (dist_κ - dist_all) / (n - κ) : dist_all / inst.m
    Δ <= 0 && (Δ = dist_all / inst.m)
    fv = [max(1.0, round(cost_scale * (0.5 + rand(rng)) * Δ)) for _ in 1:n]
    return (fv, κ, true)
end

# --- branch-and-Benders-cut with arbitrary opening costs ----------------------
function benders_general(inst::SCPInstance, fv::Vector{Float64}, tstart::Float64,
                         S::Sub; tlim = 105.0, tol = 1e-6)
    n, p, w = inst.n, inst.p, inst.w
    τ = sum(inst.q, dims = 1)[:]

    # warm start: cheapest cover under f_j, guided by attractiveness
    eval_sub!(S, inst, ones(n))
    a = zeros(n)
    for k in 1:p
        xk = value.(S.xs[k])
        for j in 1:n; a[j] += sum(@view xk[:, j]); end
    end
    ω = maximum(a) .- a
    ws = Model(CPLEX.Optimizer); set_silent(ws)
    set_optimizer_attribute(ws, "CPX_PARAM_THREADS", 1)
    set_optimizer_attribute(ws, "CPX_PARAM_TILIM", 30)
    @variable(ws, Yw[1:n], Bin)
    @objective(ws, Min, sum(fv[j] * Yw[j] for j in 1:n) + sum(ω[j] * Yw[j] for j in 1:n))
    @constraint(ws, [k = 1:p], sum(w[j, k] * Yw[j] for j in 1:n) >= τ[k])
    optimize!(ws)
    Y0 = round.(Int, value.(Yw))
    dist0, _, _, _ = eval_sub!(S, inst, Float64.(Y0))

    mp = Model(CPLEX.Optimizer)
    set_optimizer_attribute(mp, "CPX_PARAM_THREADS", 1)
    set_optimizer_attribute(mp, "CPX_PARAM_EPGAP", 0.0)
    set_silent(mp)
    @variable(mp, Y[1:n], Bin)
    @variable(mp, η >= 0)
    @objective(mp, Min, sum(fv[j] * Y[j] for j in 1:n) + η)     # <-- general costs
    @constraint(mp, [k = 1:p], sum(w[j, k] * Y[j] for j in 1:n) >= τ[k])
    set_start_value.(Y, Float64.(Y0)); set_start_value(η, dist0)

    ncuts = Ref(0)
    best_ub = Ref(sum(fv .* Y0) + dist0)
    best_Y  = Ref(copy(Y0))
    best_t  = Ref(time() - tstart)
    function lazy_cb(cb_data)
        callback_node_status(cb_data, mp) == MOI.CALLBACK_NODE_STATUS_INTEGER || return
        yv = callback_value.(cb_data, Y); ev = callback_value(cb_data, η)
        yb = round.(Int, yv)
        dist, feas, ct, coef = eval_sub!(S, inst, Float64.(yb))
        if !feas
            con = @build_constraint(sum((yb[j] == 1 ? 1 - Y[j] : Y[j]) for j in 1:n) >= 1)
            MOI.submit(mp, MOI.LazyConstraint(cb_data), con); return
        end
        tc = sum(fv .* yb) + dist
        if tc < best_ub[]
            best_ub[] = tc; best_Y[] = copy(yb); best_t[] = time() - tstart
        end
        if ev < dist - tol
            con = @build_constraint(η >= ct + sum(coef[j] * Y[j] for j in 1:n))
            MOI.submit(mp, MOI.LazyConstraint(cb_data), con); ncuts[] += 1
        end
    end
    set_attribute(mp, MOI.LazyConstraintCallback(), lazy_cb)
    set_optimizer_attribute(mp, "CPX_PARAM_TILIM", max(1.0, tlim - (time() - tstart)))
    optimize!(mp)

    proven = termination_status(mp) == MOI.OPTIMAL
    return best_ub[], proven, ncuts[], best_t[], best_Y[]
end

# --- monolithic MILP with the same heterogeneous costs (cross-check) ---------
function opt_all_general(inst::SCPInstance, fv::Vector{Float64}; tlim = 105.0)
    m, n, p, q, w, d = inst.m, inst.n, inst.p, inst.q, inst.w, inst.d
    model = Model(CPLEX.Optimizer)
    set_optimizer_attribute(model, "CPX_PARAM_EPGAP", 0.0)
    set_optimizer_attribute(model, "CPX_PARAM_TILIM", tlim)
    set_optimizer_attribute(model, "CPX_PARAM_THREADS", 1)
    set_silent(model)
    @variable(model, X[1:m, 1:n], Bin); @variable(model, Y[1:n], Bin)
    @objective(model, Min, sum(X[i, j] * d[i, j] for i in 1:m, j in 1:n) +
                           sum(fv[j] * Y[j] for j in 1:n))
    @constraint(model, [i = 1:m], sum(X[i, j] for j in 1:n) == 1)
    @constraint(model, [j = 1:n, k = 1:p], sum(q[i, k] * X[i, j] for i in 1:m) <= w[j, k] * Y[j])
    @constraint(model, [i = 1:m, j = 1:n], X[i, j] - Y[j] <= 0)
    optimize!(model)
    st = termination_status(model)
    if primal_status(model) != MOI.FEASIBLE_POINT
        return (Inf, false)
    end
    return (objective_value(model), st == MOI.OPTIMAL)
end

# --- driver -------------------------------------------------------------------
# Columns: i  kappa  n_opt  obj  ncuts  best_t  total_t  proven  milp_obj  milp_proven  match
function main_general(; seed = 1234, nsample = 20, pool = 1:120, tlim = 105.0,
                      include_real = true, cost_scale = 1.0)
    rng = MersenneTwister(seed)
    sample = sort(shuffle(rng, collect(pool))[1:nsample])
    # Instance 121 is the real municipal network (Fortaleza). Heterogeneous
    # opening costs are most natural precisely there, so it is always included
    # rather than left to the draw.
    include_real && !(121 in sample) && push!(sample, 121)
    println("Heterogeneous-cost experiment | seed=", seed,
            " | cost_scale=", cost_scale, " | instances: ", sample)
    ndone = 0; nproven = 0; nmilp = 0; nmatch = 0
    nabove_cert = 0            # among CERTIFIED optima only
    # Counted over the instances NEITHER method certified.  The size of that set
    # must be counted here, in the same branch that fills nb_better/nm_better --
    # it is NOT ndone - nproven, which is the set Benders failed to certify and
    # may include instances the MILP did certify.
    n_neither = 0
    nb_better = 0; nm_better = 0
    # Benders failed to certify but the MILP did, and Benders' incumbent already
    # equals the MILP optimum: the solution was found, only the proof was missing.
    nb_found_not_proved = 0
    for i in sample
        inst = read_scp_instance("SCP/SCP_inst$i.txt")
        S = make_sub(inst)
        fv, κ, ok = make_costs(inst, S, rng; cost_scale = cost_scale)
        if !ok
            println(i, "  [cost setup failed]"); continue
        end
        t = time()
        obj, proven, ncuts, best_t, bestY = benders_general(inst, fv, t, S; tlim = tlim)
        total_t = time() - t
        nopen = sum(bestY)
        mobj, mprov = opt_all_general(inst, fv; tlim = tlim)
        match = isfinite(obj) && isfinite(mobj) && abs(obj - mobj) <= 1e-6 * max(1.0, abs(mobj))
        ndone   += 1
        nproven += proven ? 1 : 0
        nmilp   += mprov ? 1 : 0
        nmatch  += (proven && mprov && match) ? 1 : 0
        nabove_cert += (proven && nopen > κ) ? 1 : 0
        if !proven && !mprov
            n_neither += 1
            obj < mobj - 1e-9 && (nb_better += 1)
            mobj < obj - 1e-9 && (nm_better += 1)
        end
        nb_found_not_proved += (!proven && mprov && match) ? 1 : 0
        @printf("%3d%s  kappa=%3d  n_opt=%3d  %s  obj=%.1f  cuts=%d  best_t=%.2f  total_t=%.2f  proven=%s  milp=%.1f/%s  match=%s\n",
                i, i == 121 ? "*" : " ", κ, nopen, nopen > κ ? ">kappa" : "=kappa",
                obj, ncuts, best_t, total_t, proven, mobj, mprov, match)
        flush(stdout)
    end
    println()
    println("instances run                       : ", ndone)
    println("certified by branch-and-Benders-cut : ", nproven, "/", ndone)
    println("certified by the monolithic MILP    : ", nmilp, "/", ndone)
    println("Benders optimum = MILP optimum      : ", nmatch, "/", nproven,
            "   (of the instances Benders certified)")
    println()
    # The only rigorous statement about the optimal cardinality is the one taken
    # over CERTIFIED optima: elsewhere n_opt is merely the cardinality of the
    # incumbent, not of a proven optimum. Reporting it over all instances would
    # overstate the finding.
    println("certified optima opening > kappa    : ", nabove_cert, "/", nproven,
            "   <- the minimum-cardinality restriction would exclude the optimum here")
    println()
    println("on the ", n_neither, " instances neither method certified:")
    println("   better incumbent - Benders : ", nb_better)
    println("   better incumbent - MILP    : ", nm_better)
    println()
    println("Benders found the MILP optimum without certifying it : ", nb_found_not_proved,
            "/", ndone - nproven, " of the instances Benders left open")
    println()
    println("n_opt is the number of schools opened; for uncertified instances it")
    println("refers to the incumbent, not to a proven optimum.")
    println("(* marks instance 121, the real municipal network)")
    flush(stdout)
end

# --- data deposit -------------------------------------------------------------
# Writes the opening-cost vectors actually used in Section 6.4 to a text file,
# for deposit alongside the paper (EJOR computational-experiments policy, data
# sharing).  Publishing the seed alone would not be enough: the drawn values
# depend on Julia's MersenneTwister stream and on Delta, which is itself obtained
# by solving the subproblem, so the realised vectors are the reproducible record.
#
# This function MUST consume the random stream in exactly the same order as
# main_general -- same seed, same shuffle, same per-instance call to make_costs --
# otherwise the exported costs would not be the ones behind the reported results.
# That is why it lives in this file next to main_general rather than in a script
# of its own.
#
# Usage:  include("benders_general.jl")
#         export_costs()                       # writes all three cost levels
function export_costs(; seed = 1234, nsample = 20, pool = 1:120,
                        include_real = true, scales = (1.0, 2.0, 5.0),
                        path = "heterogeneous_opening_costs.txt")
    open(path, "w") do io
        println(io, "# Opening-cost vectors for the heterogeneous-cost experiment (Section 6.4).")
        println(io, "# f_j ~ U[0.5, 1.5] * c * Delta, Delta = (dist_kappa - dist_all)/(n - kappa),")
        println(io, "# rounded to the nearest integer with a floor of 1.")
        println(io, "# Generated with seed=", seed, ", nsample=", nsample,
                    ", pool=", pool, ", include_real=", include_real, ".")
        println(io, "# Columns: cost_scale  instance  n  kappa  f_1 f_2 ... f_n")
        for cs in scales
            rng = MersenneTwister(seed)
            sample = sort(shuffle(rng, collect(pool))[1:nsample])
            include_real && !(121 in sample) && push!(sample, 121)
            for i in sample
                inst = read_scp_instance("SCP/SCP_inst$i.txt")
                S = make_sub(inst)
                fv, κ, ok = make_costs(inst, S, rng; cost_scale = cs)
                if !ok
                    println(io, "# ", i, "  [cost setup failed]")
                    continue
                end
                print(io, cs, " ", i, " ", inst.n, " ", κ)
                for v in fv
                    print(io, " ", Int(round(v)))
                end
                println(io)
                flush(io)
            end
        end
    end
    println("wrote ", path)
end
