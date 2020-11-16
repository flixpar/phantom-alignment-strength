using Distributed
using Glob

if workers() != [1]
	rmprocs(workers()...)
end

procs = 3:22
workers_per_proc = 2

procs = [("fparker9@ugrad$(x).cs.jhu.edu", workers_per_proc) for x in procs]
addprocs(procs, max_parallel=100, tunnel=true, topology=:master_worker, enable_threaded_blas=true)
println("Running on $(nworkers()) workers.")

@everywhere push!(LOAD_PATH, "/home/fparker9/FuzzySGM")
@everywhere import FuzzySGM

@everywhere using DataFrames, CSV
@everywhere using Random, Distributions
@everywhere using ProgressMeter


function run_simulations_43(iterations::Int=100)
	simulations_43a(iterations)
	return
end

@everywhere function simulations_43a(iterations)

	iter_list = 1:iterations
	n_range = 500:4000
	p_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
	m = 0
	ρe = 0.0

	println("Running $(iterations * length(p_list)) graph matches...")
	results_raw = @showprogress @distributed (vcat) for it in 1:iterations
		_results = []
		n = rand(n_range)
		N = n + m
		for p in p_list
			r = simulate_erdosrenyi(n, m, p, ρe, exp_name="43a")
			r = merge((
				N = N, n = n, m = m, p = p,
				ρe = ρe, ρh = 0, ρt = ρe,
			), r)
			push!(_results, r)
		end
		_results
	end

	folder, file_id = FuzzySGM.get_output_folder()
	results = DataFrame(results_raw)
	CSV.write(joinpath(folder, "simulatons-$(file_id).csv"), results)

	return results
end;

@everywhere function simulate_erdosrenyi(n, m, p, ρe; maxiter::Int=20, exp_name=missing)
	graphA, graphB, matching = FuzzySGM.generate_erdosrenyi(n+m, p, ρe)
	graphA, graphB, matching = FuzzySGM.seed_random(graphA, graphB, m)

	sgm_time = @elapsed P, est_matching, it = FuzzySGM.sgm(graphA, graphB, m, maxiter=maxiter, returniter=true)

	match_r    = FuzzySGM.match_ratio(matching, est_matching, m)
	algn_str_1 = FuzzySGM.alignment_strength(graphA, graphB, P, m)
	algn_str_2 = FuzzySGM.alignment_strength(graphA, graphB, P, 0)

	results = (
		match_ratio = match_r,
		alignment_strength_1 = algn_str_1,
		alignment_strength_2 = algn_str_2,
		iter = it, maxiter = maxiter, sgm_time = sgm_time,
		distribution = "erdosrenyi",
		experiment = exp_name,
	)
	return results
end

if abspath(PROGRAM_FILE) == @__FILE__
	run_simulations_43()
end

