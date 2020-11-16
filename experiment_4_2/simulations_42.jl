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


function run_simulations_42(iterations::Int=1)
	simulations_42a(iterations)
	return
end

@everywhere function simulations_42a(iterations)
	beta_params = Dict(
		:A => (α = 1.0, β = 1.0),
		:B => (α = 0.5, β = 0.5),
		:C => (α = 2.0, β = 2.0),
		:D => (α = 5.0, β = 1.0),
		:E => (α = 2.0, β = 5.0),
	)
	n = 1000

	iter_list = 1:iterations
	ρe_list = 0:0.025:1
	m_list = [0, 10, 20, 50, 250, 500, 1000]
	μ′_list = 0.1:0.1:0.9
	dist_list = [:A, :B, :C, :D, :E]

	params = collect(Iterators.product(iter_list, ρe_list, m_list, μ′_list, dist_list))
	println("Running $(length(params) * 11) graph matches...")
	results_raw = @showprogress @distributed (vcat) for (it, ρe, m, μ′, beta_params_key) in params
		_results = []
		N = n + m

		α = beta_params[beta_params_key].α
		β = beta_params[beta_params_key].β
		dist = Beta(α, β)

		p_matrix = zeros(Float64, N, N)

		δ_max = min(((α + β) / α) * μ′, ((α + β) / β) * (1 - μ′))
		δ_step = δ_max / 10
		for δ in 0:δ_step:δ_max

			dist_tfm = x -> (δ*x) + μ′ - (δ * (α / (α+β)))

			fill!(p_matrix, 0.0)
			for i in 1:N
				for j in i+1:N
					p_matrix[i,j] = dist_tfm(rand(dist))
				end
			end
			p_matrix = p_matrix + p_matrix'

			μ = mean(p_matrix)
			σ = std(p_matrix)
			ρh = (σ^2) / (μ * (1 - μ))
			ρt = 1 - ((1-ρe) * (1-ρh))

			r = simulate_bernoulli(p_matrix, m, ρe, exp_name="42a")
			r = merge((
				N = N, n = n, m = m,
				μ′ = μ′, α = α, β = β, δ = δ,
				beta_params_key = beta_params_key,
				ρe = ρe, ρh = ρh, ρt = ρt,
				μ = μ, σ = σ,
			), r)
			push!(_results, r)
		end
		_results
	end

	folder, file_id = FuzzySGM.get_output_folder()
	results = DataFrame(results_raw)
	CSV.write(joinpath(folder, "simulatons-$(file_id).csv"), results)

	return results
end

@everywhere function simulate_bernoulli(p_matrix, m, ρe; maxiter::Int=20, exp_name=missing)
	graphA, graphB, matching = FuzzySGM.generate_bernoulli(p_matrix, ρe)
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
		distribution = "bernoulli",
		experiment = exp_name,
	)
	return results
end

if abspath(PROGRAM_FILE) == @__FILE__
	run_simulations_42()
end

