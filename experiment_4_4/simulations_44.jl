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

@everywhere import GraphMatching

@everywhere using DataFrames, CSV
@everywhere using Random, Distributions
@everywhere using ProgressMeter


function run_simulations_44()
	# simulations_44a(1)
	simulations_44b(1)
	# simulations_44c(1)
	return
end

@everywhere function simulations_44a(iterations)
	n = 1000
	m = 250
	N = n + m

	b1 = round(Int, 0.2 * N)
	b2 = N - b1
	b = vcat(1 * ones(Int, b1), 2 * ones(Int, b2))

	Λ = [
		0.3 0.4
		0.4 0.5
	]

	iter_list = 1:iterations
	ρe_list = 0:0.025:1

	params = collect(Iterators.product(iter_list, ρe_list))
	println("Running $(length(params)) graph matches...")
	results_raw = @showprogress @distributed (vcat) for (it, ρe) in params
		p_matrix = zeros(Float64, N, N)
		for i in 1:N
			for j in i+1:N
				p_matrix[i,j] = Λ[b[i], b[j]]
			end
		end
		p_matrix = p_matrix + p_matrix'

		μ = mean(p_matrix)
		σ = std(p_matrix)
		ρh = (σ^2) / (μ * (1 - μ))
		ρt = 1 - ((1-ρe) * (1-ρh))

		r = simulate_bernoulli(p_matrix, m, ρe, exp_name="44a")
		r = merge((
			N = N, n = n, m = m,
			μ = μ, σ = σ,
			ρe = ρe, ρh = ρh, ρt = ρt,
		), r)
		[r]
	end

	folder, file_id = get_output_folder()
	results = DataFrame(results_raw)
	CSV.write(joinpath(folder, "simulatons-$(file_id).csv"), results)

	return results
end

@everywhere function simulations_44b(iterations)
	n = 1000
	# m = 250
	m = 40
	N = n + m

	b1 = round(Int, 0.2 * N)
	b2 = N - b1
	b = vcat(1 * ones(Int, b1), 2 * ones(Int, b2))

	Λ = [
		0.3 0.4
		0.4 0.0
	]

	iter_list = 1:iterations
	# ρe_list = 0:0.025:1
	ρe_list = 0:0.005:1

	params = collect(Iterators.product(iter_list, ρe_list))
	println("Running $(length(params)) graph matches...")
	results_raw = @showprogress @distributed (vcat) for (it, ρe) in params
		p_matrix = zeros(Float64, N, N)
		for i in 1:N
			for j in i+1:N
				if b[i] == b[j] == 2
					d = rand()
				else
					d = Λ[b[i], b[j]]
				end
				p_matrix[i,j] = d
			end
		end
		p_matrix = p_matrix + p_matrix'

		μ = mean(p_matrix)
		σ = std(p_matrix)
		ρh = (σ^2) / (μ * (1 - μ))
		ρt = 1 - ((1-ρe) * (1-ρh))

		r = simulate_bernoulli(p_matrix, m, ρe, exp_name="44b")
		r = merge((
			N = N, n = n, m = m,
			μ = μ, σ = σ,
			ρe = ρe, ρh = ρh, ρt = ρt,
		), r)
		[r]
	end

	folder, file_id = get_output_folder()
	results = DataFrame(results_raw)
	CSV.write(joinpath(folder, "simulatons-$(file_id).csv"), results)

	return results
end

@everywhere function simulations_44c(iterations)
	n = 1000
	m = 250
	N = n + m

	b1 = round(Int, 0.2 * N)
	b2 = N - b1
	b = vcat(1 * ones(Int, b1), 2 * ones(Int, b2))

	ϵ = 1e-3

	iter_list = 1:iterations
	ρe_list = 0:0.025:1

	dist11_list = [Uniform(0.25, 0.35), Uniform(0.0, 0.6)]
	dist12_list = [Uniform(0.35, 0.45), Uniform(0.0, 0.8)]
	dist22_list = [Uniform(0.45, 0.55), Uniform(0.0, 1.0)]
	dist_list = collect(Iterators.product(dist11_list, dist12_list, dist22_list))

	params = collect(Iterators.product(iter_list, ρe_list))
	println("Running $(length(params) * 8) graph matches...")
	results_raw = @showprogress @distributed (vcat) for (it, ρe) in params
		_results = []
		p_matrix = zeros(Float64, N, N)
		p_means = zeros(Float64, N, N)

		for (dist_idx, (dist11, dist12, dist22)) in enumerate(dist_list)
			Λ = [
				dist11 dist12;
				dist12 dist22;
			]

			fill!(p_matrix, 0.0)
			fill!(p_means, 0.0)
			for i in 1:N
				for j in i+1:N
					λ = Λ[b[i], b[j]]
					p_matrix[i,j] = min(max(rand(λ), ϵ), 1.0-ϵ)
					p_means[i,j] = mean(λ)
				end
			end
			p_matrix = p_matrix + p_matrix'
			p_means = p_means + p_means'

			μ = mean(p_matrix)
			σ = std(p_matrix)
			ρh = (σ^2) / (μ * (1 - μ))
			ρt = 1 - ((1-ρe) * (1-ρh))

			σ2_mod = sum((p_matrix - p_means).^2) / (N^2 - 1)
			ρh_mod = σ2_mod / (μ * (1 - μ))
			ρt_mod = 1 - ((1-ρe) * (1-ρh_mod))

			r = simulate_bernoulli(p_matrix, m, ρe, exp_name="44c")
			r = merge((
				N = N, n = n, m = m, μ = μ, dist_idx = dist_idx,
				ρe = ρe, ρh = ρh, ρt = ρt, σ = σ,
				ρh_mod = ρh_mod, ρt_mod = ρt_mod, σ2_mod = σ2_mod,
			), r)
			push!(_results, r)
		end
		_results
	end

	folder, file_id = get_output_folder()
	results = DataFrame(results_raw)
	CSV.write(joinpath(folder, "simulatons-$(file_id).csv"), results)

	return results
end

@everywhere function simulate_bernoulli(p_matrix, m, ρe; maxiter::Int=20, exp_name=missing)
	graphA, graphB, matching = GraphMatching.generate_bernoulli(p_matrix, ρe)
	graphA, graphB, matching = GraphMatching.permute_seeded(graphA, graphB, m)

	sgm_time = @elapsed P, est_matching, it = GraphMatching.sgm(graphA, graphB, m, maxiter=maxiter, returniter=true)

	match_r    = GraphMatching.match_ratio(matching, est_matching, m)
	algn_str_1 = GraphMatching.alignment_strength(graphA, graphB, P, m)
	algn_str_2 = GraphMatching.alignment_strength(graphA, graphB, P, 0)

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

@everywhere function get_output_folder()
	d = Dates.format(Dates.now(), "yyyy-mm-dd")
	t = Dates.format(Dates.now(), "yyyy-mm-dd-HH-MM-SS")
	basepath = normpath(joinpath(pathof(@__FILE__), "../", "results"))
	outpath = joinpath(basepath, d)
	mkpath(outpath)
	return outpath, t
end

if abspath(PROGRAM_FILE) == @__FILE__
	run_simulations_44()
end
