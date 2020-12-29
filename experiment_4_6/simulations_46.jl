using Distributed
using Glob

if workers() != [1]
	rmprocs(workers()...)
end

procs = 2:22
workers_per_proc = 2

procs = [("fparker9@ugrad$(x).cs.jhu.edu", workers_per_proc) for x in procs]
addprocs(procs, max_parallel=100, tunnel=true, topology=:master_worker, enable_threaded_blas=true)
println("Running on $(nworkers()) workers.")

@everywhere import GraphMatching

@everywhere using DataFrames, CSV, Dates
@everywhere using Random, Distributions
@everywhere using LinearAlgebra
@everywhere using MAT
@everywhere using ProgressMeter


function run_simulations_46()
	simulations_46a(100)
	simulations_46b(100)
	simulations_46c(10)
	return
end

function simulations_46a(iterations; real=true)
	graphA_name = "A_elegans_chem"
	graphB_name = "A_elegans_gap"
	ms = [0, 1, 5, 10, 20, 50, 75, 100, 150, 200]
	sim_fn = real ? simulations_46_real : simulations_46_synthetic
	sim_fn(graphA_name, graphB_name, ms, iterations)
	return
end

function simulations_46b(iterations; real=true)
	graphA_name = "A_enron_week130"
	graphB_name = "A_enron_week131"
	graphC_name = "A_enron_week132"
	ms = [0, 1, 5, 10, 20, 50, 60, 90, 100, 140]
	sim_fn = real ? simulations_46_real : simulations_46_synthetic
	sim_fn(graphA_name, graphB_name, ms, iterations)
	sim_fn(graphA_name, graphC_name, ms, iterations)
	sim_fn(graphB_name, graphC_name, ms, iterations)
	return
end

function simulations_46c(iterations; real=true)
	graphA_name = "A_wiki_english"
	graphB_name = "A_wiki_french"
	ms = [0, 5, 50, 150, 250, 382, 500]
	sim_fn = real ? simulations_46_real : simulations_46_synthetic
	sim_fn(graphA_name, graphB_name, ms, iterations)
	return
end

@everywhere function simulations_46_real(graphA_name, graphB_name, ms, iterations)
	A = load_graph(graphA_name)
	B = load_graph(graphB_name)

	N = size(A,1)

	params = collect(Iterators.product(1:iterations, ms))[:]

	println("Running $(length(params)) graph matches...")
	results_raw = @showprogress @distributed (vcat) for (i,m) in params
		graphA, graphB, matching = GraphMatching.permute_seeded(A, B, m)

		pA = sum(graphA) / binomial(N,2) / 2
		pB = sum(graphB) / binomial(N,2) / 2

		r = run_matching(graphA, graphB, matching, m)
		r = merge(r, (
			N = N, n = N-m, m = m,
			pA = pA, pB = pB,
			experiment = "46-real",
			datatype = "real",
			graphA_name = graphA_name,
			graphB_name = graphB_name,
		))
		r
	end

	folder, file_id = get_output_folder()
	results = DataFrame(results_raw)
	CSV.write(joinpath(folder, "simulatons-$(file_id).csv"), results)

	return results
end

@everywhere function simulations_46_synthetic(graphA_name, graphB_name, ms, iterations)
	A = load_graph(graphA_name)
	B = load_graph(graphB_name)

	N = size(A,1)

	pA = sum(A) / binomial(N,2) / 2
	pB = sum(B) / binomial(N,2) / 2

	params = collect(Iterators.product(1:iterations, ms))[:];

	println("Running $(length(params)) graph matches...")
	results_raw = @showprogress @distributed (vcat) for (i,m) in params
		graphA = generate_erdosrenyi(N, pA)
		graphB = generate_erdosrenyi(N, pB)
		graphA, graphB, matching = GraphMatching.permute_seeded(graphA, graphB, m)

		pa = sum(graphA) / binomial(N,2) / 2
		pb = sum(graphB) / binomial(N,2) / 2

		r = run_matching(graphA, graphB, matching, m)
		r = merge(r, (
			N = N, n = N-m, m = m,
			pA = pa, pB = pb,
			experiment = "46-synthetic",
			datatype = "synthetic",
			graphA_name = graphA_name,
			graphB_name = graphB_name,
		))
		r
	end

	folder, file_id = get_output_folder()
	results = DataFrame(results_raw)
	CSV.write(joinpath(folder, "simulatons-$(file_id).csv"), results)

	return results
end

@everywhere function load_graph(graph_name)
	file = matopen("data/the_adj_matrices.mat")
	adj = read(file, graph_name)
	adj = Int.(Matrix(adj))
	adj = adj .| adj'
	return adj
end

@everywhere function run_matching(graphA, graphB, matching, m; maxiter::Int=20)
    sgm_time = @elapsed P, est_matching, it = GraphMatching.sgm(graphA, graphB, m, maxiter=maxiter, returniter=true)

    match_r    = GraphMatching.match_ratio(matching, est_matching, m)
    algn_str_1 = GraphMatching.alignment_strength(graphA, graphB, P, m)
    algn_str_2 = GraphMatching.alignment_strength(graphA, graphB, P, 0)

    results = (
        match_ratio = match_r,
        alignment_strength_1 = algn_str_1,
        alignment_strength_2 = algn_str_2,
        iter = it, maxiter = maxiter, sgm_time = sgm_time,
    )
    return results
end

@everywhere function interpolate_graphs(graphA, graphB, ρ)
    p_matrix = (ρ .* graphA) + ((1 - ρ) .* graphB)
    N = size(graphA, 1)
    graphC = zeros(Int, N, N)
    for i in 1:N
        for j in i+1:N
            graphC[i,j] = rand(Bernoulli(p_matrix[i,j]))
        end
    end
    graphC = graphC + graphC'
    matching = hcat(1:N, 1:N)
    return graphC
end

@everywhere function get_output_folder()
	d = Dates.format(Dates.now(), "yyyy-mm-dd")
	t = Dates.format(Dates.now(), "yyyy-mm-dd-HH-MM-SS")
	basepath = normpath(joinpath(dirname(@__FILE__), "results"))
	outpath = joinpath(basepath, d)
	mkpath(outpath)
	return outpath, t
end

@everywhere function generate_erdosrenyi(N, p)
	G = Int.(rand(Float64, N, N) .<= p)
	G = G - tril(G)
	G = G .| G'
	return G
end

if abspath(PROGRAM_FILE) == @__FILE__
	run_simulations_46()
end
