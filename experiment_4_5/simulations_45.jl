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
@everywhere using ProgressMeter


function run_simulations_45()
	simulations_45a(20)
	simulations_45b(20)
	simulations_45c(20)
	return
end

function simulations_45a(iterations)
	fn = "data/Abar/gbar-DSDS01876.edgelist"
	simulations_45_real(fn, iterations)
	return
end

function simulations_45b(iterations)
	ns = [70, 95, 107, 139, 194, 277, 349, 445, 582, 832, 1215, 1875, 3230]
	for n in ns
		println("Running simulations for n=$(n)")
		n_str = lpad(n+1, 5, "0")
		fn = "data/Abar/gbar-DSDS$(n_str).edgelist"
		simulations_45_real(fn, iterations)
	end
	return
end

function simulations_45c(iterations)
	ns = [70, 95, 107, 139, 194, 277, 349, 445, 582, 832, 1215, 1875, 3230]
	for n in ns
		println("Running simulations for n=$(n)")
		n_str = lpad(n+1, 5, "0")
		fn = "data/Abar/gbar-DSDS$(n_str).edgelist"
		simulations_45_synthetic(fn, iterations)
	end
	return
end

@everywhere function simulations_45_real(fn, iterations)
	G = load_graph(fn)
	N = size(G,1)

	p = sum(G) / binomial(N,2) / 2
	@show p

	H = BitArray(rand(N,N) .< p);
	pH = sum(H) / binomial(N,2) / 2
	@show pH

	m = round(Int, N / 10)
	ρes = 0.0:0.025:1.0
	params = collect(Iterators.product(1:iterations, reverse(ρes)))[:];

	println("Running $(length(params)) graph matches...")
	results_raw = @showprogress @distributed (vcat) for (i,ρe) in params
		graphA = Int.(G)
		graphB = interpolate_graphs(graphA, Int.(H), ρe)
		graphA, graphB, matching = GraphMatching.permute_seeded(graphA, graphB, m)

		pA = sum(graphA) / binomial(N,2) / 2
		pB = sum(graphB) / binomial(N,2) / 2

		r = run_matching(graphA, graphB, matching, m)
		r = merge(r, (
			N = N, n = N-m, m = m,
			ρe = ρe,
			pA = pA, pB = pB,
			fn = fn,
			experiment = "51-real",
			datatype = "real",
		))
		r
	end

	folder, file_id = get_output_folder()
	results = DataFrame(results_raw)
	CSV.write(joinpath(folder, "simulatons-$(file_id).csv"), results)

	return results
end

@everywhere function simulations_45_synthetic(fn, iterations)
	G = load_graph(fn)
	N = size(G,1)
	p = sum(G) / binomial(N,2) / 2
	m = round(Int, N / 10)

	ρes = 0.0:0.025:1.0
	params = collect(Iterators.product(1:iterations, reverse(ρes)))[:];

	println("Running $(length(params)) graph matches...")
	results_raw = @showprogress @distributed (vcat) for (i,ρe) in params
		graphA, graphB, matching = GraphMatching.generate_erdosrenyi(N, p, ρe)
		graphA, graphB, matching = GraphMatching.permute_seeded(graphA, graphB, m)

		pA = sum(graphA) / binomial(N,2) / 2
		pB = sum(graphB) / binomial(N,2) / 2

		r = run_matching(graphA, graphB, matching, m)
		r = merge(r, (
			N = N, n = N-m, m = m,
			ρe = ρe,
			pA = pA, pB = pB,
			fn = fn,
			experiment = "51-synthetic",
			datatype = "synthetic",
		))
		r
	end

	folder, file_id = get_output_folder()
	results = DataFrame(results_raw)
	CSV.write(joinpath(folder, "simulatons-$(file_id).csv"), results)

	return results
end

@everywhere function load_graph(fn)
	edges = Tuple{Int,Int}[]
	open(fn) do f
	    graph_data_text = readlines(f)
	    for line in graph_data_text
	        v1, v2 = parse.(Int, split(line, " ")[1:2])
	        push!(edges, (v1,v2))
	    end
	end
	edges = hcat([e[1] for e in edges], [e[2] for e in edges]);

	vtxs = sort(unique(edges))
	vtx_to_idx = Dict(v => i for (i,v) in enumerate(vtxs));

	N = length(vtxs);
	E = size(edges, 1);

	G = BitArray(zeros(Bool, N, N))
	for i in 1:E
	    u1, u2 = edges[i,:]
	    v1, v2 = vtx_to_idx[u1], vtx_to_idx[u2]
	    G[v1,v2] = 1
	end
	G = G .| G'

	return G
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

if abspath(PROGRAM_FILE) == @__FILE__
	run_simulations_45()
end
