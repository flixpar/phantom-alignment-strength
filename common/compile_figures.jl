using Glob

print("Figures base path: ")
expfolder = readline()

infolder  = joinpath(expfolder, "tex/")
outfolder = joinpath(expfolder, "tex_cvt/")
pdffolder = joinpath(expfolder, "pdf/")

function main()

	if !isdir(outfolder)
		mkpath(outfolder)
	end
	if !isdir(pdffolder)
		mkpath(pdffolder)
	end

	for fn in glob("*.tex", infolder)
		figname = basename(fn)[1:end-4]
		println("Compiling $(figname)")
		compile_figure(figname)
	end
	for fn in glob("**/*.tex", infolder)
		figname = basename(fn)[1:end-4]
		figfolder = splitpath(fn)[end-1]
		println("Compiling $(figname)")
		compile_figure(figname, figfolder)
	end

end

function compile_figure(figname, folder="")

	infn = joinpath(infolder, folder, figname * ".tex")
	open(infn, "r") do f
		global tex = readlines(f)
	end

	textfilter = r"\\text\{(.*?)\}"
	textreplace = s"\g<1>"

	textstrings = [
		"Total Correlation",
		"Edge Correlation",
		"Alignment Strength",
		"Match Ratio",
		"Real Data",
		"Real",
		"Erdos-Renyi",
	]

	if figname == "exp42_fig1_v3"
		for l in ['a' + i for i in 0:7]
			push!(textstrings, "($(l))")
		end
	end

	symbols = [
		"ρ" => "\\rho",
		"≥" => "\\geq ",
		"\\&lt;" => "<",
	]

	for (i,line) in enumerate(tex)
		newline = replace(line, textfilter => textreplace)
		for rep in symbols
			newline = replace(newline, rep)
		end
		for s in textstrings
			newline = replace(newline, s => "\\text{$(s) }")
		end
		if newline != line
			tex[i] = newline
			println(line)
			println(newline)
			println()
		end
	end

	if !isdir(joinpath(outfolder, folder)) mkpath(joinpath(outfolder, folder)) end
	if !isdir(joinpath(pdffolder, folder)) mkpath(joinpath(pdffolder, folder)) end

	outfn = joinpath(outfolder, folder, figname * ".tex")
	open(outfn, "w") do f
		for line in tex
			write(f, line)
			write(f, "\n")
		end
	end

	run(`xelatex $(outfn)`)
	run(`rm $(figname).aux`)
	run(`rm $(figname).log`)
	run(`mv $(figname).pdf $(joinpath(pdffolder, folder, figname)).pdf`)

	return
end

if abspath(PROGRAM_FILE) == @__FILE__
	main()
end
