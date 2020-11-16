using Glob

infolder  = "figures/tex/"
outfolder = "figures/tex_cvt/"
pdffolder = "figures/pdf/"

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

end

function compile_figure(figname)

	infn = joinpath(infolder, figname * ".tex")
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

	outfn = joinpath(outfolder, figname * ".tex")
	open(outfn, "w") do f
		for line in tex
			write(f, line)
			write(f, "\n")
		end
	end

	run(`xelatex $(outfn)`)
	run(`rm $(figname).aux`)
	run(`rm $(figname).log`)
	run(`mv $(figname).pdf $(joinpath(pdffolder, figname)).pdf`)

	return
end

if abspath(PROGRAM_FILE) == @__FILE__
	main()
end
