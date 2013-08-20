#############################################################################
#
#    Will run every benchmarking unit in directory 'benchunits'
#
#    Each julia file in 'benchunits' should return a benchmark DataFrame
#     in a variable called 'res'
#
#    Results are appended to 'benchlog.csv'
#
#############################################################################

using DataFrames
using Benchmark
using MCMC

# cd to MCMC package dir to get correct git commit id
mcmcdir = joinpath(dirname(Base.find_in_path("MCMC")), "..")
cd(mcmcdir) do 
	# if readall(`git status -s`) != ""
	# 	println("There are untracked/uncommited changes to the package :")
	# 	run(`git status`)
	# 	println("\nProceed anyway with benchmarking ? (yes/NO)")
	# 	c = readline(STDIN)
	# 	match(r"^y|Y", c) || error("Benchmarking aborted")
	# end

	unitdir = joinpath(mcmcdir, "benchmarks", "benchunits")
	benchlog = joinpath(mcmcdir, "benchmarks", "benchlog.csv")

	fb = open(benchlog, "a+")
	for u in split(readall((`ls $unitdir`)))  #  u = split(readall((`ls $unitdir`)))[1]
		println("benchmarking '$u'")
		include(joinpath(unitdir, u))

		print_table(fb, res, ',', '"', false)
	end
	close(fb)
end



