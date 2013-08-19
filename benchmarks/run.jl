#############################################################################
#
#    Will run every Benchmarking unit in directory 'benchunits'
#
#    Each julia file in 'benchunits' should return a benchmark DataFrame
#     in a variable called 'res'
#
#    Result will be appended to 'benchlog.csv'
#
#############################################################################

using Benchmark
using MCMC
using DataFrames

# LIBVERSION = try ; readchomp(`git rev-parse --verify HEAD`)[1:6]; catch e; "-none-";end
mcmcdir = joinpath(dirname(Base.find_in_path("MCMC")), "..")

isdir(".git")
ls()
# cd to MCMC package dir to get correct commit id
cd(mcmcdir) do 
	unitdir = joinpath(mcmcdir, "benchmarks", "benchunits")
	benchlog = joinpath(mcmcdir, "benchmarks", "benchlog.csv")

	fb = open(benchlog, "a+")
	for u in split(readall((`ls $unitdir`)))  #  u = split(readall((`ls $unitdir`)))[1]
		fu = joinpath(benchdir, "benchunits", u)
		println("benchmarking '$u'")
		include(fu)

		print_table(fb, res, ',', '"', false)
	end
end
close(fb)
