using Base.Test
using MCMC

my_tests = ["test/test_syntax.jl",
            "test/test_diff.jl",
            "test/test_dists.jl"]

println("Running tests:")

for my_test in my_tests
    println(" * $(my_test)")
    include(my_test)
end