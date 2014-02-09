using Base.Test
using MCMC

my_tests = ["test/test_hmc.jl",
  "test/test_empmctuner.jl"]

println("Running tests:")

for my_test in my_tests
    println("  * $(my_test) *")
    include(my_test)
end
