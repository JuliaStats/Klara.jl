my_tests = ["test_hmc.jl",
            "test_empmctuner.jl",
            "test_syntax.jl",
            "parsers/test_diff.jl",
            "parsers/unit_tests.jl",
            "test_dists.jl",
            "test_ARS1.jl",
            "test_ARS2.jl"]

println("Running tests:")

for my_test in my_tests
    println("  * $(my_test) *")
    include(my_test)
end
