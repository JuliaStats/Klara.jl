my_tests = ["test/test_hmc.jl",
            "test/test_empmctuner.jl",
            "test/test_syntax.jl",
            "test/parsers/test_diff.jl",
            "test/parsers/unit_tests.jl",
            "test/test_dists.jl",
            "test/test_ARS1.jl",
            "test/test_ARS2.jl"]

println("Running tests:")

for my_test in my_tests
    println("  * $(my_test) *")
    include(my_test)
end
