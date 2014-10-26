my_tests = ["test_empmctuner.jl",
            "test_ARS1.jl",
            "test_ARS2.jl"]

println("Running tests:")

for my_test in my_tests
    println("  * $(my_test) *")
    include(my_test)
end
