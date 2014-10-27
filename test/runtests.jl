tests =
  ["test_empmctuner",
  "test_ARS1",
  "test_ARS2",
  "test_slice_sampler"]

println("Running tests:")

for t in tests
  tfile = t*".jl"
  println("  * $(tfile) *")
  include(tfile)
end
