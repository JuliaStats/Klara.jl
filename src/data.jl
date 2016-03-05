dataset(name::AbstractString, suffix::AbstractString="csv", header::Bool=true) =
  readcsv(joinpath(dirname(@__FILE__), "..", "data", join([name, suffix], '.')), header=header)

datasets(header::Bool=true) = readcsv(joinpath(dirname(@__FILE__), "..", "doc", "datasets.csv"), header=header)
