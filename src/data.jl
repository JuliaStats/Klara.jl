dataset(example::AbstractString, datafile::AbstractString, suffix::AbstractString="csv", header::Bool=true) =  
  readcsv(joinpath(dirname(@__FILE__), "..", "data", example, join([datafile, suffix], '.')), header=header)

function datasets(example::AbstractString)
  fnames = readcsv(joinpath(dirname(@__FILE__), "..", "doc", "datasets.csv"), header=false)
  fnames[fnames[:, 2] .== example, 4]
end
  
function datasets(example::AbstractString, header::Vector{Bool})
  fnames = datasets(example)
  Any[readcsv(joinpath(dirname(@__FILE__), "..", "data", example, fnames[i]), header=header[i]) for i in 1:length(fnames)]
end

function datasets(example::AbstractString, header::Bool)
  fnames = datasets(example)
  Any[readcsv(joinpath(dirname(@__FILE__), "..", "data", example, f), header=header) for f in fnames]
end

datasets(header::Bool=true) = readcsv(joinpath(dirname(@__FILE__), "..", "doc", "datasets.csv"), header=header)

examples(header::Bool=true) = readcsv(joinpath(dirname(@__FILE__), "..", "doc", "examples", "examples.csv"), header=header)
