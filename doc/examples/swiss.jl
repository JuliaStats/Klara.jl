using Lora

DATADIR = joinpath(dirname(@__FILE__), "data")

data, header = readcsv(joinpath(DATADIR, "swiss.csv"), header=true)
