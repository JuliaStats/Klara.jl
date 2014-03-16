############# Distribution types vectorizations   ################
# single parameter distributions
import Distributions: Bernoulli, TDist, Exponential, Poisson

for d in [:Bernoulli, :TDist, :Exponential, :Poisson]  
	@eval begin
		function ($d)(ps::Array)
			ds = Array($d, size(ps))
			for i in 1:length(ds)
				ds[i] = ($d)(ps[i])
			end
			ds
		end
	end 
end

# two parameter distributions
import Distributions: Normal, Uniform, Weibull, Gamma, Cauchy, LogNormal, Binomial, Beta, Laplace

for d in [:Normal, :Uniform, :Weibull, :Gamma, :Cauchy, :LogNormal, :Binomial, :Beta, :Laplace]
	@eval begin
		function ($d)(p1::Array, p2::Array)
			ds = Array($d, size(p1))
			for i in 1:length(ds)
				ds[i] = ($d)(p1[i], p2[i])
			end
			ds
		end

		function ($d)(p1::Array, p2::Real)
			ds = Array($d, size(p1))
			for i in 1:length(ds)
				ds[i] = ($d)(p1[i], p2)
			end
			ds
		end

		function ($d)(p1::Real, p2::Array)
			ds = Array($d, size(p2))
			for i in 1:length(ds)
				ds[i] = ($d)(p1, p2[i])
			end
			ds
		end
	end 
end

############# logpdf vectorization on the distribution argument   ################
import Distributions: logpdf, logcdf, logccdf

function logpdf{T<:Distribution}(ds::Array{T}, x::AbstractArray)
	res = Array(Float64, size(ds))
	size(ds) == size(x) || error("x and distributions sizes do not match")
	for i in 1:length(x)
		res[i] = logpdf(ds[i], x[i])
	end
	res
end

function logcdf{T<:Distribution}(ds::Array{T}, x::AbstractArray)
	res = Array(Float64, size(ds))
	size(ds) == size(x) || error("x and distributions sizes do not match")
	for i in 1:length(x)
		res[i] = logcdf(ds[i], x[i])
	end
	res
end

function logccdf{T<:Distribution}(ds::Array{T}, x::AbstractArray)
	res = Array(Float64, size(ds))
	size(ds) == size(x) || error("x and distributions sizes do not match")
	for i in 1:length(x)
		res[i] = logccdf(ds[i], x[i])
	end
	res
end
