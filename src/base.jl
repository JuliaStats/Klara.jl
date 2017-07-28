IntegerVector{I<:Integer} = Vector{I}
RealVector{N<:Real} = Vector{N}
RealMatrix{N<:Real} = Matrix{N}

RealLowerTriangular{T<:Real, S<:AbstractMatrix} = LowerTriangular{T, S}

RealNormal{N<:Real} = Normal{N}
MultivariateGMM{D<:MvNormal} = MultivariateMixture{Continuous, D}

multivecs{T}(::Type{T}, n::Int) = [T[] for _ =1:n]
