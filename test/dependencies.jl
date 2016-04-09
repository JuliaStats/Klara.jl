using Base.Test
using Graphs
using Lora

println("    Testing conversion of dependencies to edges compatible with Graphs...")

θ = BasicContUnvParameter(:θ, 1, signature=:low)
x = Data(:x, 1)

dxθ = Dependence(1, x, θ)

gdxθ = convert(Edge, dxθ)

@test dxθ.index == gdxθ.index
@test dxθ.source.index == gdxθ.source.index
@test dxθ.source.key == gdxθ.source.key
@test dxθ.target.index == gdxθ.target.index
@test dxθ.target.key == gdxθ.target.key

λ = Hyperparameter(:λ, 2)

dxλ = Dependence(1, x, λ)

medges = [dxθ, dxλ]
gedges = convert(Vector{Edge}, medges)

for i in 1:length(medges)
  @test medges[i].index == gdxθ.index
  @test medges[i].source.index == gedges[i].source.index
  @test medges[i].source.key == gedges[i].source.key
  @test medges[i].target.index == gedges[i].target.index
  @test medges[i].target.key == gedges[i].target.key
end
