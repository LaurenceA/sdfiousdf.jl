using Distributions
using ReverseDiffOverload
import Distributions.logpdf

test(dist, x, params...) = begin
    @assert isapprox(logpdf(dist, x, params...), logpdf(dist(params...), x))
    if dist <: DiscreteDistribution
        testdiff((params...) -> logpdf(dist, x, params...), params...)
    else
        testdiff((x, params...) -> logpdf(dist, x, params...), x, params...)
    end
end

#Normal
valid(::Type{Normal}, x, μ, σ) = begin
    @assert all(σ .> 0)
end
#Differentiable versions.
logpdf(::Type{Normal}, x, μ, σ) = begin
    #valid(Normal, x, μ, σ)
    σ2 = σ.*σ
    diff = x-μ
    -(log(2*pi*σ2) + diff.*diff ./ σ2)/2
end
test(Normal, 1, 2, 3)


#MvNormal
valid(::Type{MvNormal}, x::Vector, μ::Vector, Σ::Matrix) = begin
    @assert isposdef(Σ)
end
#Differentiable versions.
logpdf(::Type{MvNormal}, x::Vector, μ::Vector, Σ::Matrix) = begin
    #valid(MvNormal, x, μ, Σ)
    diff = x-μ
    -(length(x)*log(2*pi) + logdet(Σ) + (diff'/Σ)*x)/2
end
#test(MvNormal, [1., 2], [3., 4], eye(2) + ones(2,2))


#Gamma
valid(::Type{Gamma}, x, shape, scale) = begin
    @assert all(x .>= 0)
    @assert all(shape .>= 0)
    @assert all(scale .>= 0)
end
logpdf(::Type{Gamma}, x, shape, scale) = begin
    #valid(Gamma, x, shape, scale)
    -logΓ(shape) - shape.*log(scale) + (shape-1).*log(x) - x./scale
end
test(Gamma, 1, 2, 3)


#InverseGamma
valid(::Type{InverseGamma}, x, shape, scale) = begin
    @assert all(x .>= 0)
    @assert all(shape .>= 0)
    @assert all(scale .>= 0)
end
logpdf(::Type{InverseGamma}, x, shape, scale) = begin
    #valid(InverseGamma, x, shape, scale)
    -logΓ(shape) + shape.*log(scale) - (shape+1).*log(x) - scale./x
end
test(InverseGamma, 1, 2, 3)


#Bernoulli
valid(::Type{Bernoulli}, x, p) = begin
    @assert all(0 .<= p .<= 1)
    @assert all((x .== 1)|(x .== 0))
end
logpdf(::Type{Bernoulli}, x, p) = begin
    #valid(Bernoulli, x, p)
    x*log(p) + (1-x)*log(1-p)
end
test(Bernoulli, 1., 0.9)
test(Bernoulli, 0., 0.9)


#Categorical
valid(::Type{Categorical}, x, ps::Vector{Float64}) = begin
    @assert isapprox(sum(ps), 1)
    @assert all(0 .<= ps .<= 1)
end
logpdf(::Type{Categorical}, x, ps) = begin
    #valid(Categorical, x, ps)
    log(ps[x])
end
test(Categorical, 2, [0.3, 0.4, 0.3])


#Beta
valid(::Type{Beta}, x, α, β) = begin
    @assert all(α .> 0)
    @assert all(β .> 0)
    @assert all(0 .<= x .<= 1)
end
logB(α, β) = logΓ(α) + logΓ(β) - logΓ(α + β)
logpdf(::Type{Beta}, x, α, β) = begin
    #valid(Beta, x, α, β)
    (α-1).*log(x) + (β-1).*log(1-x) - logB(α, β)
end
test(Beta, 0.6, 1.3, 1.5)


#Dirichlet
valid(::Type{Dirichlet}, x, α::Vector{Float64}) = begin
    @assert all(0 .<= x)
    @assert all(0 .<= α)
end
logB(α) = sum(logΓ(α)) - logΓ(sum(α))
logpdf(::Type{Dirichlet}, x, α) = begin
    #valid(Dirichlet, x, α)
    sum((α-1).*log(x)) - logB(α)
end
test(Dirichlet, [1.2, 1.6, 1.3], [1.3, 1.6, 1.3])
    

#Poisson
valid(::Type{Poisson}, k, λ) = begin
    @assert all(k .>= 0)
    @assert all(λ .>= 0)
end
logpdf(::Type{Poisson}, k, λ) = begin
    #valid(Poisson, k, λ)
    k*log(λ) - log(factorial(k)) - λ
end
test(Poisson, 3, 5)

#Non-differentiable fallbacks from Distributions.jl
logpdf{D <: Distribution}(d::Type{D}, x, params...) = 
    logpdf(d(params...), x)
