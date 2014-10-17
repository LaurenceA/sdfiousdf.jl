#Machinery to convert calls to random variables to scalls, e.g.
#normal(a,3) = scall(Normal, (), nocond, a, 3).
#scall has no optional args, so we can exploit type information properly.

as = map(i -> symbol(string("a", i)), 1:10)
#typed_as = map(a -> :($a::Union(Number, Array)), as)
macro entry(Dist, i) 
    dist = symbol(string(lowercase(string(Dist))))
    esc(quote
            @sf function $dist($(as[1:i]...); dims::(Int...)=(), condition=nocond, whitened=default_whitened(condition, $Dist), sampler=default_sampler(condition, $Dist))
                scall($Dist, dims, condition, whitened, sampler, $(as[1:i]...))
            end
            $(Expr(:export, symbol(lowercase(string(Dist)))))
        end)
end
@entry(Normal, 2)
@entry(MvNormal, 2)
@entry(Chisq, 1)
@entry(Exponential, 1)
@entry(Bernoulli, 1)
@entry(Cauchy, 2)
@entry(Laplace, 2)
import Base.gamma
@entry(Gamma, 2)
@entry(Beta, 2)
@entry(Dirichlet, 1)
@entry(Categorical, 1)

default_whitened(::NoCond, _) = whitened
default_whitened(_, ::Type) = unwhitened
default_whitened(::NoCond, ::Type{Bernoulli})   = unwhitened
default_whitened(::NoCond, ::Type{Categorical}) = unwhitened

default_sampler(::NoCond, _) = langevinprior
default_sampler(_, ::Type) = conditioned
default_sampler(::NoCond, ::Type{Bernoulli})   = gibbs
default_sampler(::NoCond, ::Type{Categorical}) = gibbs

const VectorParams = Union(MvNormal, Dirichlet, Categorical)
const RealParams   = Union(Normal, LogNormal, Chisq, Exponential, Bernoulli, Cauchy, Laplace, Gamma, Beta)

const AN = Union(Array, Number)

#Conditioning functions.
scall(state::State, D, dims::(Int...), x::Union(Number, Array), ::UnWhitened, ::Conditioned, params...) = begin
    state.counter[end] += 1
    if isend(state, state.counter[end])
        push!(state, Sample(unwhitened, conditioned, x, logpdf(D, x, params...)))
    else
        state[state.counter[end]] = Sample(unwhitened, conditioned, x, logpdf(D, x, params...))
    end
    x
end

#Sampling functions
#Transform N(0, 1), (parameter independent)
@sf lognormal(μ, σ; condition=nocond, dims=()::(Int...)) = 
    lognormal(μ, σ, dims, condition)
@sf lognormal(μ, σ, dims::(Int...), cond::NoCond) =
    exp(normal(μ, σ; dims=dims))
@sf lognormal(μ, σ, dims::(Int...), cond) =
    exp(normal(μ, σ; condition=log(cond), dims=dims))

#Transform N(0, 1), (parameter dependent)
macro direct_sampler(Dist, expr)
    name = Dist.args[1]
    params = Dist.args[2:end] 
    esc(quote 
        @sf scall(::Type{$name}, dims, ::NoCond, ::Whitened, sampler::Sampler, $(params...)) = $expr
    end)
end
@direct_sampler(Normal(μ, σ), μ .+ σ.*white_rand(maxsize(dims, μ, σ), sampler))
@direct_sampler(MvNormal(μ, Σ), μ + sqrtm(Σ)*white_rand((length(μ),), sampler))
@direct_sampler(Chisq(k::Int), squeeze(sum(white_rand(tuple(k,dims...), sampler.value).^2, 1), 1))
    #Only works for multiple samples, not multiple

#Transform cdf
normal_cdf(z) = 0.5*erfc(-z/√2)
@sf scall(D, dims, ::NoCond, ::Whitened, sampler::Sampler, params...) =
    quantile(D, normal_cdf(white_rand(maxsize(D, dims, params...), sampler)), params...)
#Non-differentiable fallback using Distributions.jl
import Distributions.quantile
quantile{D <: UnivariateDistribution}(d::Type{D}, p, params...) = begin
    quantile_(d, p, params...)
end
quantile_{D <: RealParams}(d::Type{D}, p, params::Number...) = 
    quantile(d(params...), p)
import Base.broadcast
broadcast(d::DataType, argss...) =
    broadcast!((args...) -> d(args...), Array(d, maxsize(argss...)), argss...)
quantile_{D <: RealParams}(d::Type{D}, p, params...) = begin
    #Check for subtle bug where not enough gaussian random variables are drawn.
    @assert size(p) == maxsize(params...)
    broadcast(quantile, broadcast(d, params...), p)
end
quantile_{D <: VectorParams}(d::Type{D}, p, params...) = 
    quantile(d(params...), p)
#Differentiable definitions.
quantile(::Type{Exponential}, p, scale) = -scale.*log1p(-p)
quantile(::Type{Bernoulli}, p, p0) = p.<p0
quantile(::Type{Cauchy}, p, location, scale) = location .- scale.*cospi(p)./sinpi(p)

#laplace_quantile(p::Real, location::Real, scale::Real) =
#    p < 0.5 ? location + scale*log(2.0*p) : location - scale*log(2.0*(1.0-p))
#laplace_quantile(p, location, scale) = broadcast(laplace_quantile, p, location, scale)
#@cdf_transform(laplace(location, scale), laplace_quantile(p, location, scale))
#
#mapfirst(f::Function, ps, args...) = broadcast(p -> f(p, args...), ps)
#categorical_quantile(p::Real, pv::Vector) = begin
#    i = 1
#    v = pv[1]
#    while v < p && i < length(pv)
#        i += 1
#        @inbounds v += pv[i]
#    end
#    i
#end
#@cdf_transform(categorical(pv), mapfirst(categorical_quantile, p, pv))
#

#Directly call Rmath quantile, because fallbacks are super-slow
quantile_gamma(p::Real, shape::Real, scale::Real) = begin
    ccall((:qgamma, "libRmath-julia"),
          Float64, (Float64, Float64, Float64, Int32, Int32),
          p, shape, scale, 1, 0)
end
const delta = 1E-6
import ReverseDiffOverload.diff_con
#@d3(quantile_gamma, 
#    if 0.5<x
#        d*(quantile_gamma(x, y, z) - quantile_gamma(x-delta, y, z))/delta
#    else
#        d*(quantile_gamma(x+delta, y, z) - quantile_gamma(x, y, z))/delta
#    end,
#    d*(quantile_gamma(x, y+delta, z) - quantile_gamma(x, y, z))/delta,
#    d*(quantile_gamma(x, y, z+delta) - quantile_gamma(x, y, z))/delta)
quantile_gamma(x, y, z) = broadcast(quantile_gamma, x, y, z)
dx_quantile_gamma(x::Real, y::Real, z::Real) = 
    if 0.5<x
        (quantile_gamma(x, y, z) - quantile_gamma(x-delta, y, z))/delta
    else
        (quantile_gamma(x+delta, y, z) - quantile_gamma(x, y, z))/delta
    end
dx_quantile_gamma(x, y, z) = broadcast(dx_quantile_gamma, x, y, z)
dy_quantile_gamma(x, y, z) = (quantile_gamma(x, y+delta, z) - quantile_gamma(x, y, z))/delta
dz_quantile_gamma(x, y, z) = (quantile_gamma(x, y, z+delta) - quantile_gamma(x, y, z))/delta
@d3(quantile_gamma, (res = d.*dx_quantile_gamma(x, y, z); isa(x, Real) ? sum(res) : res),
                    (res = d.*dy_quantile_gamma(x, y, z); isa(y, Real) ? sum(res) : res),
                    (res = d.*dz_quantile_gamma(x, y, z); isa(z, Real) ? sum(res) : res))
testdiff(quantile_gamma, 0.4, 5., 6.)
testdiff(quantile_gamma, 0.6, 5., 6.)
testdiff((x, y, z) -> sum(quantile_gamma(x, y, z)), [0.4, 0.6], 5., 6.)
testdiff((x, y, z) -> sum(quantile_gamma(x, y, z)), 0.4, [5., 6], 6.)

quantile(::Type{Gamma}, args::Real...) = quantile_gamma(args...)
quantile(::Type{Gamma}, args...) = broadcast(quantile_gamma, args...)


#Transform gamma
@direct_sampler(Beta(alpha, beta), begin
    X = gamma(alpha, 1; dims=dims)
    Y = gamma(beta, 1; dims=dims)
    X./(X+Y)
end)
@direct_sampler(Dirichlet(alpha::Vector), begin
    X = gamma(alpha, 1)
    X/sum(X)
end)
    
#MaxDims reports the dimensionality of the largest incoming parameter.
#Common cases for speed.
maxsize(as::Number...) = ()
maxsize(a::Array)            = size(a)
maxsize(r::Number,  a::Array) = size(a)
maxsize(a::Array,   r::Number) = size(a)
#General case.
maxdims(as::AN...) = max(map(ndims, as)...)
maxsize(as::AN...) = map(i -> max(map(a -> size(a, i), as)...), tuple(1:maxdims(as...)...))
#Include dims argument.
maxsize(dims::(),       as::Number...) = ()
maxsize(dims::(Int...), as::Number...) = dims
maxsize(dims::(),       as::AN...)     = maxsize(as...)
#Deal with distributions with vector arguments.
maxsize{D <: RealParams}(d::Type{D}, dims, as...) = maxsize(dims, as...)
maxsize{D <: VectorParams}(d::Type{D}, dims, as...) = dims

#UnWhitened sample
scall(state::State, d, dims::(Int...), condition::NoCond, ::UnWhitened, sampler::Sampler, params...) = begin
    state.counter[end] += 1
    isend_     = isend(state, state.counter[end])
    if isend_ || isa(state[state.counter[end]], Resample)
        dist = d(params...)
        val = rand(dist, dims...)
        sample = Sample(unwhitened, sampler, val, sum(logpdf(dist, val)))
        if isend_
            push!(state, sample)
        else#if isresample
            state[state.counter[end]] = sample
        end
    end
    #val
    state[state.counter[end]].value
    #state.counter += 1
    #if (state.endifs_before_resume > 0) || 
    #   (state.counter == length(state.trace) + 1)
    #    dist = d(params...)
    #    value = rand(dist, dims...)
    #    insert!(state.trace, state.counter, 
    #            Sample(unwhitened, sampler, value, sum(logpdf(dist, value))))
    #elseif (state.trace[state.counter] == resample)
    #    dist = d(params...)
    #    value = rand(dist, dims...)
    #    state.trace[state.counter] = Sample(unwhitened, sampler, value, sum(logpdf(dist, value)))
    #end
    #state.trace[state.counter].logpdf = sum(logpdf(d(params...), state.trace[state.counter].value))
    #@assert size(state.trace[state.counter].value) == dims
    #state.trace[state.counter].value
end
