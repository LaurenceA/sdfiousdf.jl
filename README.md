# Turing

[![Build Status](https://travis-ci.org/LaurenceA/Turing.jl.svg?branch=master)](https://travis-ci.org/LaurenceA/Turing.jl)


Turing.jl aims to make it easy for anyone to perform MCMC inference in complex, and simple, probabilistic models.
We aim to be:
 - Fast
   - Supports gradient based inference techniques (e.g. MALA) in addition to standard Gibbs sampling.
 - Practical
 - General
   - Allows Turing complete probabilistic models
   - Whitens variables, to allow for arbitrarily strong prior dependencies

Getting started
---------------
To install, use
```julia
Pkg.add("Turing")
```
To load, use
```julia
using Turing
```

Constructing a model
---------------------
To construct a simple model, we might write,
```julia
@sf foo() = begin
    a = normal(0, 2)
    b = normal(1, 3)
    c = a+b
    c^2
end
```
Unpacking this call, we see that:
 - All code in Turing.jl should be wrapped in a function call, with the @sf macro applied.
 - We get random variables by calling the distribution name (with a lowercase first letter).
 - We can combine random variables using any functions.

We can sample from this probabilistic program by calling,
```julia
sample(foo, 10^5)
```
which gives `10^5` correlated random samples.

Conditioning
------------
So far, we haven't done anything interesting - you could sample `foo` simply by defining a suitable function.
Howerver, In Turing.jl, you can condition these draws on known data, using the keyword argument `condition`, for instance,
```julia
@sf foo() = begin
    a = normal(0, 2)
    b = normal(1, 3)
    c = a+b
    normal(c^2, 1; condition=3)
    c
end
```
This function now states that we use `c` to draw a random variable with distribution N(c^2, 0), and we condition that draw to be `3`.
Importantly, calling `sample` now returns samples of `c`, given that we drew a random variable from N(c^2, 1), which turned out to be `3`.

Example: mixture model - fixed number of components.
-------------------------------
Using only what we've seen so far, we can do quite alot.  For instance, to define and infer a mixture model, one would use the following code:
```julia
using Turing

#Generate some data
const data = [randn(10)+6, randn(10)-6]

@sf mixture_model() = begin
    #The model parameters
    K = 2
    ms = [normal(0, 10) for i = 1:K]
    vs = [gamma(2, 2) for i = 1:K]
    ps = dirichlet(ones(K))

    #Which mixture component does each data item belong to?
    ks = [categorical(ps) for i = 1:length(data)]

    for i = 1:length(data)
      #Condition on the data.
      normal(ms[ks[i]], vs[ks[i]]; condition=data[i])
    end
    (ms, vs, ps)
end
samples = sample(mixture_model, 10^4)
```

Defining new distributions
--------------------------
However, to really exploit the fact that the language is Turing complete, we need to define new distributions.
Really, the functions we've defined so far (e.g. `foo`) define new distributions --- just distributions that aren't useful in further compositions.
We can therefore give stochastic functions, or distributions, arguments, for instance, we could define a scale mixture of Gaussians by,
```julia
@sf gsm(m::Real) = normal(m, gamma(2, 2))
```
We can also allow conditioning on new distributions, if the final call also allows conditioning.  In this case,
```julia
@sf gsm(m::Real; condition=nocond) = normal(m, gamma(2, 2); condition=condition)
```

Not only can stochastic functions take arguments --- they can also be defined recursively --- giving a huge amount of flexibility.
For instance, we can define a geometric distribution using,
```julia
@sf geom(p) = bool(bernoulli(p)) ? 1+geom(p) : 1
```
Again, to condition calls to `geom` on data, we define another stochastic function, which includes a conditioned random variable,
```julia
@sf test_geom() = begin
    a = geom(0.5)
    normal(a, 3; condition=15)
    a
end
```

This link between distributions and (stochastic) functions allows us to write down a distribution over distributions simply as a function that returns a function.  For instance
```julia
#Returns a categorical distribution.
@sf fdirichlet(args...) = begin
    ps = dirichlet(args...)
    () -> categorical(ps)
end

#Makes 10 draws from the categorical distribution returned by fdirichlet.
@sf fdp() = begin
    dist = fdirichlet(9, 0.1)
    Int[dist() for i = 1:10]
end

#Because we haven't conditioned on anything, we only need to draw one sample.
println(sample(fdp, 1)[end])
```
