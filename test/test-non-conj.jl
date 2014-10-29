using Turing
using Distributions

const p = 0.5

#@sf geom() = normal(0, 1) > 0. ? 1+geom() : 1
@sf geom(p) = bool(bernoulli(p)) ? 1+geom(p) : 1

@sf test_geom() = begin
    a = geom(0.5)
    normal(a, 3; condition=15)
    a
end

as = float(sample(test_geom, 10^6))

ms = mean(as)
vs = sqrt(var(as))

#Analytic
n = 30
xs = 1:n
prior = p.^xs 
likelihood = pdf(Normal(15, 3), xs)
posterior = prior .* likelihood
norm_posterior = posterior/sum(posterior)
ma = sum(norm_posterior.*xs)
va = sqrt(sum(norm_posterior.*xs.^2) - ma^2)

println((ms, ma))
println((vs, va))

