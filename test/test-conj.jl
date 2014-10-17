using Turing
using Distributions

const n_samples = 10^5

@sf test_norm() = begin
    a = normal(0, 2; sampler=langevinposterior)
    normal(a, 1; condition=3)
    a
end

as = float(sample(test_norm, n_samples))
ms = mean(as)
vs = var(as)

#Analytic Moments
va =  1/(1/4 + 1/1)
ma = va*(3/1 + 0/4)

@assert 0.95*ma < ms < 1.05*ma
@assert 0.95*va < vs < 1.05*va

#@sf test_gamma_poisson() = begin
#    rate = gamma(2, 2)
#    for i = 1:10
#        poisson(rate; condition=10)
#    end
#    rate
#end

@sf test_beta_bernoulli() = begin
    p = beta(0.5, 1.5)#, sampler=langevinposterior)
    for i = 1:10
        bernoulli(p; condition=1)
    end
    p
end

as = float(sample(test_beta_bernoulli, n_samples))
dist = Beta(9.5, 1.5)
ma = mean(dist)
va = var(dist)
ms = mean(as)
vs = var(as)

@assert 0.95*ma < ms < 1.05*ma
@assert 0.8*va < vs < 1.2*va
