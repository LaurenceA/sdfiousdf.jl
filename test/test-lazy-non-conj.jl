using Turing

@sf geom() = bool(bernoulli(0.4)) ? 1+geom() : 1

n = 4
p = 0.5
@sf test_geom() = begin
    a = 0.
    all_true = 1.
    for i = 1:(n+1)
        a += all_true
        all_true = all_true * (normal(0, 1) < 0)
    end
#    v = Array(Int, n)
#    for i = 1:n
#        v[i] = bernoulli(p)
#    end
#    v = bernoulli(p; dims=(n,))
#    a = sum(cumprod(v)) + 1
    normal(a, 3; condition=15)
    a
end

as = float(sample(test_geom, 10^6))

ms = mean(as)
vs = sqrt(var(as))

#Analytic
xs = 1:(n+1)
prior = [p.^(1:n), p^(n-1)*(1-p)]
@assert isapprox(sum(prior), 1.)
likelihood = pdf(Normal(15, 3), xs)
posterior = prior .* likelihood
norm_posterior = posterior/sum(posterior)
ma = sum(norm_posterior.*xs)
va = sqrt(sum(norm_posterior.*xs.^2) - ma^2)

println((ms, ma))
println((vs, va))

