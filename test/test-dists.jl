using Turing

const n_samples = 10^4

#Sample from transformed distributions, check moments.
within_4_stdev(dist::Distribution, samples) = begin
    s = sqrt(var(dist))
    m = mean(dist)
    n = length(samples)
    stdevs = 4
    @assert 0.95*s < sqrt(var(samples)) < 1.05*s
    @assert m-stdevs*s/sqrt(n) < mean(samples) < m+stdevs*s/sqrt(n)
end

macro test_dist(Dist)
    dist = symbol(lowercase(string(Dist.args[1])))
    args = Dist.args[2:end]
    quote
        within_4_stdev($Dist, $dist(State(), $(args...); dims=($n_samples,)))
    end
    #dist = deepcopy(Dist)
    #dist.args[1] = symbol(lowercase(string(dist.args[1])))
    #insert!(dist.args, 2, :(State()))
    #push!(dist.args, :(($n_samples,)))
    #:(within_4_stdev($Dist, $dist))
end

@test_dist Normal(2, 3)
@test_dist LogNormal(1., 0.5)
#@test_dist Chisq(2)
#MvNormal
m = [1., 2.]
v = [5. -1; -1 3.]
samples = [mvnormal(State(), m, v) for i =1:n_samples]
println((m, mean(samples)))
println((v, cov(hcat(samples...)')))

@test_dist Exponential(3.)
@test_dist Bernoulli(0.7)
#@test_dist Uniform()
@test_dist Laplace(1., 3.)
@test_dist Categorical([0.1, 0.3, 0.6])
@test_dist Gamma(1., 3.)
@test_dist Beta(1., 3.)
#
#Dirichlet
println(); println(); println(); 
alpha = [1., 1.5, 0.5]
dist = Dirichlet(alpha)
samples = [dirichlet(State(), alpha) for i =1:n_samples]
println((mean(dist), mean(samples)))
println((cov(dist), cov(hcat(samples...)')))

#Multidimensional sampling fallbacks work
laplace(State(), ones(3), 1.)
laplace(State(), ones(3), ones(3))
gamma(State(), ones(3), 1.)
gamma(State(), ones(3), ones(3))
