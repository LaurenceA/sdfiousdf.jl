using Turing

#Returns a categorical distribution.
@sf fdirichlet(alpha) = begin
    ps = dirichlet(alpha)
    () -> categorical(ps)
end

#Makes 10 draws from the categorical distribution returned by fdirichlet.
@sf fdp() = begin
    dist = fdirichlet(ones(10)/5)
    Int[dist() for i = 1:30]
end

#Because we haven't conditioned on anything, we only need to draw one sample.
println(sample(fdp, 1)[end])
