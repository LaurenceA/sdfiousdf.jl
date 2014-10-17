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
print(sample(mixture_model, 10^4)[end])

