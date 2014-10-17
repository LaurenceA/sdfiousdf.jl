samplers = [gibbs, langevinprior, langevinposterior, gibbslazy]

sample_pop(f::Function, iters::Int, n::Int=10) = begin
    states = State[State() for i = 1:n]
    results = map(f, states)
    ress = Array(Any, iters*n)
    for i = 1:iters
        state_index = rand(Categorical(n))
        if rand() < 0.5
            (proposed_state, proposed_result) = 
                sample(langevinprior, f, states[state_index])
            logp = logpdf_diff(proposed_state, states[state_index])
        else
            (proposed_state, proposed_result) = 
                elliptic(f, states[state_index], states[rand(Categorical(n))])
            logp = logpdf_diff(proposed_state, states[state_index])
        end
        (state, res) = sample(samplers[rand(Categorical(current_ps))], f, state, res)
        ress[i] = res
    end
    ress
end

import StatsBase.sample
sample(f::Function, iters::Int; ps=ones(length(samplers))) = begin
    state = State()
    res = f(state)
    ress = Array(Any, iters)
    for i = 1:iters
        current_ps = ps .* map(sampler -> is_sampler_applicable(sampler, state), samplers)
        current_ps = current_ps/sum(current_ps)
        (state, res) = sample(samplers[rand(Categorical(current_ps))], f, state, res)
        ress[i] = res
    end
    ress
end

is_sample{W, S <: Sampler}(::S, ::Sample{W, S}) = true
is_sample(::Sampler, ::TraceElement) = false

is_sampler_applicable_elements(s::Sampler, state::State) = 
    map(el -> is_sample(s, el), state.trace)
is_sampler_applicable(s::Sampler, state::State) = 
    any(is_sampler_applicable_elements(s, state))

sample(sampler::MH, f::Function, old_state::State, old_res) = begin
    #ngso = n_gibbs_samples(old_state)
    (proposed_state, proposed_res, proposal_logpdf_diff) = sample(sampler, f, old_state)
    #@assert ngso == n_gibbs_samples(old_state)
    log_ap = value(logpdf_diff(sampler, proposed_state, old_state)) + proposal_logpdf_diff
    if rand() < exp(log_ap)
        (proposed_state, proposed_res)
    else
        (old_state, old_res)
    end
end

abstract Count
immutable Lf <: Count
    count::Int
end
immutable Br <: Count
    count::Int
    leaves::Vector{Count}
end
count(sampler::Sampler, si::SI) = begin
    total = 0
    leaves = Array(Count, length(si.trace))
    for i = 1:length(si.trace)
        leaves[i] = count(sampler, si.trace[i])
        total += leaves[i].count
    end
    Br(total, leaves)
end
count{W, S <: Sampler}(::S, ::Sample{W, S}) = Lf(1)
count{S <: Sampler}(s::S, sample::Sample) = Lf(0)
    
resample_addr(c::Count) = begin
    counter = Int[]
    resample_addr(counter, c)
    counter
end
resample_addr(counter::Vector{Int}, c::Lf) = nothing
resample_addr(counter::Vector{Int}, c::Count) = begin
    ns = map(x -> x.count, c.leaves)
    push!(counter, rand(Categorical(ns/sum(ns))))
    resample_addr(counter, c.leaves[counter[end]])
    nothing
end

import Base.deepcopy
dc(s::State) = State(dc(s.trace))
dc(t::Vector{TraceElement}) = TraceElement[dc(x) for x in t]
dc(i::If) = If(i.cond, dc(i.trace))
#Don't copy when not necessary
dc(s::Sample{Whitened}) = s
dc{S}(s::Sample{UnWhitened, S}) = Sample(unwhitened, S(), s.value, s.logpdf)
dc(s::Sample{Whitened, Conditioned}) = error()
sample(::Gibbs, f::Function, old_state::State) = begin
    old_count = count(gibbs, old_state)
    counter = resample_addr(old_count)
    proposed_state = dc(old_state)
    proposed_state[counter] = resample
    proposed_res = f(proposed_state)
    proposed_count = count(gibbs, proposed_state)
    proposal_diff = log(old_count.count) - log(proposed_count.count) +
                    old_state[counter].logpdf - proposed_state[counter].logpdf 
    (proposed_state, proposed_res, proposal_diff)
end
#Gibbs
isgibbs{W}(::Sample{W, Gibbs}) = true
isgibbs(_) = false
isunwhitened(::Sample{UnWhitened}) = true
isunwhitened(u) = false

#GibbsLazy
sample(::GibbsLazy, f::Function, old_state::State) = begin
    
end

#Langevin prior
sample(::LangevinPrior, f::Function, old_state::State) = begin
    proposed_state = State(TraceElement[langevinprior_(x) for x in old_state.trace])
    (proposed_state, f(proposed_state), 0.)
end
langevinprior_(s::Sample{Whitened, LangevinPrior}) = begin
    new_val = (1-1/tau)*s.value + sqrt(2/tau)*randn(size(s.value)...)
    Sample(whitened, langevinprior, new_val, white_logpdf(new_val))
end
langevinprior_(i::If) =
    If(i.cond, TraceElement[langevinprior_(x) for x in i.trace])
langevinprior_(s::Sample{UnWhitened, LangevinPrior}) = error()
langevinprior_(el::TraceElement) = el

sample(::LangevinPosterior, f::Function, old_state::State) = begin
    #Propagate differentiated variables.
    old_state = diff_rand(old_state)
    f(old_state)
    #Differentiate logpdf.
    logpdf = sum(map(el -> el.logpdf, old_state.trace))
    ReverseDiffOverload.diff(logpdf)
    #Take a gradient step
    proposed_state = diff_rand(grad_step(old_state))#State(TraceElement[diff_rand(grad_step(x)) for x in old_state.trace])
    proposed_res = value(f(proposed_state))
    #Differentiate logpdf.
    logpdf = sum(map(el -> el.logpdf, proposed_state.trace))
    ReverseDiffOverload.diff(logpdf)
    #println()
    pp = - proposal_p(old_state, proposed_state) + proposal_p(proposed_state, old_state)
    (proposed_state, proposed_res, pp)
end

printlnval(x) = (println(x); x)
diff_rand{W}(s::Sample{W, LangevinPosterior}) = 
    Sample(W(), langevinposterior, ReverseDiffOverload.Call(value(s)), nothing)
diff_rand{W, S}(s::Sample{W, S}) = s
diff_rand(si::SI) = map(diff_rand, si)
import Base.map
map(f::Function, state::State) = State(TraceElement[f(x) for x in state.trace])
map(f::Function, i::If) = If(i.cond, TraceElement[f(x) for x in i.trace])

const tau = 100.
proposal_mean{W}(s::Sample{W, LangevinPosterior}) = s.value.val + s.value.dval/tau
#proposal_mean(s::Sample{UnWhitened, LangevinPosterior}) = s.value.val + s.value.dval/tau
#proposal_mean(s::Sample{Whitened, LangevinPosterior}) = (1-1/tau)*s.value.val + s.value.dval/tau
grad_step{W}(s::Sample{W, LangevinPosterior}) = begin
    res = proposal_mean(s) + sqrt(2/tau)*randn(size(s.value.val)...)
    Sample(W(), langevinposterior, res, nothing)
end
grad_step(si::SI) = map(grad_step, si)
grad_step(s::Sample{UnWhitened}) = dc(s)
grad_step(s::Sample{Whitened}) = s

proposal_p(state_from::State, state_to::State) = sum(map(proposal_p, state_from.trace, state_to.trace))
proposal_p{W}(sample_from::Sample{W, LangevinPosterior}, sample_to::Sample{W, LangevinPosterior}) =
    logpdf(Normal, value(sample_to), proposal_mean(sample_from), sqrt(2/tau))
proposal_p{W, V}(::Sample{W, LangevinPosterior}, ::Sample{V, LangevinPosterior}) = error()
proposal_p(sample_from::Sample, sample_to::Sample) = 0.

logpdf_diff(sampler::Sampler, proposed_trace::Vector{TraceElement}, old_trace::Vector{TraceElement}) = begin
    total = 0.
    @assert length(proposed_trace) == length(old_trace)
    for i = 1:length(proposed_trace)
        total += logpdf_diff(sampler, proposed_trace[i], old_trace[i])
    end
    total
end
logpdf_diff(sampler::Sampler) = (xy) -> logpdf_diff(sampler, xy[1], xy[2])
logpdf_diff(sampler::Sampler, proposed_si::State, old_si::State) = 
    logpdf_diff(sampler, proposed_si.trace, old_si.trace)
logpdf_diff(sampler::Sampler, proposed_si::If, old_si::If) = 
    if proposed_si.cond == old_si.cond
        logpdf_diff(sampler, proposed_si.trace, old_si.trace)
    else
        0.
    end
logpdf_diff(sampler::LangevinPrior, proposed_sample::Sample{UnWhitened}, old_sample::Sample{UnWhitened}) =
    proposed_sample.logpdf - old_sample.logpdf
logpdf_diff(sampler::LangevinPrior, proposed_sample::Sample{Whitened}, old_sample::Sample{Whitened}) = 0.
logpdf_diff(sampler::Sampler, proposed_sample::Sample, old_sample::Sample) =
    proposed_sample.logpdf - old_sample.logpdf
