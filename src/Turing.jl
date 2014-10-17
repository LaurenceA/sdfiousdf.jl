module Turing

using Distributions
using ReverseDiffOverload

export @sf, State, sample, normal, langevinposterior, If_, EndIf_

immutable NoCond end
const nocond = NoCond()

abstract TraceElement
Trace = Vector{TraceElement}
type If <: TraceElement
    cond::Bool
    trace::Vector{TraceElement}
end

immutable Resample <: TraceElement end
const resample = Resample()

abstract AbstractWhitened
type Whitened <: AbstractWhitened end
const whitened = Whitened()
type UnWhitened <: AbstractWhitened end
const unwhitened = UnWhitened()

abstract Sampler
abstract MH <: Sampler
type Gibbs <: MH end
const gibbs = Gibbs()
type GibbsLazy <: MH end
const gibbslazy = GibbsLazy()
type LangevinPrior <: MH end
const langevinprior = LangevinPrior()
type LangevinPosterior <: MH end
const langevinposterior = LangevinPosterior()
type Conditioned <: Sampler end
const conditioned = Conditioned()

type Sample{W <: AbstractWhitened, S <: Sampler, V, L} <: TraceElement
  value::V
  logpdf::L
end

value(call::ReverseDiffOverload.Call) = value(call.val)
value(sample::Sample) = value(sample.value)
value(v) = v

Trace = Vector{TraceElement}
type State
    trace::Vector{TraceElement}
    counter::Vector{Int}
    current_frame::Vector{TraceElement}
end
State(trace::Trace=TraceElement[]) = State(trace, [0], trace)

Sample{W <: AbstractWhitened, S <: Sampler, V, L}(::W, ::S, value::V, logpdf::L) =
    Sample{W, S, V, L}(value, logpdf)

stochastic_functions = Set([:white_rand, :scall])

white_logpdf(x) = -sum(x.*x)/2
white_rand(state::State, dims::(Int...), sampler::Sampler) = begin
    state.counter[end] += 1
    if isend(state, state.counter[end])
        val = randn(dims...)
        push!(state, Sample(whitened, sampler, val, white_logpdf(val)))
    else
        val = state[state.counter[end]].value
        state[state.counter[end]] = Sample(whitened, sampler, val, white_logpdf(val))
    end
    #@assert size(value(state.trace[state.counter])) == dims
    state[state.counter].value
end

SI = Union(State, If)
import Base.getindex
getindex(si::State, counter::Vector{Int}) = begin
    for i = 1:length(counter)
        si = si.trace[counter[i]]
    end
    si
end
getindex(si::State, counter::Int) = si.current_frame[counter]
import Base.setindex!
setindex!(si::State, value::TraceElement, counter::Vector{Int}) = begin
    for i = 1:(length(counter) - 1)
        si = si.trace[counter[i]]
    end
    si.trace[counter[end]] = value
end
setindex!(si::State, value::TraceElement, counter::Int) = 
    setindex!(si.current_frame, value, counter)
isend(si::State, counter::Vector{Int}) = begin
    for i = 1:(length(counter) - 1)
        si = si.trace[counter[i]]
    end
    @assert length(si.trace)+1 >= counter[end]
    counter[end] == length(si.trace)+1
end
isend(si::State, counter::Int) = begin
    @assert length(si.current_frame)+1 >= counter
    counter == length(si.current_frame)+1
end
import Base.push!
push!(si::State, value) = begin
    #si_base = si
    #counter = si.counter
    #@assert isend(si, counter)
    #for i = 1:(length(counter) - 1)
    #    si = si.trace[counter[i]]
    #end
    #@assert si.trace == si_base.current_frame
    #push!(si.trace, value)
    push!(si.current_frame, value)
end

If_(state::State, cond::Bool) = begin
    state.counter[end] += 1
    if isend(state, state.counter[end])
        push!(state, If(cond, TraceElement[]))
    elseif state[state.counter[end]].cond != cond
      state[state.counter[end]] = If(cond, TraceElement[])
    end
    state.current_frame = state.current_frame[state.counter[end]].trace
    push!(state.counter, 0)
    nothing
end

EndIf_(state) = begin
    pop!(state.counter)
    #Slow line
    state.current_frame = state[state.counter[1:(length(state.counter)-1)]].trace
    nothing
end

propagate_state(ex::Expr) = 
    if (ex.head == :call) && in(ex.args[1], stochastic_functions)
        if (length(ex.args) >= 2) && isa(ex.args[2], Expr) && (ex.args[2].head == :parameters)
            Expr(:call, ex.args[1:2]..., :(state::State), map(propagate_state, ex.args[3:end])...)
        else
            Expr(:call, ex.args[1], :(state::State), map(propagate_state, ex.args[2:end])...)
        end
    else
        Expr(ex.head, map(propagate_state, ex.args)...)
    end
propagate_state(ex) = ex

wrap_if(ex::Expr) =
    if ex.head == :if
        if length(ex.args) == 2
            Expr(:if, ex.args[1], wrap_if(ex.args[2], true))
        elseif length(ex.args) == 3
            Expr(:if, ex.args[1], wrap_if(ex.args[2], true), wrap_if(ex.args[3], false))
        else
            error()
        end
    else
        Expr(ex.head, map(wrap_if, ex.args)...)
    end
wrap_if(ex) = ex
wrap_if(ex, cond::Bool) = quote
    If_(state, $cond)
    res = $(wrap_if(ex))
    EndIf_(state)
    res
end

macro sf(ex)
    f_name = ex.args[1].args[1]
    push!(stochastic_functions, f_name)
    esc(wrap_if(propagate_state(ex)))
end

include("samplers.jl")
include("dists.jl")
include("logpdf.jl")
end
