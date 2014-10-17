abstract Lazy{T}

type LazyF{T} <: Lazy{T}
    f::Function
    evaluated::Bool
    deps::Vector{LazyF}
    value::T
    LazyF(f::Function) = new(f, false, LazyF[])
end
LazyF(f::Function, T::Type=Any) = LazyF{T}(f)

type LazyV{T} <: Lazy{T}
    deps::Vector{LazyF}
    value::T
    LazyV(value) = new(LazyF[], value)
end
LazyV{T}(value::T) = LazyV{T}(value)

value(l::LazyV) = l.value
value(l::LazyF) = begin
    if !l.evaluated
        l.value = l.f()
        l.evaluated = true
    end
    l.value
end

for op in [+, -, *, /]
    op(x::Lazy, y::Lazy) = begin
        res = LazyF(() -> op(value(x), value(y)))
        push!(x.deps, res)
        push!(y.deps, res)
        res
    end
    op(x, y::Lazy) = begin
        res = LazyF(() -> op(value(x), value(y)))
        push!(y.deps, res)
        res
    end
    op(x::Lazy, y) = begin
        res = LazyF(() -> op(value(x), value(y)))
        push!(x.deps, res)
        res
    end
end

unevaluate(lt::Lazy) = begin
    for l in lt.deps
        l.evaluated=false
        unevaluate(l)
    end
    nothing
end
update!(l::Lazy, value) = begin
    unevaluate(l)
    l.value = value
end

If(cond::Bool, val_true::Function, val_false::Function) =
    cond ? val_true() : val_false

#Testing
x = LazyV(3)
y = LazyV(4)
xpy = x+y
@assert 7 == value(xpy)
update!(x, 5)
@assert xpy.value == 7
@assert 9 == value(xpy)

c = LazyV(true)
v = If(c, val_true, val_false)

#Output types for functions.
immutable Symb{s}
end
Symb(s::Symbol) = Symb{s}()
f_type(s::Symb, args::Any...) = Any

