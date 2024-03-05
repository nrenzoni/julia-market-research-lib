using DataFrames, Arrow

function py_bytes_to_arrow_table(py_bytes)
    jbytes = copy(reinterpret(UInt8, py_bytes))
    return Arrow.Table(jbytes)
end

function table_to_py_bytes(table)
    io = IOBuffer()
    Arrow.write(io, table)
    seekstart(io)
    return PyCall.pybytes(take!(io))
end

gentup(struct_T) = NamedTuple{(fieldnames(struct_T)...,),Tuple{(fieldtype(struct_T, i) for i = 1:fieldcount(struct_T))...}}

@generated function to_named_tuple_generated(x)
    nt = Expr(:quote, gentup(x))
    tup = Expr(:tuple)
    for i = 1:fieldcount(x)
        push!(tup.args, :(getfield(x, $i)))
    end
    return :($nt($tup))
end

to_named_tuples(d::Dict) = (; (p.first => to_named_tuples(p.second) for p in d)...)
to_named_tuples(d) = d