module MarketResearchFuncs

using DataFrames, Arrow, Dates

function py_bytes_to_arrow_table(py_bytes)
    jbytes = copy(reinterpret(UInt8, py_bytes))
    return Arrow.Table(jbytes)
end

function py_bytes_to_arrow_table_1(vec_of_py_bytes::Vector)
    s = Set()
    for item in vec_of_py_bytes
        push!(s, MarketResearchFuncs.py_bytes_to_arrow_table(item))
    end
    return s
end

function table_to_py_bytes(table)
    io = IOBuffer()
    Arrow.write(io, table)
    seekstart(io)
    return PyCall.pybytes(take!(io))
end

function calc_return(price_vec2::AbstractArray, price_vec1::AbstractArray, lookback::Integer)

    lookback < 0 && throw("lookback must be >= 1")

    len = size(price_vec2, 1)
    ret_vec = Vector{Float64}(undef, len)
    @inbounds for i in 1:lookback
        ret_vec[i] = 0.0
    end
    @inbounds for i in lookback+1:len
        ret_vec[i] = log(price_vec2[i] / price_vec1[i-lookback])
    end

    return ret_vec
end

@inline above_0(x) = x .> 0

function bytes_to_df_with_preprocessing(bytes, is_disallowmissing=true)
    df = py_bytes_to_arrow_table(bytes) |> DataFrame
    is_disallowmissing && disallowmissing!(df)
    subset!(df, :low => above_0, :high => above_0)
    return df
end

function calc_tr_vec(df::AbstractDataFrame)
    return calc_tr_vec(
        log.(df.high),
        log.(df.low),
        log.(df.close)
    )
end

function calc_tr_vec(high_vec::Vector{T}, low_vec::Vector{T}, close_vec::Vector{T}) where {T}
    len = size(high_vec, 1)
    tr_vec = Vector{Union{Float64,Missing}}(undef, len)
    tr_vec[1] = 0.0
    @inbounds for i in 2:len
        tr1 = high_vec[i] - low_vec[i]
        tr2 = high_vec[i] - close_vec[i-1]
        tr3 = low_vec[i] - close_vec[i-1]
        tr_vec[i] = max(tr1, tr2, tr3)
    end
    return tr_vec
end

function calc_expanding_rolling_mean(vec::Vector{T}, lookback::Integer) where {T}
    len = size(vec, 1)
    rolling_mean_sum::Float64 = 0
    rolling_mean_vec = Vector{Float64}(undef, len)
    @inbounds for i in 1:len
        val = vec[i]
        rolling_mean_sum += val
        if i <= lookback
            rolling_mean = rolling_mean_sum / i
        else
            rolling_mean_sum -= vec[i-lookback]
            rolling_mean = rolling_mean_sum / lookback
        end

        rolling_mean_vec[i] = rolling_mean
    end
    return rolling_mean_vec
end

function calc_expanding_rolling_mean(vec::Vector{Union{Missing,T}}, lookback::Integer) where {T}

    len = size(vec, 1)

    rolling_mean = calc_expanding_rolling_mean(vec |> skipmissing |> collect, lookback)

    rolling_mean_dst = Vector{Union{Missing,Float64}}(undef, len)

    dest_idx = 1
    rolling_mean_idx = 1

    for i in eachindex(vec)
        if ismissing(vec[i])
            res = missing
        else
            res = rolling_mean[rolling_mean_idx]
            rolling_mean_idx += 1
        end
        @inbounds rolling_mean_dst[i] = res
    end

    return rolling_mean_dst
end

function ffill!(v, init=zero(eltype(v)))
    accumulate!((n0, n1) -> ismissing(n1) ? n0 : n1, v, v, init=init)
end

function calc_vwap(df, price_col="close", volume_col="volume")
    return cumsum(df[!, price_col] .* df[!, volume_col]) ./ cumsum(df[!, volume_col])
end

function count_flips(vec, flip_point=0.0)
    return cum_count_flips(vec, flip_point)[end]
end

function cum_count_flips(vec, flip_point=0.0, buffer_thresh=0.0)
    len = size(vec, 1)
    flip_count::Int = 0
    cum_flips_vec = Vector{Int}(undef, len)
    prior_zone = 0
    @inbounds for i in 1:len
        if prior_zone == 1
            if vec[i] < flip_point - buffer_thresh
                flip_count += 1
            end
        elseif prior_zone == -1
            if vec[i] > flip_point + buffer_thresh
                flip_count += 1
            end
        end

        if vec[i] > flip_point + buffer_thresh
            prior_zone = 1
        elseif vec[i] < flip_point - buffer_thresh
            prior_zone = -1
        else
            prior_zone = 0
        end

        cum_flips_vec[i] = flip_count
    end
    return cum_flips_vec
end

mkt_open_time = Time(9, 30)
mkt_close_time = Time(16, 0)

function calc_had_halt_signal(df)
    len = size(df, 1)

    halt_signal_vec = Vector{Bool}(undef, len)


    @inbounds for i in 1:len

        if i < 5
            halt_signal_vec[i] = false
            continue
        end

        curr_time = Time(df[i, :ts])

        if curr_time < mkt_open_time || curr_time >= mkt_close_time
            halt_signal_vec[i] = false
            continue
        end

        time_diff = curr_time - Time(df[i-1, :ts])
        if time_diff <= Minute(3)
            halt_signal_vec[i] = false
            continue
        end

        before_halt_5_min_pct_change_1 = abs(df[i-1, :high] / df[i-6, :low] - 1)
        before_halt_5_min_pct_change_2 = abs(df[i-1, :low] / df[i-6, :high] - 1)
        before_halt_5_min_pct_change =
            max(before_halt_5_min_pct_change_1, before_halt_5_min_pct_change_2)
        if before_halt_5_min_pct_change < 0.10
            halt_signal_vec[i] = false
            continue
        end

        halt_signal_vec[i] = true
    end

    return halt_signal_vec
end

function calc_mae_mfe_pf(high_vec, low_vec, close_vec, start_idx, end_idx, entry_price)
    highest_high::Float64 = 0.0
    lowest_low::Float64 = typemax(Float64)
    pf_num = 0.0
    pf_denom = 0.0
    @inbounds for i in start_idx+1:end_idx
        if low_vec[i] < lowest_low
            lowest_low = low_vec[i]
        end
        if high_vec[i] > highest_high
            highest_high = high_vec[i]
        end

        curr_ret = close_vec[i] / close_vec[i-1] - 1
        if curr_ret < 0
            pf_num -= curr_ret
        elseif curr_ret > 0
            pf_denom += curr_ret
        end
    end

    mae = entry_price / highest_high - 1.0
    mfe = entry_price / lowest_low - 1.0

    pf = pf_num / pf_denom

    return (
        mae=mae,
        mfe=mfe,
        pf=pf
    )
end

end