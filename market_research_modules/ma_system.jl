module MaSystems

import ..StocBiases
import ..DiffEvs

struct SystemResult
    n_trades::Int64
    cum_ret::Float64
end

function ma_crossover_system(
    max_lookback::Int64,    # Max lookback that will ever be used
    log_prices::Vector{Float64},
    long_term::Int64,       # Long-term lookback
    short_pct::Float64,     # Short-term lookback is this / 100 times long_term, 0-100
    short_thresh::Float64,  # Short threshold times 10_000
    long_thresh::Float64,   # Long threshold times 10_000
    returns_vec::Union{Vector{Float64},Nothing},
)::SystemResult

    ncases = size(log_prices, 1)

    short_term = floor(0.01 * short_pct * long_term)
    if short_term < 1
        short_term = 1
    end

    if short_term >= long_term
        short_term = long_term - 1
    end

    short_thresh /= 10000.0
    long_thresh /= 10000.0

    cum_ret = 0.0                          # Cumulate performance for this trial

    ntrades::Int64 = 0
    k::Int64 = 1                              # Will index returns

    for i in max_lookback:ncases-1    # Sum performance across history

        short_mean = 0.0                # Cumulates short-term lookback sum

        j::Int64 = i
        while j > i - short_term
            short_mean += log_prices[j]
            j -= 1
        end

        long_mean = short_mean          # Cumulates long-term lookback sum
        while j > i - long_term
            long_mean += log_prices[j]
            j -= 1
        end

        short_mean /= short_term
        long_mean /= long_term

        # We now have the short-term and long-term means ending at day i
        # Take our position and cumulate return

        change = short_mean / long_mean - 1.0   # Fractional difference in MA of log prices

        if (change > long_thresh)         # Long position
            ret = log_prices[i+1] - log_prices[i]
            ntrades += 1
        elseif (change < -short_thresh)  # Short position
            ret = log_prices[i] - log_prices[i+1]
            ntrades += 1
        else
            ret = 0.0
        end

        cum_ret += ret

        if (returns_vec !== nothing)
            returns_vec[k] = ret
            k += 1
        end

    end # For i, summing performance for this trial

    return SystemResult(ntrades, cum_ret)
end


function build_ma_crossover_criter_func(
    max_lookback,
    prices
)

    return function criter(
        params::AbstractArray{Float64,1},
        mintrades::Int64,
        stoc_bias::Union{StocBiases.StocBias,Nothing}=nothing
    )

        long_term = floor(Int64, params[1] + 1.e-10)
        short_pct = params[2]
        short_thresh = params[3]
        long_thresh = params[4]

        test_system_res::SystemResult = ma_crossover_system(
            max_lookback,
            prices,
            long_term,
            short_pct,
            short_thresh,
            long_thresh,
            (stoc_bias !== nothing) ? stoc_bias.returns : nothing)

        ret_val::Float64 = test_system_res.cum_ret

        if (stoc_bias !== nothing && ret_val > 0.0)
            StocBiases.process_stoc(stoc_bias)
        end

        if (test_system_res.n_trades >= mintrades)
            return ret_val
        else
            return -1.e20
        end
    end

end


function generate_random_log_price_series(
    price_vector::Vector{Float64}
)

    trend = 1

    price_vector[1] = 0.0

    for i in 2:size(price_vector, 1)
        if (i - 1) % 50 == 0   # Reverse the trend every 50 days
            trend = -trend
        end
        price_vector[i] = price_vector[i-1] + trend + DiffEvs.unifrand()
    end

end

end