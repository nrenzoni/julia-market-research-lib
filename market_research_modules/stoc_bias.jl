module StocBiases

# export StocBias, process_stoc, compute_stoc

mutable struct StocBias
    nreturns::Int64             # Number of returns
    got_first_case::Bool        # Have we processed the first case (set of returns)?
    returns::Vector{Float64}    # Returns for currently processed case
    IS_best::Vector{Float64}    # In-sample best total return
    OOS::Vector{Float64}        # Corresponding out-of-sample return

    StocBias(nc::Int64) = new(nc, false, Vector{Float64}(undef, nc), Vector{Float64}(undef, nc), Vector{Float64}(undef, nc))
end

function process_stoc(
    stoc_bias::StocBias
)
    total = 0.0
    for i in 1:stoc_bias.nreturns
        total += stoc_bias.returns[i]
    end

    # Initialize if this is the first call

    if !stoc_bias.got_first_case
        stoc_bias.got_first_case = true
        for i in 1:stoc_bias.nreturns
            this_x = stoc_bias.returns[i]
            stoc_bias.IS_best[i] = total - this_x
            stoc_bias.OOS[i] = this_x
        end

    else  # Keep track of best if this is a subsequent call
        for i in 1:stoc_bias.nreturns
            this_x = stoc_bias.returns[i]
            if total - this_x > stoc_bias.IS_best[i]
                stoc_bias.IS_best[i] = total - this_x
                stoc_bias.OOS[i] = this_x
            end
        end
    end

end

struct StocBiasRes
    IS_return::Float64
    OOS_return::Float64
    bias::Float64
end

function compute_stoc(
    stoc_bias::StocBias)::StocBiasRes

    IS_return = OOS_return = 0.0

    for i in 1:stoc_bias.nreturns
        IS_return += stoc_bias.IS_best[i]
        OOS_return += stoc_bias.OOS[i]
    end

    IS_return /= (stoc_bias.nreturns - 1)  # Each IS_best is the sum of nreturns-1 returns

    return StocBiasRes(IS_return, OOS_return, IS_return - OOS_return)
end

end  # module