include("include.jl")

import .Sensitivitys
import .MaSystems
import Random

function main()

    nprices = 1_000
    log_price_vec = Vector{Float64}(undef, nprices)

    Random.seed!(1234)

    MaSystems.generate_random_log_price_series(log_price_vec)

    max_lookback = 100
    max_thresh = 100.0

    low_bounds = [2, 0.01, 0.0, 0.0]
    high_bounds = [max_lookback, 99.0, max_thresh, max_thresh]

    mintrades::Integer = 20

    # stoc_bias = StocBiases.StocBias(nprices - max_lookback)

    criter = MaSystems.build_ma_crossover_criter_func(max_lookback, log_price_vec)

    params = [4.0, 24.7384, 0.0, 844.9521]

    Sensitivitys.sensitivity(criter, 1, mintrades, params, low_bounds, high_bounds)

end

main()