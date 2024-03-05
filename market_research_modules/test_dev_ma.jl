include("include.jl")
include("ma_system.jl")
import Random
import Printf

import .MaSystems

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

    stoc_bias = StocBiases.StocBias(nprices - max_lookback)

    criter = MaSystems.build_ma_crossover_criter_func(max_lookback, log_price_vec)

    n_ints = 1

    pop_size = 10 # 100
    over_init = 10 # 10_000
    max_evals = 10_000_000
    max_generations = 5

    diff_ev_results = DiffEvs.diff_ev(
        criter, n_ints, pop_size, over_init, mintrades, max_evals, 300, 0.2, 0.2, 0.3, low_bounds, high_bounds, stoc_bias,
        max_generations;
        print_progress=true
    )

    println(
        Printf.@sprintf("\nBest performance = %.4lf param vals:  ", diff_ev_results.best_value)
        *
        join((Printf.@sprintf("%.4lf", diff_ev_results.best_params[i]) for i in size(diff_ev_results.best_params, 1)), "  ")
    )

    stoc_bias_res::StocBiases.StocBiasRes = StocBiases.compute_stoc(stoc_bias)

    println("\nVery rough estimates from differential evolution initialization...")
    Printf.@sprintf("  In-sample mean = %.4lf", stoc_bias_res.IS_return) |> println
    Printf.@sprintf("  Out-of-sample mean = %.4lf", stoc_bias_res.OOS_return) |> println
    Printf.@sprintf("  Bias = %.4lf", stoc_bias_res.bias) |> println
    Printf.@sprintf("  Expected = %.4lf\n", diff_ev_results.best_value - stoc_bias_res.bias) |> println

    # sensitivity ( criter , 4 , 1 , 30 , 80 , mintrades , params , low_bounds , high_bounds ) ;

end

# function test_test_system()

#     nprices = 10_000
#     log_price_vec = Vector{Float64}(undef, nprices)
#     generate_random_log_price_series(log_price_vec)

#     max_lookback = 100
#     max_thresh = 100.0

#     low_bounds = [2, 0.01, 0.0, 0.0]
#     high_bounds = [max_lookback, 99.0, max_thresh, max_thresh]

#     mintrades::Integer = 20

#     stoc_bias = StocBiases.StocBias(nprices - max_lookback)

#     params = [110, 99, max_thresh, max_thresh, 0.0]
#     params_population = Matrix{Float64}(undef, 1, size(params, 1))
#     params_population[1, :] = params

#     criter = build_ma_crossover_criter_func(max_lookback, log_price_vec)

#     DiffEvs.ensure_legal(1, low_bounds, high_bounds, @view params_population[1, :])

#     OOS_result = criter(params_population[1, :], mintrades, stoc_bias)

#     println("OOS results: $OOS_result")
# end

main()
# test_test_system()