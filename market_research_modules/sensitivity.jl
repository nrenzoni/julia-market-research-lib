module Sensitivitys

import Printf

struct SensitivityResults
    # row_i: vector sensitivity curve for parameter_i
    value_matrix::Matrix{Float64}
    params_matrix::Matrix{Float64}
end

function sensitivity(
    criter,         # Crit function maximized
    nints::Int,     # Number of first variables that are integers
    min_trades::Int,
    best::AbstractArray{Float64,1},        # Optimal parameters
    low_bounds::AbstractArray{Float64,1},  # Lower bounds for parameters
    high_bounds::AbstractArray{Float64,1}, # And upper
    npoints::Int=30,                # Number of points at which to evaluate performance
    nresolution::Int=80,            # Number of max points in plot for max per param
    debug::Bool=false
)

    nvars = size(low_bounds, 1)

    vals = Vector{Float64}(undef, npoints)
    params = Vector{Float64}(undef, nvars)

    vals_matrix = Matrix{Float64}(undef, nvars, npoints)
    params_matrix = Matrix{Float64}(undef, nvars, npoints)

    ival::Int = 0
    rval::Float64 = 0.0

    for ivar in 1:nvars

        # might be bug, might need to reset params after each permutation
        copyto!(params, best)

        maxval::Float64 = floatmin(Float64)

        if ivar <= nints  # integer parameters

            debug && Printf.@printf("\n\nSensitivity curve for integer parameter %d (optimum=%d)",
                ivar, floor(best[ivar] + 1.e-10))

            label_frac = (high_bounds[ivar] - low_bounds[ivar] + 0.99999999) / (npoints - 1)

            for ipoint in 1:npoints
                ival = floor(low_bounds[ivar] + (ipoint - 1) * label_frac)
                params_matrix[ivar, ipoint] = params[ivar] = ival
                vals_matrix[ivar, ipoint] = criter(params, min_trades)
                if vals_matrix[ivar, ipoint] > maxval
                    maxval = vals_matrix[ivar, ipoint]
                end
            end

            hist_frac = (nresolution + 0.9999999) / maxval
            debug && for ipoint in 1:npoints
                ival = floor(low_bounds[ivar] + (ipoint - 1) * label_frac)
                Printf.@printf("\n%6d|", ival)
                k = floor(vals_matrix[ivar, ipoint] * hist_frac)
                println("*"^k)
            end

        else # real parameters

            debug && Printf.@printf("\n\nSensitivity curve for real parameter %d (optimum=%.4lf)\n",
                ivar, best[ivar])

            label_frac = (high_bounds[ivar] - low_bounds[ivar]) / (npoints - 1)

            for ipoint in 1:npoints
                rval = low_bounds[ivar] + (ipoint - 1) * label_frac
                params_matrix[ivar, ipoint] = params[ivar] = rval
                vals_matrix[ivar, ipoint] = vals[ipoint] = criter(params, min_trades)
                if vals[ipoint] > maxval
                    maxval = vals[ipoint]
                end
            end

            hist_frac = (nresolution + 0.9999999) / maxval
            debug && for ipoint in 1:npoints
                rval = low_bounds[ivar] + (ipoint - 1) * label_frac
                Printf.@printf("\n%10.3lf|", rval)
                k = floor(vals[ipoint] * hist_frac)
                println("*"^k)
            end
        end

    end

    return SensitivityResults(vals_matrix, params_matrix)

end

function sensitivity2(
    criter,         # Crit function maximized
    nints::Int,     # Number of first variables that are integers
    min_trades::Int,
    best::AbstractArray{Float64,1},        # Optimal parameters
    low_bounds::AbstractArray{Float64,1},  # Lower bounds for parameters
    high_bounds::AbstractArray{Float64,1}, # And upper
    npoints::Int=30,                # Number of points at which to evaluate performance
    nresolution::Int=80            # Number of max points in plot for max per param
    ;
    debug::Bool=false
)

    @inbounds for i in eachindex(best)
        if best[i] < low_bounds[i] || best[i] > high_bounds[i]
            throw("Parameter #$(i) is out of range (value $(best[i]))")
        end
    end

    nvars = size(low_bounds, 1)

    vals = Vector{Float64}(undef, npoints)
    params = Vector{Float64}(undef, nvars)

    vals_matrix = Matrix{Float64}(undef, nvars, npoints)
    params_matrix = Matrix{Float64}(undef, nvars, npoints)

    ival::Int = 0
    rval::Float64 = 0.0
    k::Int64 = 0

    copyto!(params, best)

    for ivar in 1:nvars

        maxval::Float64 = floatmin(Float64)

        if ivar <= nints  # integer parameters

            debug && Printf.@printf("\n\nSensitivity curve for integer parameter %d (optimum=%d)",
                ivar, floor(best[ivar] + 1.e-10))

            label_frac = (high_bounds[ivar] - low_bounds[ivar] + 0.99999999) / (npoints - 1)

            for ipoint in 1:npoints
                old_val = params[ivar]
                ival = floor(low_bounds[ivar] + (ipoint - 1) * label_frac)
                params_matrix[ivar, ipoint] = params[ivar] = ival
                vals_matrix[ivar, ipoint] = criter(params, min_trades)
                if vals_matrix[ivar, ipoint] > maxval
                    maxval = vals_matrix[ivar, ipoint]
                end
                params[ivar] = old_val
            end

            hist_frac = (nresolution + 0.9999999) / maxval
            debug && for ipoint in 1:npoints
                ival = floor(low_bounds[ivar] + (ipoint - 1) * label_frac)
                Printf.@printf("%6d|", ival)
                k = max(floor(vals_matrix[ivar, ipoint] * hist_frac), 0)
                println("*"^k)
            end

        else # real parameters

            debug && Printf.@printf("\n\nSensitivity curve for real parameter %d (optimum=%.4lf)\n",
                ivar, best[ivar])

            label_frac = (high_bounds[ivar] - low_bounds[ivar]) / (npoints - 1)

            for ipoint in 1:npoints
                old_val = params[ivar]
                rval = low_bounds[ivar] + (ipoint - 1) * label_frac
                params_matrix[ivar, ipoint] = rval
                params[ivar] = rval
                vals_matrix[ivar, ipoint] = vals[ipoint] = criter(params, min_trades)
                if vals[ipoint] > maxval
                    maxval = vals[ipoint]
                end
                params[ivar] = old_val
            end

            hist_frac = (nresolution + 0.9999999) / maxval
            debug && for ipoint in 1:npoints
                rval = low_bounds[ivar] + (ipoint - 1) * label_frac
                Printf.@printf("%10.3lf|", rval)
                k = floor(Int64, vals[ipoint] * hist_frac)
                println("*"^k)
            end
        end

    end

    return SensitivityResults(vals_matrix, params_matrix)

end


end