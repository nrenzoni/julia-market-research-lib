module DiffEvs

using Random
import ..BrentMaxs
import ..StocBiases
import Printf

unifrand() = rand(Random.uniform(Float64))

function glob_max(
    low::Float64,                # Lower limit for search
    high::Float64,               # Upper limit
    npts::Int64,                 # Number of points to try
    log_space::Bool,            # Space by log?
    c_func::Function,            # Criterion function
    x1::Ref{Float64},
    y1::Ref{Float64},
    x2::Ref{Float64},
    y2::Ref{Float64},
    x3::Ref{Float64},
    y3::Ref{Float64}
)

    if npts < 0
        npts = -npts
        know_first_point = true
    else
        know_first_point = false
    end

    if log_space
        rate = exp(log(high / low) / (npts - 1))
    else
        rate = (high - low) / (npts - 1)
    end

    x = low

    previous = 0.0
    best_indvl = -1      # For proper critlim escape
    turned = false      # Must know if function improved

    for i in 1:npts

        if i > 1 || !know_first_point
            y = c_func(x)
        else
            y = y2[]
        end

        if i == 1 || y > y2[]   # Keep track of best here
            best_indvl = i
            x2[] = x
            y2[] = y
            y1[] = previous   # Function value to its left
            turned = false    # Flag that min is not yet bounded

        elseif i == best_indvl + 1  # Didn't improve so this point may
            y3[] = y                # be the right neighbor of the best
            turned = true          # Flag that min is bounded
        end

        previous = y              # Keep track for left neighbor of best

        if log_space
            x *= rate
        else
            x += rate
        end

        #=
        At this point we have a maximum (within low,high) at (x2,y2).
        Compute x1 and x3, its neighbors.
        We already know y1 and y3 (unless the maximum is at an endpoint!).
        =#

        if log_space
            x1[] = x2[] / rate
            x3[] = x2[] * rate
        else
            x1[] = x2[] - rate
            x3[] = x2[] + rate
        end

        #=
        Normally we would now be done.  However, the careless user may have
        given us a bad x range (low,high) for the global search.
        If the function was still improving at an endpoint, bail out the
        user by continuing the search.
        =#

        if !turned  # Must extend to the right (larger x)

            while true
                y3[] = c_func(x3[])

                if y3[] < y2[]  # If function decreased we are done
                    break
                end

                if (y1[] == y2[]
                    &&
                    y2[] == y3[]) # Give up if flat
                    break
                end

                x1[] = x2[]       # Shift all points
                y1[] = y2[]
                x2[] = x3[]
                y2[] = y3[]

                rate *= 3.0     # Step further each time
                if log_space   # And advance to new frontier
                    x3[] *= rate
                else
                    x3[] += rate
                end
            end

        elseif best_indvl == 1   # Must extend to the left (smaller x)

            while true

                y1[] = c_func(x1[])

                if y1[] < y2[]   # If function decreased we are done
                    break
                end

                if (y1[] == y2[]
                    &&
                    y2[] == y3[]) # Give up if flat
                    break
                end

                x3[] = x2[]       # Shift all points
                y3[] = y2[]
                x2[] = x1[]
                y2[] = y1[]

                rate *= 3.0     # Step further each time

                if log_space   # And advance to new frontier
                    x1[] /= rate
                else
                    x1[] -= rate
                end

            end
        end
    end
end


function validate_legal(
    low_bounds::Vector{Float64},
    high_bounds::Vector{Float64},
    params1::SubArray{Float64,1,TParent,TL},
    loc::Int
) where {TParent,TL}

    nvars = size(low_bounds, 1)

    for i in 1:nvars

        if (params1[i] > high_bounds[i])
            throw("param $i ($(params1[i])) > high ($(high_bounds[i])) at loc $loc")
        end

        if (params1[i] < low_bounds[i])
            throw("param $i ($(params1[i])) < low ($(low_bounds[i])) at loc $loc")
        end

    end

end

function ensure_legal(
    n_ints::Integer,
    low_bounds::Vector{Float64},
    high_bounds::Vector{Float64},
    params1::SubArray{Float64,1,TParent,TL},
) where {TParent,TL}

    nvars = size(low_bounds, 1)

    penalty::Float64 = 0.0

    for i in 1:nvars

        if (i <= n_ints)
            params1[i] = round(params1[i])
        end

        if (params1[i] > high_bounds[i])
            penalty += 1.e10 * (params1[i] - high_bounds[i])
            params1[i] = high_bounds[i]
        end

        if (params1[i] < low_bounds[i])
            penalty += 1.e10 * (low_bounds[i] - params1[i])
            params1[i] = low_bounds[i]
        end

    end

    return penalty
end

struct DiffEvResults
    best_params::Vector{Float64}
    best_value::Float64
end

function diff_ev(
    criter,                      # Crit function maximized
    n_ints::Integer,             # Number of first variables that are integers
    popsize::Integer,            # Population size
    over_init::Integer,          # Overinitialization for initial population
    min_trades::Integer,         # Minimum number of trades for candidate system
    max_evals::Integer,          # For safety, max number of failed initial performance evaluations should be very large
    max_bad_gen::Integer,        # Max number of contiguous generations with no improvement of best
    mutate_dev::Float64,         # Deviation for differential mutation
    pcross::Float64,             # Probability of crossover
    pclimb::Float64,             # Probability of taking a hill-climbing step, can be zero
    low_bounds::Vector,          # Lower bounds for parameters
    high_bounds::Vector,         # And upper
    stoc_bias::Union{StocBiases.StocBias,Nothing}=nothing, # Optional and unrelated to differential evolution
    max_generations::Int=10_000_000
    ;
    print_progress::Bool=false  # Print progress to screen?
)::DiffEvResults

    popsize <= 4 && throw("Popsize must be > 4")

    if size(low_bounds, 1) != size(high_bounds, 1)
        throw("size(low_bounds) ($(size(low_bounds, 1))) != size(high_bounds) ($(size(high_bounds, 1)))")
    end

    nvars = size(low_bounds, 1)

    k::Int64 = 0

    dim = nvars + 1

    pop1 = Matrix{Float64}(undef, popsize, dim)
    # fill!(pop1, -100.)
    pop2 = Matrix{Float64}(undef, popsize, dim)
    best = Vector{Float64}(undef, dim)

    function finish()
        return DiffEvResults(best[begin:end-1], best[end])
    end

    failures = 0                           # Counts consecutive failures
    n_evals::Int64 = 0                            # Counts evaluations for catastrophe escape

    grand_best::Float64 = 0.0
    worstf::Float64 = 0.0
    avgf::Float64 = 0.0

    child_ind = 0
    while child_ind < popsize + over_init
        child_ind += 1

        if child_ind <= popsize                  # If we are in pop1
            pop = pop1
            pop_idx = child_ind                 # Point to the slot in pop1
        else                                # Use first slot in pop2 for work
            pop = pop2
            pop_idx = 1                   # Point to first slot in pop2
        end

        for i in 1:nvars                # For all variables
            val::Float64 = 0.0
            if i <= n_ints  # Is this an integer?
                val = low_bounds[i] + floor(unifrand() * (high_bounds[i] - low_bounds[i] + 1.0))
            else  # real
                val = low_bounds[i] + (unifrand() * (high_bounds[i] - low_bounds[i]))
            end

            if val < low_bounds[i]
                val = low_bounds[i]
            elseif val > high_bounds[i]
                val = high_bounds[i]
            end

            pop[pop_idx, i] = val
        end

        crit_value = criter(@view(pop[pop_idx, :]), min_trades, stoc_bias)
        pop[pop_idx, end] = crit_value           # Also save criterion after variables
        n_evals += 1                       # Count evaluations for emergency escape

        if child_ind == 1
            grand_best = worstf = crit_value
            avgf = 0.0
            copyto!(best, @view(pop[1, :])) # Best so far is first!
        end

        if crit_value <= 0.0   # If this individual is totally worthless
            if n_evals > max_evals  # Safety escape should ideally never happen
                println("max evals ($n_evals) reached, exiting")
                return finish()
            end
            child_ind -= 1            # retry this child
            failures += 1
            if failures >= 500   # This many in a row
                failures = 0
                min_trades = floor(Int64, min_trades * 9 / 10)
                if min_trades < 1
                    Printf.@printf("hit min trades of 1, maybe try with higher min_trades arg, exiting")
                    # min_trades = 1
                    return finish()
                end
            end
            continue
        else
            failures = 0
        end

        #=
        Maintain best, worst, and average
        These are strictly for user updates, and they have nothing to do with the algorithm.
        Well, we do keep the grand best throughout, as this is what is ultimately returned.
        =#

        if crit_value > grand_best    # Best ever
            copyto!(best, @view(pop[pop_idx, :]))
            grand_best = crit_value
        end

        if crit_value < worstf
            worstf = crit_value
        end

        avgf += crit_value

        if print_progress
            if child_ind <= popsize        # Before overinit we update average as each new trial is done
                avg = avgf / child_ind
            else                      # After overinit we examine only the population
                avg = avgf / popsize
            end

            println(
                Printf.@sprintf("%d: Val=%.4lf Best=%.4lf Worst=%.4lf Avg=%.4lf  (fail rate=%.1lf)",
                    child_ind, crit_value, grand_best, worstf, avg, n_evals / (child_ind + 1.0))
                *
                join((Printf.@sprintf(" %.4lf", pop[pop_idx, i]) for i in 1:nvars), " ")
            )
        end

        #=
        If we have finished pop1 and we are now into overinit, the latest
        candidate is in the first slot of pop2.  Search pop1 for the worst
        individual.  If the new candidate is better than the worst in pop1,
        replace the old with the new.
        We recompute the average within the original population.
        =#

        if child_ind > popsize       # If we finished pop1, now doing overinit
            avgf = 0.0
            idx_of_min::Int64 = 0
            for i in 1:popsize    # Search pop1 for the worst
                dtemp = pop1[i, end]
                avgf += dtemp
                if i == 1 || dtemp < worstf
                    idx_of_min = i  # todo: does pop1 need to be reference later on too?
                    worstf = dtemp
                end
            end # Searching pop1 for worst
            if crit_value > worstf   # If this is better than the worst, replace worst with it
                copyto!(@view(pop1[idx_of_min, :]), @view(pop[pop_idx, :]))
                avgf += crit_value - worstf   # Account for the substitution
            end

        end # If doing overinit

    end # For all individuals (population and overinit)

    #=
    We now have the initial population and also have completed overinitialization.
    Search the initial population to find the subscript of the best.
    This is to let us periodically tweak the best.
    =#

    for i in 1:size(pop1, 1)
        validate_legal(low_bounds, high_bounds, @view(pop1[i, :]), 5)
    end

    best_indvl::Int64 = 1
    crit_value = pop1[1, end]
    for child_ind in 2:popsize
        if pop1[child_ind, end] > crit_value
            crit_value = pop1[child_ind, end]
            best_indvl = child_ind
        end
    end

    #=
    --------------------------------------------------------------------------------

    This is the main loop.  For each generation, use old_gen for the parents
    and create the children in new_gen.  These flip between pop1 and pop2.
    'Repeats' counts the number of generations with no improvements.
    This allows automatic escape for batch runs.

    --------------------------------------------------------------------------------
    =#

    pop = pop1
    n_tweaked = 0

    old_gen = pop1        # This is the old, parent generation
    new_gen = pop2        # The children will be produced here
    bad_generations::Int64 = 0   # Counts contiguous generations with no improvement of best

    generation = 0

    while true
        generation += 1

        if generation > max_generations
            println("max generations ($generation) > max_generations ($max_generations) reached, exiting")
            return finish()
        end

        worstf = 1.e60
        avgf = 0.0
        improved = false       # Will flag if we improved in this generation

        for indvl in 1:popsize   # Generate all children

            #  Generate three different random numbers for parent2 and the differentials

            get_rand() = floor(Int32, unifrand() * popsize) + 1

            parent_to_mutate::Int64 = 0
            while true
                parent_to_mutate = get_rand()
                parent_to_mutate <= popsize && parent_to_mutate != indvl && break
            end

            diff1::Int64 = 0
            while true
                diff1 = get_rand()
                diff1 <= popsize && diff1 != indvl && diff1 != parent_to_mutate && break
            end

            diff2::Int64 = 0
            while true
                diff2 = get_rand()
                diff2 <= popsize && diff2 != indvl && diff2 != diff1 && diff2 != parent_to_mutate && break
            end


            #=
            Build the child in the destination array, even though it may have to be
            overwritten with parent1 if this child is not superior.

            We need to randomly pick a starting point in the parameter vector because
            when we get to the end if we have not used any parameters from the mutation
            vector we force one into the child.  We do not want that forced parameter
            to always be the last position in the vector!
            =#

            function get_rand_j()
                while true
                    j = floor(Int32, unifrand() * nvars) + 1
                    j <= nvars && return j   # Pick a starting parameter
                end
            end

            j = @inline get_rand_j()

            used_mutated_parameter = false

            for i in nvars:-1:1
                if (i == 1 && !used_mutated_parameter) || unifrand() < pcross
                    new_gen[indvl, j] = old_gen[parent_to_mutate, j] + mutate_dev * (old_gen[diff1, j] - old_gen[diff2, j])
                    used_mutated_parameter = true
                    # We mutated this variable
                else   # We did not mutate this variable, so copy old value
                    new_gen[indvl, j] = old_gen[indvl, j]
                end
                j = (j % nvars) + 1   # Rotate through all variables
            end

            #=
            For all parameters, the above operation may have pushed the value outside
            its legal limit.  For integer parameters, illegal values have
            almost certainly resulted.  Fix these problems.
            =#

            ensure_legal(n_ints, low_bounds, high_bounds, @view(new_gen[indvl, :]))
            # println(
            #     "post ensure_legal: " *
            #     join((Printf.@sprintf("%.4lf", new_gen[indvl, i]) for i in 1:nvars), " ")
            # )

            #=
            Mutation is complete.  Evaluate the performance of this child.
            If the child is better than parent1, keep it right here in the destination
            array where it was created.  (Put its criterion there too.)
            If it is inferior to parent1, move that parent and its criterion to the
            destination array.
            =#
            
            crit_value = criter(@view(new_gen[indvl, :]), min_trades)

            if crit_value > old_gen[indvl, end]   # If the child is better than parent1
                new_gen[indvl, end] = crit_value    # Get the child's value (The vars are already there)
                if crit_value > grand_best    # And update best so far
                    grand_best = crit_value
                    copyto!(best, @view(new_gen[indvl, :]))
                    best_indvl = indvl
                    n_tweaked = 0
                    improved = true    # Flag that the best improved in this generation
                end
            else                         # Else copy parent and its value
                copyto!(@view(new_gen[indvl, :]), @view(old_gen[indvl, :]))
                crit_value = old_gen[indvl, end]
            end

            #=
            If we are to randomly tweak (do a hill-climbing step), do it now.
            Note that it is rarely possible for this step to cause the 'worst' to
            get worse!  The glob_max routine may search an interval that does not
            have the current parameters as a trial point, and never find anything
            quite as good.  This should happen only very late in the game, and not
            have any bad consequences.

            We use n_tweaked to count how many times this particular 'grand best'
            has been tweaked.  It is incremented each time the grand best is tweaked,
            and it is reset to zero whenever we get a new grand best.

            In order to do this hill-climbing step, we must have:
                pclimb > 0.0 (The user is allowing hill climbing) AND
                    (This individual is the grand best AND we have not yet tweaked every variable) OR
                    We randomly tweak some variable in this individual
            =#

            if pclimb > 0.0 && ((indvl == best_indvl && n_tweaked < nvars) || (unifrand() < pclimb))
                if indvl == best_indvl           # Once each generation tweak the best
                    n_tweaked += 1              # But quit if done all vars
                    k = ((generation - 1) % nvars) + 1   # Cycle through all vars
                else                        # Randomly choose an individual
                    k = floor(unifrand() * nvars) + 1  # Which var to optimize
                    if k > nvars           # Safety only
                        k = nvars
                    end
                end

                # Handle integer parameters
                if k <= n_ints  # this parameter is an integer
                    ivar = ibase = floor(new_gen[indvl, k])
                    ilow = floor(Int64, low_bounds[k])
                    ihigh = floor(Int64, high_bounds[k])
                    success = false
                    if print_progress
                        Printf.@printf("Crit max of indvl %d (int param #%d) base val: %d, base crit val: %.6lf\n",
                            indvl, k, ibase, crit_value
                        )
                    end

                    # tweak int var up to find adjacent best
                    while ivar < ihigh
                        ivar += 1
                        new_gen[indvl, k] = ivar
                        test_val = criter(@view(new_gen[indvl, :]), min_trades)
                        if print_progress
                            Printf.@printf("  %d = %.6lf\n", ivar, test_val)
                        end
                        if test_val > crit_value
                            crit_value = test_val
                            ibase = ivar
                            success = true
                        else
                            new_gen[indvl, k] = ibase
                            break
                        end
                    end

                    # tweak int var down to find adjacent best
                    if !success
                        ivar = ibase
                        while ivar > ilow
                            ivar -= 1
                            new_gen[indvl, k] = ivar
                            test_val = criter(@view(new_gen[indvl, :]), min_trades)
                            if print_progress
                                Printf.@printf("  %d = %.6lf\n", ivar, test_val)
                            end
                            if test_val > crit_value
                                crit_value = test_val
                                ibase = ivar
                                success = true
                            else
                                new_gen[indvl, k] = ibase
                                break
                            end
                        end
                    end

                    if print_progress
                        if success
                            Printf.@printf("Climb Improvement at %.0lf = %.6lf\n", new_gen[indvl, k], crit_value)
                        else
                            Printf.@printf("No Climb Improvement at %.0lf = %.6lf\n", new_gen[indvl, k], crit_value)
                        end
                    end
                else  # This is a real parameter
                    non_tweaked_param = new_gen[indvl, k] # Preserve orig var
                    old_crit_value = crit_value

                    if print_progress
                        Printf.@printf("Criter max of indvl #%d (real var #%d), base val: %.5lf, crit val: %.6lf\n",
                            indvl, k, non_tweaked_param, crit_value
                        )
                    end

                    lower = non_tweaked_param - 0.1 * (high_bounds[k] - low_bounds[k])
                    upper = non_tweaked_param + 0.1 * (high_bounds[k] - low_bounds[k])
                    if lower < low_bounds[k]
                        lower = low_bounds[k]
                        upper = low_bounds[k] + 0.2 * (high_bounds[k] - low_bounds[k])
                    end
                    if upper > high_bounds[k]
                        upper = high_bounds[k]
                        lower = high_bounds[k] - 0.2 * (high_bounds[k] - low_bounds[k])
                    end

                    univariate_maximize = build_univariate_maximize(
                        @view(new_gen[indvl, :]),
                        k,
                        n_ints,
                        low_bounds,
                        high_bounds,
                        min_trades,
                        criter)

                    x1 = Ref{Float64}(0)
                    y1 = Ref{Float64}(0)          # Lower X value and function there
                    x2 = Ref{Float64}(0)
                    y2 = Ref{Float64}(0)          # Middle (best)
                    x3 = Ref{Float64}(0)
                    y3 = Ref{Float64}(0)          # And upper

                    glob_max(lower, upper, 7, false, univariate_maximize, x1, y1, x2, y2, x3, y3)
                    BrentMaxs.brentmax(5, 1.e-8, 1e-4, univariate_maximize, x1, x2, x3, y2[])
                    new_gen[indvl, k] = x2[]   # Optimized var value

                    ensure_legal(n_ints, low_bounds, high_bounds, @view(new_gen[indvl, :]))

                    crit_value = criter(@view(new_gen[indvl, :]), min_trades)

                    if crit_value > old_crit_value
                        new_gen[indvl, end] = crit_value
                        if print_progress
                            Printf.@printf("Climb Improvement at %.5lf = %.6lf\n", new_gen[indvl, k], crit_value)
                        end
                    else
                        new_gen[indvl, k] = non_tweaked_param     # Restore original value
                        crit_value = old_crit_value
                        if (print_progress)
                            Printf.@printf("No climb improvement at %.5lf = %.6lf\n", new_gen[indvl, k], crit_value)
                        end
                    end

                    if crit_value > grand_best       # Update best so far
                        grand_best = crit_value
                        copyto!(best, @view(new_gen[indvl, :]))
                        best_indvl = indvl
                        n_tweaked = 0
                        improved = true    # Flag that the best improved in this generation
                    end

                end # If optimizing real parameter

            end # If doing hill-climbing step

            if crit_value < worstf
                worstf = crit_value
            end

            avgf += crit_value

        end # Create all children in this generation

        if print_progress
            println(
                Printf.@sprintf("Gen %d Best=%.4lf Worst=%.4lf Avg=%.4lf ", generation, grand_best, worstf, avgf / popsize)
                *
                join((Printf.@sprintf("%.4lf", best[i]) for i in 1:nvars), " ")
            )
        end

        #=
        This generation is complete.  See if we must quit due to too many contiguous failures to improve.
        Reverse old_gen and new_gen in pop1 and pop2.
        =#

        if !improved
            bad_generations += 1
            if bad_generations > max_bad_gen
                break
            end
        else
            bad_generations = 0
        end

        if old_gen === pop1
            old_gen = pop2
            new_gen = pop1
        else
            old_gen = pop1
            new_gen = pop2
        end

    end # For all generations

    return finish()

end

@inline function build_univariate_maximize(
    params1,
    param_idx,
    n_ints,
    low_bounds,
    high_bounds,
    mintrades,
    crit
)
    return function univariate_maximize(
        param::Float64
    )

        penalty::Float64 = 0.0

        params1[param_idx] = param
        penalty = ensure_legal(n_ints, low_bounds, high_bounds, params1)
        return crit(params1, mintrades) - penalty
    end

end

end