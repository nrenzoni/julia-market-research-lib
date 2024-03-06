
# Brain Dump of Julia auxilary functions for researching marking trading

### Project Motivation

In my interactive Jupyter notebooks, I primarily use Python, but I find that it becomes sluggish when iterating over rows in dataframe-like structures. Julia, being compiled to machine code, offers orders of magnitude faster performance compared to Python. I employ Python for its convenience, and seamlessly transfer data to Julia for the computationally intensive tasks. Once the heavy lifting is done, I bring the processed data back to Python for further analysis.

## Current Functionality

**Differential Evolution Optimization**

* This functionality receives an objective function, its input parameters, and parameter metadata, the optimization routine returns the the parameter set which maximize the function.  

**Parameter Sensitivity Analysis**

* Once an objective function and the set of parameters maximizing it are determined, the parameter sensitivity analysis involves varying each parameter's values individually while keeping the others constant. The analysis routine yields a matrix where each row corresponds to a parameter, and each column represents a different value within the parameter's range, spanning from its minimum to maximum. The matrix contains the objective function values corresponding to each parameter variation.

**More to come hopefuly...**

---



#### Credit
* 90% of the implementation logic is inspired by Timothy Master's book [Testing and Tuning Market Trading Systems: Algorithms in C++](http://www.timothymasters.info/market-trading.html). I do plan to implement more of his code, as well as ideas I have taken from other places to use in my analysis.

---
---

# Examples

* parameter sensitivity analysis of a trading system variable called "take_profit_pct"

    ![download](https://github.com/nrenzoni/julia-market-research-lib/assets/31897391/415cbe9e-a88a-4e1e-b21a-d6b2f9b6812b)
    
