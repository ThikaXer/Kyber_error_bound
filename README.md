# Supplementary material: Comparison of Kyber's error bounds

Code in folder `Delta_by_Ducas` is taken from https://github.com/pq-crystals/security-estimates and was developed by Leo Ducas.

Two main experimants are implemented:

1) An exact calculation of the original error bound delta (calculated by the script from Ducas) and an alternative error bound delta' and the actual correctness error. For very small parameter sets, these values can be computed exactly.
   This can be found in the file `Concrete_delta.ipynb`.

2) A stochastic estimate for slightly larger (still quite small) parameter sets unsing Monte-Carlo samplings. Again, we compare the original delta (calculated by the script from Ducas), the alternative bound delta' and the estimated correctness error.
   This can be found in the file `Compare_deltas_big.ipynb`. The Monte-Carlo smaplings for the calculation of delta' can be found in the file `sample_delta_1.py`.
   Outcomes of the experiments have been recorded in the folders `data` and `figs`.

In these small parameter sets, the inequality between delta and the actual correctness error is violated.
