# Mathematical notes

Equation-level transcription of the models used by `causal-pred`, together
with the approximations we make and the exact formulae implemented in the
Python and Rust code. Section citations follow each derivation.

## 1. MrDAG posterior over edge inclusion

MrDAG (Zuber et al. 2025) defines a joint posterior over directed acyclic
graphs between exposures using per-exposure Mendelian-randomisation summary
statistics. Let `theta_{jk}` denote the causal effect of exposure `j` on
exposure `k`, let `beta_{jk}` be the IVW point estimate with standard
error `se_{jk}`, and let `gamma_{jk} \in {0,1}` indicate edge inclusion in
the DAG. The model factorises as

```
p(G, theta | beta, se)
  \propto p(G)
  * prod_{(j,k): gamma_{jk}=1} N(beta_{jk} | theta_{jk}, se_{jk}^2)
                                N(theta_{jk} | 0, tau^2)
  * prod_{(j,k): gamma_{jk}=0} N(beta_{jk} | 0, se_{jk}^2 + omega^2).
```

Our approximation collapses `theta` analytically and reports a per-edge
Bayes factor `BF_{jk} = p(beta_{jk} | gamma=1) / p(beta_{jk} | gamma=0)`,
from which the edge-inclusion posterior marginal is

```
pi_{jk} = BF_{jk} * p_prior / (BF_{jk} * p_prior + (1 - p_prior)).
```

The implementation lives in `src/causal_pred/mrdag/pipeline.py` and the
Bayes factor is the Wakefield 2009 spike-vs-practical-null construction
(see section 5 below).

Citations: Zuber V, Gkatzionis A, Jones H, Burgess S. (2025) *MrDAG:
Bayesian causal discovery with Mendelian randomization*. Draper (1995).

## 2. DAGSLAM algorithm

DAGSLAM (Zhao & Jia 2025) is a greedy hill-climber in DAG space that
performs edge additions, deletions, and reversals while enforcing
acyclicity. At each iteration it evaluates

```
Delta = score(G + e) - score(G)
      + log pi_e / (1 - pi_e)        if adding edge e
      - log pi_e / (1 - pi_e)        if deleting edge e
```

and accepts the move of largest positive `Delta`. Acyclicity is checked by
a topological-sort test on the candidate adjacency. The algorithm returns
a warm-start DAG `G_0` consumed by the structure MCMC.

Citations: Zhao P, Jia B. (2025) *DAGSLAM: scalable warm-starts for
Bayesian structure learning*. Pearl (2009).

## 3. BGe score (Gaussian marginal likelihood)

For a Gaussian node `j` with parent set `Pa(j)` and `n` observations, the
BGe marginal log-likelihood (Kuipers-Moffa-Heckerman 2014, derived from
Geiger-Heckerman 2002) is

```
log p(X_j | X_{Pa(j)})
  = c(n, |Pa(j)|, alpha_w)
    + (alpha_w + n - |Pa(j)| - 1)/2 * log det(T_0 + S_{N,Pa(j)})
    - (alpha_w + n - |Pa(j)|)/2 * log det(T_0 + S_{N, Pa(j) U {j}})
```

where `T_0` is the prior scale matrix, `S_N` is the empirical scatter
matrix, `alpha_w` is the Wishart hyperparameter, and `c(...)` is a
normalisation term involving multivariate gamma functions. Our
implementation follows the numerically stable form in section 3 of
Kuipers-Moffa-Heckerman 2014, using Cholesky updates when a parent is
added to avoid recomputing `det` from scratch.

Citations: Kuipers J, Moffa G, Heckerman D. (2014). *Addendum on the
scoring of Gaussian directed acyclic graphical models*. Annals of
Statistics 42(4). Geiger D, Heckerman D. (2002).

## 4. Laplace-logistic marginal for binary nodes

For a binary node `j` with parents `Pa(j)`, we approximate the marginal
likelihood of a logistic regression by a Laplace expansion around the MAP
estimate `beta_hat`:

```
log p(X_j | X_{Pa(j)})
  \approx log p(X_j | X_{Pa(j)}, beta_hat)
        + log p(beta_hat)
        + (d/2) log(2 pi) - (1/2) log det(H(beta_hat))
```

where `H(beta_hat) = -grad^2 log p(X_j | X_{Pa(j)}, beta) |_{beta_hat}` is
the observed information and `d = |Pa(j)| + 1` including the intercept.
We use a Ridge prior `beta ~ N(0, lambda^{-1} I)` with `lambda = 1` so that
the Laplace approximation is well-defined even when `X_{Pa(j)}` is
rank-deficient.

Citations: Bishop (2006) *Pattern Recognition and Machine Learning*,
chapter 4.4. Murphy (2012) chapter 8.4.

## 5. Wakefield 2009 Bayes factor

The spike-vs-practical-null Bayes factor used in `mrdag/pipeline.py`
contrasts an alternative `theta ~ N(0, W)` against a practical null
`theta ~ N(0, omega^2)` with `omega^2 << W`. For a point estimate
`beta_hat` with standard error `se`,

```
BF = sqrt((se^2 + omega^2) / (se^2 + W))
   * exp( (1/2) * beta_hat^2 * (W - omega^2)
          / ((se^2 + W)*(se^2 + omega^2)) ).
```

We use `W = 0.21^2` (prior on plausible log-odds-ratio effects) and
`omega^2 = 0.05^2` (practical-null slab) per the Wakefield 2009
recommendation for complex-trait GWAS effect sizes.

Citations: Wakefield J. (2009). *Bayes factors for genome-wide association
studies: comparison with P-values*. Genet Epidemiol 33(1).

## 6. Metropolis-Hastings with Giudici-Castelo neighbourhood

Structure MCMC proposes an edge addition, deletion, or reversal from the
current DAG `G` yielding `G'`. The Giudici-Castelo 2003 correction
balances the forward and reverse proposal probabilities by the size of
each graph's neighbourhood `|N(G)|, |N(G')|`:

```
alpha(G -> G')
  = min(1,
        (score(G') * prior(G') * |N(G)|)
        / (score(G) * prior(G) * |N(G')|)).
```

For reversals of a shared edge, both neighbourhood sizes change and the
ratio must be recomputed by re-enumerating the valid moves from `G'`.
Our implementation caches the per-node score deltas so only the two
affected nodes are re-scored on each accepted move.

Citations: Giudici P, Castelo R. (2003). *Improving Markov chain Monte
Carlo model search for data mining*. Machine Learning 50(1-2). Madigan &
York (1995).

## 7. Distributional survival GAM

Following Rigby-Stasinopoulos 2005 (GAMLSS), we parameterise the survival
time `T` conditional on covariates `x` by a distributional form, eg

```
T | x ~ Weibull(mu(x), sigma(x))
log mu(x)    = sum_k f_k(x_k)
log sigma(x) = sum_k g_k(x_k),
```

with each `f_k, g_k` a P-spline (Eilers-Marx 1996) basis expansion. The
posterior over spline coefficients is sampled with NUTS (Hoffman-Gelman
2014) after an initial REML fit for the smoothing parameters (Wood 2011).

Citations: Rigby RA, Stasinopoulos DM. (2005). *Generalized additive
models for location, scale and shape*. Eilers PHC, Marx BD. (1996).
Hoffman MD, Gelman A. (2014). Wood SN. (2011).

## 8. Bayesian model averaging over parent sets

Let `\mathcal{G}` be the posterior set of DAGs from the structure MCMC.
For a downstream functional `f(G, theta)` (eg individual survival curve),
the posterior predictive is

```
E[f | data] = sum_{G in \mathcal{G}} w_G * E[f | G, data],
```

with weights `w_G` equal to the MCMC visitation frequency. For parent-set
functionals we collapse further to a per-node weight over parent sets,
which is numerically stable when the same parent set appears under many
DAG orderings (Madigan-York 1995).

Citations: Madigan D, York J. (1995). *Bayesian graphical models for
discrete data*. Int Stat Rev 63(2).

## 9. Brier decomposition (Murphy 1973)

For a probability forecast `p_i` of a binary event `y_i`, the mean Brier
score decomposes as

```
BS = (1/n) sum_i (p_i - y_i)^2
   = reliability - resolution + uncertainty.
```

With predictions binned into `K` groups of size `n_k`, mean forecast
`p_bar_k`, and observed frequency `o_bar_k`,

```
reliability = (1/n) sum_k n_k * (p_bar_k - o_bar_k)^2,
resolution  = (1/n) sum_k n_k * (o_bar_k - o_bar)^2,
uncertainty = o_bar * (1 - o_bar).
```

We report reliability and resolution alongside the integrated Brier score
(Graf 1999) in `validation/metrics.py`.

Citations: Murphy AH. (1973). *A new vector partition of the probability
score*. J Appl Meteorol 12(4). Brier GW. (1950). Graf E. (1999).

## 10. Time-dependent AUC (Heagerty 2000 / Uno 2007)

The incident/dynamic time-dependent ROC at horizon `t` treats cases as
subjects with event time `T_i \le t` and controls as subjects with
`T_i > t`. Heagerty 2000 defines

```
AUC(t) = P(risk_i > risk_j | T_i \le t, T_j > t),
```

estimated by kernel smoothing with weights determined by the
Kaplan-Meier estimator of the censoring distribution. Uno 2007 proposed
an IPCW estimator that is consistent under random censoring:

```
AUC_hat(t)
  = sum_{i,j} I(T_i \le t) I(T_j > t) I(risk_i > risk_j) * w_i w_j
    / sum_{i,j} I(T_i \le t) I(T_j > t) * w_i w_j,
```

with `w_i = delta_i / G_hat(T_i-)`, `G_hat` the Kaplan-Meier of the
censoring time. We report Uno's estimator at horizons 5, 10 and 15 years
and the integrated time-dependent AUC.

Citations: Heagerty PJ, Lumley T, Pepe MS. (2000). *Time-dependent ROC
curves for censored survival data and a diagnostic marker*. Biometrics
56(2). Uno H, Cai T, Tian L, Wei LJ. (2007).
