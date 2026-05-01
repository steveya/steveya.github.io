# Master Plan: From Mean-Reversion Blog Notes to a Publishable Paper

**Working paper title:**
**Following Mean Reversion with an Exponentially Weighted Fair-Value Estimate: Analytical Results for a Gaussian Model**

**Purpose of this document:**
This is the Codex handoff plan. It is meant to be detailed enough that the paper, blog posts, notebooks, code, figures, and LaTeX document can be built without re-deriving the mathematics in every working session.

**Strategic pivot:**
Discard the current draft Part 5 and Part 6 notebooks as main-line paper material. Keep them only as scratch/reference. The new Part 5 begins from scratch by discretizing the OU mean-reversion model, introducing an exponentially weighted moving-average (EWM) fair-value estimate, and showing that the resulting P&L can be expressed as a Gaussian quadratic form. This aligns the project with the Grebenkov-Serror EMA momentum paper, but for a value/mean-reversion strategy.

---

## 0. Target Paper Architecture

The final paper should mirror the analytical completeness of the EMA trend-following paper while making the opposite strategic object central: **mean reversion/value trading instead of momentum/trend following**.

### Proposed paper sections

1. **Introduction**
   - Motivation: value/mean-reversion strategies require a fair-value estimate.
   - Problem: fair value is estimated, and estimation affects P&L distribution, turnover, transaction costs, and optimal timescale.
   - Contribution: discrete Gaussian OU framework, EWM fair-value estimator, quadratic-form P&L distribution, transaction-cost optimality.

2. **Discrete Gaussian OU Mispricing Model**
   - Exact discretization of OU.
   - Return autocovariance.
   - Comparison with positively autocorrelated trend-following models.

3. **EWM Fair-Value Estimate and Mean-Reversion Signal**
   - EWM fair-value estimator.
   - Estimated mispricing as an exponentially weighted filter of past returns.
   - Signal timing and no-lookahead convention.

4. **P&L as a Gaussian Quadratic Form**
   - Matrix formulation.
   - Characteristic function.
   - Cumulants.
   - Numerical distribution via eigenvalues and Fourier inversion.

5. **Distribution, Skewness, Kurtosis, and Quantiles**
   - Incremental P&L distribution.
   - Cumulative P&L distribution.
   - Left and right tails.
   - Comparison with momentum/trend-following asymmetry.

6. **Transaction Costs and Cost-Aware Trading Rules**
   - Expected turnover.
   - Net risk-adjusted P&L.
   - Optimal EWM half-life under costs.
   - No-trade bands and cost hurdles.
   - Position caps and left-tail controls.
   - Sensitivity to mean-reversion speed and cost level.

7. **Calibration Robustness and Parameter Estimation Error**
   - OU/AR(1) MLE.
   - Mean-estimation error as fair-value bias.
   - Speed-estimation error and cost-adjusted Sharpe overconfidence.
   - Robust half-life, band, and leverage selection.

8. **Moving Fair Value and Empirical Illustration**
   - Latent moving fair-value extension.
   - Synthetic calibrated OU.
   - Optional pairs/spread data.
   - Theory vs simulation vs empirical diagnostics.
   - Policy comparison under costs and tail constraints.

9. **Discussion**
   - What survives beyond Gaussian OU?
   - Limitations: true fair value, moving fundamentals, jumps, heavy tails, no-trade bands.
   - Future work.

10. **Appendices**
    - Proofs.
    - Matrix identities.
    - Numerical methods.
    - Optional singular-control or no-transaction-region derivations.
    - Optional moving-fair-value extension.

---

# Part 5: Discretizing the OU Mean-Reversion Strategy

## Blog title

**Discretizing the Mean-Reversion Strategy: EWM Fair Value, Return Autocovariance, and Gross P&L**

## Paper role

This post becomes the foundation for Sections 2 and 3 of the paper.

## Main message

The continuous-time model is useful for intuition, but proportional transaction costs make continuous rebalancing ill-posed. To study turnover, transaction costs, and P&L distributions, we discretize the OU mispricing process. Once discretized, an EWM fair-value estimate turns the mean-reversion signal into a linear filter of past returns, allowing the P&L to be written as a Gaussian quadratic form.

---

## 5.1 Model setup

Let the true fair value be constant and normalized to zero:

\[
v_t \equiv 0.
\]

Let the log price equal the mispricing:

\[
p_t = X_t.
\]

The continuous-time reference model is

\[
dX_t = -\theta X_t\,dt + \sigma\,dW_t.
\]

For a fixed rebalancing interval \(\Delta\), the exact discrete-time model is

\[
X_i = aX_{i-1} + \varepsilon_i,
\qquad
a=e^{-\theta\Delta},
\]

where

\[
\varepsilon_i\sim N(0,\sigma_\varepsilon^2),
\qquad
\sigma_\varepsilon^2=\omega^2(1-a^2),
\]

and

\[
\omega^2 = \operatorname{Var}(X_i)=\frac{\sigma^2}{2\theta}
\]

is the stationary mispricing variance.

The one-period return is

\[
r_i = p_i-p_{i-1}=X_i-X_{i-1}.
\]

---

## 5.2 Stationary covariance of returns

### Proposition 5.1

Under the stationary AR(1)/OU model,

\[
\operatorname{Var}(r_i)=2\omega^2(1-a),
\]

and for \(h\ge 1\),

\[
\operatorname{Cov}(r_i,r_{i-h})
=
-\omega^2(1-a)^2a^{h-1}.
\]

### Proof

Stationarity gives

\[
\operatorname{Cov}(X_i,X_j)=\omega^2 a^{|i-j|}.
\]

First,

\[
\begin{aligned}
\operatorname{Var}(r_i)
&=
\operatorname{Var}(X_i-X_{i-1}) \\
&=
\operatorname{Var}(X_i)+\operatorname{Var}(X_{i-1})
-2\operatorname{Cov}(X_i,X_{i-1})\\
&=
2\omega^2-2\omega^2 a\\
&=
2\omega^2(1-a).
\end{aligned}
\]

For \(h\ge 1\),

\[
r_i=X_i-X_{i-1},
\qquad
r_{i-h}=X_{i-h}-X_{i-h-1}.
\]

Thus

\[
\begin{aligned}
\operatorname{Cov}(r_i,r_{i-h})
&=
\operatorname{Cov}(X_i-X_{i-1},X_{i-h}-X_{i-h-1})\\
&=
\operatorname{Cov}(X_i,X_{i-h})
-\operatorname{Cov}(X_i,X_{i-h-1})\\
&\quad
-\operatorname{Cov}(X_{i-1},X_{i-h})
+\operatorname{Cov}(X_{i-1},X_{i-h-1})\\
&=
\omega^2 a^h-\omega^2 a^{h+1}
-\omega^2 a^{h-1}+\omega^2 a^h\\
&=
\omega^2 a^{h-1}(2a-a^2-1)\\
&=
-\omega^2(1-a)^2a^{h-1}.
\end{aligned}
\]

This negative autocovariance is the discrete-time signature of mean reversion.

---

## 5.3 EWM fair-value estimate

The trader does not observe true fair value externally. Instead, the trader estimates fair value with an exponentially weighted moving average of price:

\[
\widehat v_i = q\widehat v_{i-1} + (1-q)p_i,
\qquad
0\le q<1.
\]

Here \(q\) is the persistence of the estimator. The smoothing parameter is

\[
\eta=1-q.
\]

The estimated mispricing is

\[
\widetilde X_i = p_i-\widehat v_i.
\]

Since \(p_i=X_i\), we can write

\[
\begin{aligned}
\widetilde X_i
&=p_i-\widehat v_i\\
&=p_i-q\widehat v_{i-1}-(1-q)p_i\\
&=q(p_i-\widehat v_{i-1})\\
&=q(p_i-p_{i-1}+p_{i-1}-\widehat v_{i-1})\\
&=q(r_i+\widetilde X_{i-1}).
\end{aligned}
\]

So the estimated mispricing obeys the recursion

\[
\boxed{
\widetilde X_i=q(\widetilde X_{i-1}+r_i).
}
\]

If \(\widetilde X_0=0\), then iterating yields

\[
\boxed{
\widetilde X_i
=
q\sum_{j=1}^{i}q^{i-j}r_j.
}
\]

This identity is central. It shows that an EWM fair-value estimate converts the value signal into an exponentially weighted filter of past returns.

---

## 5.4 Trading signal and timing convention

The strategy trades against the estimated mispricing. To avoid lookahead bias, the position held during return \(r_i\) must be based on information available before \(r_i\).

Define the position during period \(i\) as

\[
s_i=-\gamma \widetilde X_{i-1},
\]

where \(\gamma>0\) is a position scale.

Using the EWM identity,

\[
\widetilde X_{i-1}
=
q\sum_{j=1}^{i-1}q^{i-1-j}r_j.
\]

Therefore,

\[
\boxed{
s_i
=
-\gamma q\sum_{j=1}^{i-1}q^{i-1-j}r_j.
}
\]

This is the mean-reversion analogue of the EMA trend-following signal. The sign is negative because the strategy trades against the estimated mispricing.

---

## 5.5 Stationary gross one-period P&L

The incremental gross P&L is

\[
\delta P_i = r_i s_i=-\gamma r_i\widetilde X_{i-1}.
\]

Since

\[
\widetilde X_{i-1}
=
q\sum_{h=1}^{\infty}q^{h-1}r_{i-h}
\]

in stationarity,

\[
\begin{aligned}
E[\delta P_i]
&=
-\gamma q\sum_{h=1}^{\infty}q^{h-1}
\operatorname{Cov}(r_i,r_{i-h})\\
&=
-\gamma q\sum_{h=1}^{\infty}q^{h-1}
\left[-\omega^2(1-a)^2a^{h-1}\right]\\
&=
\gamma q\omega^2(1-a)^2
\sum_{h=1}^{\infty}(aq)^{h-1}\\
&=
\boxed{
\gamma
\frac{q\omega^2(1-a)^2}{1-aq}
}.
\end{aligned}
\]

This is the first key performance formula.

**Interpretation:**
The strategy earns positive expected gross P&L because \(r_i\) is negatively correlated with past returns, while the EWM-estimated mispricing is an exponentially weighted average of past returns.

---

## 5.6 Stationary variance components

It is useful to compute the covariance structure of \(r_i\) and \(\widetilde X_{i-1}\). Let

\[
Z_i=\widetilde X_i.
\]

The state vector \((X_i,Z_i)\) obeys

\[
X_i=aX_{i-1}+\varepsilon_i,
\]

\[
Z_i=qZ_{i-1}+q(X_i-X_{i-1})
=
qZ_{i-1}+q(a-1)X_{i-1}+q\varepsilon_i.
\]

Solving the stationary covariance equations gives

\[
\operatorname{Var}(X_i)=\omega^2,
\]

\[
\operatorname{Cov}(X_i,Z_i)
=
\frac{\omega^2 q(1-a)}{1-aq},
\]

\[
\boxed{
\operatorname{Var}(Z_i)
=
\frac{2\omega^2 q^2(1-a)}{(1+q)(1-aq)}.
}
\]

Because

\[
r_i=(a-1)X_{i-1}+\varepsilon_i
\]

and \(\varepsilon_i\) is independent of \(Z_{i-1}\),

\[
\boxed{
\operatorname{Cov}(r_i,Z_{i-1})
=
-\frac{\omega^2q(1-a)^2}{1-aq}.
}
\]

Then

\[
E[\delta P_i]
=
-\gamma\operatorname{Cov}(r_i,Z_{i-1}),
\]

as above.

The variance of the product of two centered jointly Gaussian variables \(U,V\) is

\[
\operatorname{Var}(UV)
=
\operatorname{Var}(U)\operatorname{Var}(V)
+
\operatorname{Cov}(U,V)^2.
\]

Thus

\[
\boxed{
\operatorname{Var}(\delta P_i)
=
\gamma^2
\left[
\operatorname{Var}(r_i)\operatorname{Var}(Z_{i-1})
+
\operatorname{Cov}(r_i,Z_{i-1})^2
\right].
}
\]

Substituting the explicit terms:

\[
\operatorname{Var}(r_i)=2\omega^2(1-a),
\]

\[
\operatorname{Var}(Z_{i-1})
=
\frac{2\omega^2 q^2(1-a)}{(1+q)(1-aq)},
\]

\[
\operatorname{Cov}(r_i,Z_{i-1})^2
=
\frac{\omega^4 q^2(1-a)^4}{(1-aq)^2}.
\]

Therefore,

\[
\boxed{
\operatorname{Var}(\delta P_i)
=
\gamma^2\omega^4q^2(1-a)^2
\frac{
a^2q+a^2-6aq-2a+q+5
}{
(1+q)(1-aq)^2
}.
}
\]

This formula should be verified in simulation and used for the one-period gross Sharpe.

---

## 5.7 Gross one-period risk-adjusted P&L

Define

\[
\mu_P(q)
=
E[\delta P_i]
=
\gamma
\frac{q\omega^2(1-a)^2}{1-aq}.
\]

Define

\[
v_P(q)=\operatorname{Var}(\delta P_i).
\]

The stationary one-period gross risk-adjusted P&L is

\[
\mathcal S_{\mathrm{gross}}(q)
=
\frac{\mu_P(q)}{\sqrt{v_P(q)}}.
\]

Since both \(\mu_P\) and \(\sqrt{v_P}\) are proportional to \(\gamma\omega^2q\), the gross risk-adjusted P&L is scale invariant:

\[
\boxed{
\mathcal S_{\mathrm{gross}}(q)
=
(1-a)
\sqrt{
\frac{1+q}{
a^2q+a^2-6aq-2a+q+5
}
}.
}
\]

This is the local, incremental version. Cumulative P&L has serial dependence, so the full-horizon risk-adjusted P&L should be computed using the quadratic-form framework in Part 6.

---

## 5.8 Turnover preview

Transaction cost will be studied fully in Part 8, but Part 5 can preview the key formula.

For linear transaction cost \(c\), one-period cost is

\[
\mathcal C_i=c|s_i-s_{i-1}|.
\]

Since

\[
s_i=-\gamma Z_{i-1},
\]

we have

\[
\mathcal C_i=c\gamma |Z_{i-1}-Z_{i-2}|.
\]

The difference \(\Delta Z_i=Z_i-Z_{i-1}\) is Gaussian with mean zero. Therefore,

\[
E[\mathcal C_i]
=
c\gamma
\sqrt{\frac{2}{\pi}}
\sqrt{\operatorname{Var}(\Delta Z_i)}.
\]

Using

\[
Z_i=q(Z_{i-1}+r_i),
\]

one can show

\[
\boxed{
\operatorname{Var}(\Delta Z_i)
=
\frac{
2q^2\omega^2(1-a)(3-a-q-aq)
}{
(1+q)(1-aq)
}.
}
\]

Thus

\[
\boxed{
E[\mathcal C_i]
=
c\gamma
\sqrt{\frac{2}{\pi}}
\sqrt{
\frac{
2q^2\omega^2(1-a)(3-a-q-aq)
}{
(1+q)(1-aq)
}
}.
}
\]

This preview motivates the transaction-cost optimality problem.

---

## 5.9 Charts and tables for Part 5

### Figure 5.1: OU return autocovariance

**Goal:** validate the formula

\[
\operatorname{Cov}(r_i,r_{i-h})
=
-\omega^2(1-a)^2a^{h-1}.
\]

**Plot:**
- x-axis: lag \(h\)
- y-axis: autocovariance or autocorrelation
- lines: theory
- dots: simulation

**Parameter values:**
- \(\theta=1\)
- \(\sigma=1\)
- \(\Delta=1/252\) and maybe \(\Delta=1/12\)

### Figure 5.2: EWM estimated mispricing as return filter

**Goal:** numerically verify

\[
Z_i=q\sum_{j=1}^{i}q^{i-j}r_j.
\]

**Plot:**
- simulated \(Z_i=p_i-\widehat v_i\)
- reconstructed filtered-return version
- difference should be near zero.

### Figure 5.3: Expected gross P&L vs EWM persistence \(q\)

**Formula:**

\[
\mu_P(q)
=
\gamma
\frac{q\omega^2(1-a)^2}{1-aq}.
\]

**Plot:**
- x-axis: \(q\in[0,0.999]\)
- y-axis: expected one-period gross P&L
- several \(a\) values.

### Figure 5.4: Gross one-period risk-adjusted P&L vs \(q\)

Use

\[
\mathcal S_{\mathrm{gross}}(q)
=
(1-a)
\sqrt{
\frac{1+q}{
a^2q+a^2-6aq-2a+q+5
}
}.
\]

### Table 5.1: Key stationary formulas

| Quantity | Formula |
|---|---|
| \(\operatorname{Var}(r_i)\) | \(2\omega^2(1-a)\) |
| \(\operatorname{Cov}(r_i,r_{i-h})\) | \(-\omega^2(1-a)^2a^{h-1}\) |
| \(\operatorname{Var}(Z_i)\) | \(\frac{2\omega^2q^2(1-a)}{(1+q)(1-aq)}\) |
| \(\operatorname{Cov}(r_i,Z_{i-1})\) | \(-\frac{\omega^2q(1-a)^2}{1-aq}\) |
| \(E[\delta P_i]\) | \(\gamma\frac{q\omega^2(1-a)^2}{1-aq}\) |
| \(\operatorname{Var}(\Delta Z_i)\) | \(\frac{2q^2\omega^2(1-a)(3-a-q-aq)}{(1+q)(1-aq)}\) |

---

## 5.10 Code/notebook tasks for Part 5

Create notebook:

```text
notebooks/part05_discrete_ou_ewm.ipynb
```

Suggested Python functions:

```python
def simulate_ou_ar1(n_paths, n_steps, a, omega, rng):
    """Simulate stationary AR(1) OU mispricing X_i = a X_{i-1}+eps_i."""
```

```python
def ewm_fair_value(price, q):
    """Return vhat and estimated mispricing z = price - vhat."""
```

```python
def return_cov_theory(h, a, omega2):
    """Theoretical covariance of returns at lag h."""
```

```python
def stationary_formulas(a, q, omega2, gamma=1.0):
    """Return Var(r), Var(z), Cov(r,zprev), mean pnl, var pnl, Var(delta z)."""
```

```python
def pnl_linear_mean_reversion(X, q, gamma=1.0):
    """Compute returns, EWM fair value, signal, and one-period P&L."""
```

---

# Part 6: Quadratic-Form Representation of EWM Mean-Reversion P&L

## Blog title

**The P&L of an EWM Mean-Reversion Strategy as a Gaussian Quadratic Form**

## Paper role

This becomes Section 4 of the paper. It is the mathematical core that allows full distributional analysis.

---

## 6.1 Matrix notation

Let

\[
r=(r_1,\dots,r_T)^\top.
\]

Define the lower-triangular EWM matrix \(E_q\) by

\[
(E_q)_{ij}
=
\begin{cases}
q^{i-j-1}, & i>j,\\
0, & i\le j.
\end{cases}
\]

Then

\[
(E_qr)_i
=
\sum_{j=1}^{i-1}q^{i-j-1}r_j.
\]

Since

\[
s_i=-\gamma q(E_qr)_i,
\]

the signal vector is

\[
s=-\gamma qE_qr.
\]

Let \(O(t,t_0)\) be the diagonal selection matrix with ones for periods \(t_0+1,\ldots,t_0+t\), and zeros elsewhere.

The cumulative P&L over the selected window is

\[
P_{t,t_0}
=
r^\top O s
=
-\gamma q\,r^\top O E_qr.
\]

Only the symmetric part matters, so

\[
\boxed{
P_{t,t_0}
=
\frac12 r^\top M_q^{(t,t_0)}r
}
\]

where

\[
\boxed{
M_q^{(t,t_0)}
=
-\gamma q
\left(OE_q+E_q^\top O\right).
}
\]

---

## 6.2 Return covariance matrix

For stationary OU returns,

\[
C_{ij}=\operatorname{Cov}(r_i,r_j).
\]

Thus

\[
C_{ii}=2\omega^2(1-a),
\]

and for \(i\ne j\),

\[
C_{ij}
=
-\omega^2(1-a)^2a^{|i-j|-1}.
\]

This Toeplitz covariance matrix is the mean-reverting counterpart of the autocorrelated Gaussian return covariance matrix in the EMA trend-following paper.

---

## 6.3 Quadratic-form characteristic function

Let

\[
P=\frac12 r^\top M r,
\qquad
r\sim N(0,C).
\]

Then

\[
\boxed{
\phi_P(k)
=
E[e^{ikP}]
=
\det(I-ikMC)^{-1/2}.
}
\]

If \(\mu_1,\dots,\mu_T\) are the eigenvalues of \(MC\), then

\[
\phi_P(k)
=
\prod_{\ell=1}^{T}(1-ik\mu_\ell)^{-1/2}.
\]

This expression gives the full P&L distribution by inverse Fourier transform:

\[
p_P(z)
=
\frac{1}{2\pi}
\int_{-\infty}^{\infty}e^{-ikz}\phi_P(k)\,dk.
\]

---

## 6.4 Cumulants

The cumulant generating function is

\[
\log \phi_P(k)
=
-\frac12\log\det(I-ikMC).
\]

Using

\[
\log\det(I-A)=\operatorname{tr}\log(I-A),
\]

and the series expansion,

\[
\log(I-A)=-\sum_{m=1}^{\infty}\frac{A^m}{m},
\]

we obtain

\[
\log \phi_P(k)
=
\frac12\sum_{m=1}^{\infty}
\frac{(ik)^m}{m}
\operatorname{tr}[(MC)^m].
\]

Therefore,

\[
\boxed{
\kappa_m
=
\frac{(m-1)!}{2}
\operatorname{tr}[(MC)^m].
}
\]

In particular,

\[
E[P]=\kappa_1
=
\frac12\operatorname{tr}(MC),
\]

\[
\operatorname{Var}(P)=\kappa_2
=
\frac12\operatorname{tr}[(MC)^2],
\]

\[
\operatorname{Skew}(P)
=
\frac{\kappa_3}{\kappa_2^{3/2}},
\]

\[
\operatorname{ExcessKurt}(P)
=
\frac{\kappa_4}{\kappa_2^2}.
\]

If reporting ordinary kurtosis, use

\[
\operatorname{Kurt}(P)=3+\frac{\kappa_4}{\kappa_2^2}.
\]

---

## 6.5 Incremental P&L distribution

For one-period P&L,

\[
\delta P_i=-\gamma r_i Z_{i-1}.
\]

Because \((r_i,Z_{i-1})\) is bivariate Gaussian, \(\delta P_i\) is a scaled product of correlated Gaussian variables.

Let

\[
U=r_i,\qquad V=Z_{i-1}.
\]

Let

\[
\sigma_U^2=\operatorname{Var}(U),
\qquad
\sigma_V^2=\operatorname{Var}(V),
\qquad
\rho=\operatorname{Corr}(U,V).
\]

Then

\[
UV
\]

has moment generating function

\[
E[e^{tUV}]
=
\left(
1-2\rho\sigma_U\sigma_V t
-(1-\rho^2)\sigma_U^2\sigma_V^2 t^2
\right)^{-1/2}.
\]

For

\[
\delta P_i=-\gamma UV,
\]

replace \(t\mapsto -\gamma t\).

This gives a useful closed form for incremental P&L moments and density. The density of a product of correlated centered Gaussian variables is

\[
f_{UV}(z)
=
\frac{1}{\pi\sigma_U\sigma_V\sqrt{1-\rho^2}}
\exp\left(
\frac{\rho z}{(1-\rho^2)\sigma_U\sigma_V}
\right)
K_0\left(
\frac{|z|}{(1-\rho^2)\sigma_U\sigma_V}
\right),
\]

where \(K_0\) is the modified Bessel function of the second kind.

This is a useful contrast with the EMA trend-following paper’s incremental P&L Bessel density.

---

## 6.6 Charts and tables for Part 6

### Figure 6.1: Eigenvalue spectrum of \(MC\)

Plot eigenvalues of \(MC\) for:
- several horizons \(t\),
- several EWM persistence values \(q\),
- several OU persistence values \(a\).

### Figure 6.2: Simulated vs quadratic-form P&L distribution

Compare:
- Monte Carlo histogram of \(P_{t,t_0}\),
- numerical inverse Fourier density,
- Gaussian approximation using mean and variance.

### Figure 6.3: Skewness and kurtosis vs horizon

Analogous to the EMA paper’s skewness/kurtosis figure.

### Table 6.1: Matrix objects

| Object | Meaning | Formula |
|---|---|---|
| \(E_q\) | EWM memory matrix | \((E_q)_{ij}=q^{i-j-1}\mathbf 1_{i>j}\) |
| \(O\) | trading-window selector | diagonal |
| \(M_q\) | P&L quadratic-form matrix | \(-\gamma q(OE_q+E_q^\top O)\) |
| \(C\) | OU return covariance | Toeplitz |
| \(MC\) | spectral object | eigenvalues determine distribution |

---

## 6.7 Code/notebook tasks for Part 6

Create notebook:

```text
notebooks/part06_quadratic_form_pnl.ipynb
```

Functions:

```python
def E_matrix(T, q):
    """Lower-triangular EWM matrix E_q."""
```

```python
def O_matrix(T, t0, t):
    """Diagonal window selector."""
```

```python
def C_return_ou(T, a, omega2):
    """Toeplitz covariance matrix of OU returns."""
```

```python
def M_pnl(T, q, gamma, t0, t):
    """Symmetric P&L matrix."""
```

```python
def cumulants_quadratic(M, C, max_order=4):
    """Return cumulants using trace powers."""
```

```python
def charfun_quadratic(k, M, C):
    """Characteristic function det(I - i k M C)^(-1/2)."""
```

```python
def sample_gaussian_quadratic(M, C, n_paths, rng):
    """Monte Carlo sample P = 0.5 r.T @ M @ r."""
```

---

# Part 7: Distributional Shape and Tail Asymmetry

## Blog title

**The Tail Shape of Mean-Reversion P&L: Frequent Small Gains, Occasional Large Losses?**

## Paper role

This becomes Section 5 of the paper.

## Research hypothesis

Trend following tends to produce many small losses and occasional large profits. Mean reversion is expected to produce a different asymmetry: many small gains while harvesting oscillations, but occasional large losses when mispricing continues to move away from the estimate.

We should not assert this before the eigenvalue and quantile analysis. Part 7 should test and quantify it.

---

## 7.1 Tail asymptotics from eigenvalues

For the quadratic form

\[
P=\frac12 r^\top Mr,
\]

the eigenvalues of \(MC\) determine the exponential tail decay.

Let

\[
\mu_-=\min_\ell \mu_\ell <0,
\qquad
\mu_+=\max_\ell \mu_\ell >0.
\]

Then, asymptotically,

\[
p_P(z)\sim A_- |z|^{\nu_-}\exp(-z/\mu_-),
\qquad z\to-\infty,
\]

and

\[
p_P(z)\sim A_+ z^{\nu_+}\exp(-z/\mu_+),
\qquad z\to+\infty.
\]

Because \(\mu_-<0\), the negative-tail exponent is

\[
\exp(-|z|/|\mu_-|).
\]

The larger \(|\mu_-|\), the heavier the left tail. The larger \(\mu_+\), the heavier the right tail.

---

## 7.2 Approximate quantiles

For small \(q_L\),

\[
z_{q_L}
\approx
-|\mu_-|
\log\left(\frac{A_-|\mu_-|}{q_L}\right).
\]

For large \(q_R\),

\[
z_{q_R}
\approx
\mu_+
\log\left(\frac{A_+\mu_+}{1-q_R}\right).
\]

The prefactors are model-dependent, but the eigenvalues provide useful tail-scale diagnostics.

---

## 7.3 Charts for Part 7

### Figure 7.1: P&L density under mean reversion

Plot density on:
- linear scale,
- semi-log scale.

Overlay Gaussian approximation.

### Figure 7.2: Quantile fan

For \(q\in\{1\%,5\%,25\%,50\%,75\%,95\%,99\%\}\), plot quantiles of cumulative P&L versus horizon \(t\).

### Figure 7.3: Tail scales

Plot:
- \(|\mu_-|\) vs horizon,
- \(\mu_+\) vs horizon.

### Figure 7.4: Momentum vs mean-reversion schematic

Use same covariance strength but opposite autocorrelation sign if possible:
- positive autocorrelation + trend following,
- negative autocorrelation + mean reversion.

Compare skewness and tail eigenvalues.

---

# Part 8: Transaction Costs for the Linear EWM Strategy

## Blog title

**Transaction Costs in a Discrete EWM Mean-Reversion Strategy**

## Paper role

This becomes Section 6.1.

## Main message

Parts 5-7 analyze the gross distribution of the linear EWM mean-reversion strategy. Part 8 keeps the same exact-tracking rule,

\[
s_i=-\gamma Z_{i-1},
\]

and asks whether the frictionless edge survives proportional transaction costs. This is the direct mean-reversion analogue of the transaction-cost analysis in the EMA trend-following paper. The purpose is not yet to redesign the trading rule. It is to quantify expected turnover, expected transaction costs, net expected P&L, break-even costs, and cost-adjusted local risk-adjusted P&L for the baseline linear strategy.

The practical interpretation is that a frictionless mean-reversion signal can look attractive while still being untradeable if the EWM timescale is too short, the spread is too noisy, or the mean-reversion speed is too weak relative to costs.

---

## 8.1 Exact tracking of the frictionless target

The frictionless target from the earlier analysis is proportional to estimated mispricing:

\[
m_i=-\gamma Z_{i-1}.
\]

In Part 8, the actual position tracks this target exactly:

\[
s_i=m_i.
\]

This assumption is deliberately restrictive. It isolates the transaction cost of exact target tracking before Part 10 introduces no-trade bands, thresholds, and caps.

The trade size is

\[
\Delta s_i=s_i-s_{i-1}
=-\gamma(Z_{i-1}-Z_{i-2}).
\]

Thus transaction costs depend on the dynamics of the estimated mispricing \(Z_i\), not only on the dynamics of the raw asset return.

---

## 8.2 Linear transaction cost

Let

\[
\mathcal C_i = c|s_i-s_{i-1}|.
\]

Since

\[
s_i=-\gamma Z_{i-1},
\]

\[
s_i-s_{i-1}=-\gamma(Z_{i-1}-Z_{i-2}).
\]

So

\[
\mathcal C_i=c\gamma |\Delta Z_{i-1}|.
\]

Because \(\Delta Z\) is Gaussian,

\[
E[\mathcal C_i]
=
c\gamma
\sqrt{\frac{2}{\pi}}
\sqrt{\operatorname{Var}(\Delta Z)}.
\]

From Part 5,

\[
\operatorname{Var}(\Delta Z)
=
\frac{
2q^2\omega^2(1-a)(3-a-q-aq)
}{
(1+q)(1-aq)
}.
\]

Thus

\[
\boxed{
E[\mathcal C_i]
=
c\gamma
\sqrt{\frac{2}{\pi}}
\sqrt{
\frac{
2q^2\omega^2(1-a)(3-a-q-aq)
}{
(1+q)(1-aq)
}
}.
}
\]

---

## 8.3 Net one-period mean

Gross mean:

\[
\mu_P(q)
=
\gamma
\frac{q\omega^2(1-a)^2}{1-aq}.
\]

Cost mean:

\[
\mu_C(q)
=
c\gamma
\sqrt{\frac{2}{\pi}}
\sqrt{
\frac{
2q^2\omega^2(1-a)(3-a-q-aq)
}{
(1+q)(1-aq)
}
}.
\]

Net mean:

\[
\boxed{
\mu_{\mathrm{net}}(q)
=
\mu_P(q)-\mu_C(q).
}
\]

---

## 8.4 Net risk-adjusted P&L

Following the EMA paper’s convention, start by subtracting mean turnover from the numerator while using gross P&L variance in the denominator:

\[
\boxed{
\mathcal S_{\mathrm{net}}(q)
=
\frac{
\mu_P(q)-\mu_C(q)
}{
\sqrt{v_P(q)}
}.
}
\]

Later, optionally include the variance of costs:

\[
\operatorname{Var}(\delta P_i-\mathcal C_i)
\]

by simulation or numerical Gaussian integration.

---

## 8.5 Break-even transaction cost

The strategy is profitable in expectation if

\[
\mu_P(q)>\mu_C(q).
\]

Solving for \(c\),

\[
c < c_{\max}(q),
\]

where

\[
\boxed{
c_{\max}(q)
=
\frac{
\gamma\frac{q\omega^2(1-a)^2}{1-aq}
}{
\gamma\sqrt{\frac{2}{\pi}}
\sqrt{
\frac{
2q^2\omega^2(1-a)(3-a-q-aq)
}{
(1+q)(1-aq)
}
}
}.
}
\]

Canceling \(\gamma q\),

\[
\boxed{
c_{\max}(q)
=
\sqrt{\frac{\pi}{2}}
\frac{
\omega^2(1-a)^2/(1-aq)
}{
\sqrt{
\frac{
2\omega^2(1-a)(3-a-q-aq)
}{
(1+q)(1-aq)
}
}
}.
}
\]

Simplify:

\[
\boxed{
c_{\max}(q)
=
\sqrt{\frac{\pi}{2}}
\,
\omega(1-a)^{3/2}
\,
\frac{\sqrt{1+q}}
{\sqrt{2(3-a-q-aq)(1-aq)}}.
}
\]

This is the mean-reversion analogue of the transaction-cost viability condition.

---

## 8.6 Net P&L distribution

The gross cumulative P&L remains the quadratic-form object from Parts 6-7. Net P&L with proportional costs is

\[
P_T^{\mathrm{net}}
=
\sum_{i=1}^T r_i s_i
-
c\sum_{i=1}^T |s_i-s_{i-1}|.
\]

The absolute-value turnover term means net P&L is no longer a pure Gaussian quadratic form. Part 8 should therefore use the gross quadratic-form distribution as the analytical benchmark and evaluate net distributional effects by Monte Carlo or numerical Gaussian integration.

---

## 8.7 Charts and tables for Part 8

### Figure 8.1: Expected turnover vs \(q\)

Plot

\[
E|s_i-s_{i-1}|
\]

as a function of \(q\).

### Figure 8.2: Gross mean, cost, and net mean vs \(q\)

For several \(c\) values.

### Figure 8.3: Net risk-adjusted P&L vs \(q\)

Show an interior optimum.

### Figure 8.4: Break-even transaction cost \(c_{\max}(q)\)

Plot \(c_{\max}\) vs \(q\) and \(a\).

### Figure 8.5: Gross vs net P&L distribution

For a fixed \(a,q,c\), compare:
- gross quadratic-form P&L distribution from Part 6;
- simulated net P&L distribution after transaction costs;
- Gaussian approximation using net mean and variance.

### Table 8.1: Cost formulas

| Quantity | Formula |
|---|---|
| Turnover | \(|s_i-s_{i-1}|=\gamma|\Delta Z_{i-1}|\) |
| \(E[\text{turnover}]\) | \(\gamma\sqrt{2/\pi}\sqrt{\operatorname{Var}(\Delta Z)}\) |
| Expected cost | \(cE[\text{turnover}]\) |
| Net mean | \(\mu_P-\mu_C\) |
| Break-even cost | \(c_{\max}(q)\) |

---

## 8.8 Code/notebook tasks for Part 8

Implement:

```python
def turnover_variance(a, q, omega2):
    """Return Var(Delta Z_i) under the stationary discrete OU EWM model."""
```

```python
def expected_turnover(a, q, omega2, gamma=1.0):
    """Return E|s_i - s_{i-1}| for exact target tracking."""
```

```python
def expected_cost(a, q, omega2, c, gamma=1.0):
    """Return expected one-period linear transaction cost."""
```

```python
def net_mean_local(a, q, omega2, c, gamma=1.0):
    """Return gross mean, expected cost, and net mean."""
```

```python
def break_even_cost(a, q, omega):
    """Return c_max(q) such that expected net P&L is zero."""
```

```python
def net_pnl_exact_tracking(X, q, c, gamma=1.0):
    """Compute pathwise gross P&L, turnover cost, and net P&L."""
```

---

# Part 9: Optimal EWM Timescale

## Blog title

**The Optimal EWM Half-Life for a Mean-Reversion Strategy under Transaction Costs**

## Paper role

This becomes Section 6.2.

## Main message

Part 9 optimizes the EWM persistence \(q\), or equivalently the EWM half-life, after transaction costs. This is the direct counterpart to the Physica A paper's optimal EMA timescale analysis, but with the sign and covariance structure appropriate for mean reversion rather than trend following.

The practical question is:

\[
\text{How slowly should fair value be estimated when faster estimates trade more but slower estimates may leave stale signals?}
\]

Part 9 should express the answer in half-life units and study sensitivity to mean-reversion speed, volatility, rebalancing interval, and cost level.

---

## 9.1 Optimization problem

Find

\[
q^\star
=
\arg\max_{0\le q<1}
\mathcal S_{\mathrm{net}}(q).
\]

Then convert to an EWM half-life:

\[
h_{1/2}
=
\frac{\log(1/2)}{\log q}.
\]

In terms of \(\eta=1-q\), for small \(\eta\),

\[
h_{1/2}\approx\frac{\log 2}{\eta}.
\]

---

## 9.2 Qualitative regimes

### No transaction costs

When \(c=0\), optimize

\[
\mathcal S_{\mathrm{gross}}(q)
=
(1-a)
\sqrt{
\frac{1+q}{
a^2q+a^2-6aq-2a+q+5
}
}.
\]

This may be monotone in \(q\) for many parameter regimes. Verify analytically or numerically.

### With transaction costs

Costs penalize fast changes in \(Z_i\), which depend on \(q\). The optimal \(q\) may be interior.

Need to study whether \(q\to 1\) or \(q\to 0\) is optimal under costs. Use numerical optimization first, then derive asymptotic approximations.

---

## 9.3 Asymptotic expansions

For daily \(\Delta\) and mean-reversion speed \(\theta\), let

\[
a=e^{-\theta\Delta}\approx 1-\theta\Delta.
\]

For slow EWM, let

\[
q=e^{-\lambda\Delta}\approx 1-\lambda\Delta.
\]

Then

\[
1-a\approx\theta\Delta,
\qquad
1-q\approx\lambda\Delta,
\qquad
1-aq\approx(\theta+\lambda)\Delta.
\]

Use these to approximate:

\[
\mu_P(q)
=
\gamma
\frac{q\omega^2(1-a)^2}{1-aq}
\approx
\gamma\omega^2
\frac{\theta^2\Delta^2}{(\theta+\lambda)\Delta}
=
\gamma\omega^2
\frac{\theta^2}{\theta+\lambda}\Delta.
\]

Per unit time,

\[
\frac{\mu_P}{\Delta}
\approx
\gamma\omega^2
\frac{\theta^2}{\theta+\lambda}.
\]

Since \(\omega^2=\sigma^2/(2\theta)\),

\[
\frac{\mu_P}{\Delta}
\approx
\gamma
\frac{\sigma^2\theta}{2(\theta+\lambda)}.
\]

This matches the continuous-time EWM fair-value penalty from the earlier blog notes.

Now approximate turnover cost and derive per-unit-time scaling. This is important because turnover can diverge as \(\Delta\to 0\), confirming why discrete rebalancing matters.

---

## 9.4 Charts for Part 9

### Figure 9.1: Net objective vs EWM half-life

x-axis: half-life, not \(q\), because it is more interpretable.

### Figure 9.2: Optimal half-life vs transaction cost

Plot \(h_{1/2}^\star\) against \(c\).

### Figure 9.3: Optimal half-life vs mean-reversion speed

Plot \(h_{1/2}^\star\) against \(\theta\).

### Figure 9.4: Heatmap of \(q^\star\)

Axes:
- \(a\) or \(\theta\),
- transaction cost \(c\).

Color:
- \(q^\star\) or half-life.

---

## 9.5 Practical selection rule

The post should end with a practical rule:

1. Estimate the OU persistence \(a\) or speed \(\theta\).
2. Choose a plausible transaction cost \(c\) in the same price units as \(p_i\).
3. Compute \(\mathcal S_{\mathrm{net}}(q)\) over a stable half-life grid.
4. Avoid knife-edge optima by choosing the shortest half-life whose objective is within a tolerance of the maximum, or by imposing a turnover cap.
5. Carry the selected \(q^\star\) into Part 10, where exact target tracking is relaxed.

This final step matters because the theoretical maximizer may not be robust to calibration error.

---

## 9.6 Code/notebook tasks for Part 9

Implement:

```python
def q_to_half_life(q):
    """Convert EWM persistence q to half-life in rebalance periods."""
```

```python
def half_life_to_q(h):
    """Convert EWM half-life to persistence q."""
```

```python
def net_local_sharpe(a, q, omega2, c, gamma=1.0):
    """Return expected-cost-adjusted local risk-adjusted P&L."""
```

```python
def optimize_q_grid(a, omega2, c, q_grid, gamma=1.0):
    """Return q*, half-life*, and objective values on a grid."""
```

```python
def optimal_half_life_surface(theta_grid, cost_grid, sigma, delta):
    """Return a sensitivity surface for the cost-adjusted optimal half-life."""
```

---

# Part 10: No-Trade Bands and Tail-Aware Controls

## Blog title

**Trading Mean Reversion with Costs: No-Trade Bands, Position Caps, and Tail Control**

## Paper role

This becomes Section 6.3 and bridges analytical transaction-cost adjustment to practical strategy design.

## Main message

Parts 8 and 9 assume exact tracking of the frictionless target

\[
m_i=-\gamma Z_{i-1}.
\]

With proportional transaction costs, exact tracking is generally not the natural policy. Small changes in the target should often be ignored because the benefit of rebalancing is smaller than the cost. Part 10 introduces controlled trading rules that approximate the no-transaction-region logic from transaction-cost portfolio theory while staying compatible with the Gaussian OU simulation framework.

The main practical message is:

\[
\text{Costs change the strategy from target tracking to target-region control.}
\]

The post should compare exact tracking against no-trade bands, signal thresholds, position caps, smooth saturation, and tail-aware parameter choices.

---

## 10.1 Frictionless target and actual position

Define the frictionless target:

\[
m_i=-\gamma Z_{i-1}.
\]

The actual position \(s_i\) need not equal \(m_i\). It evolves according to a trading rule that balances three forces:
- expected mean-reversion edge from moving closer to \(m_i\);
- transaction cost from changing \(s_i\);
- risk and tail exposure from holding a large position while mispricing continues to move away.

The baseline exact-tracking rule is

\[
s_i=m_i.
\]

Part 10 should treat this as the benchmark, not as the final strategy.

---

## 10.2 No-trade band around the target

A simple no-trade band is:

\[
s_i=s_{i-1}
\quad \text{if} \quad
|s_{i-1}-m_i|\le b.
\]

If the previous position is outside the band, trade only to the nearest boundary:

\[
s_i=
\begin{cases}
m_i-b, & s_{i-1}<m_i-b,\\
s_{i-1}, & |s_{i-1}-m_i|\le b,\\
m_i+b, & s_{i-1}>m_i+b.
\end{cases}
\]

Here \(b\ge 0\) is the no-trade half-width. When \(b=0\), the rule becomes exact tracking. As \(b\) increases, turnover falls but tracking error rises.

This is not the full singular-control solution, but it captures the central economic implication of proportional costs.

---

## 10.3 Signal thresholds and cost hurdles

Another practical rule is to trade only when estimated mispricing exceeds a threshold:

\[
s_i=
\begin{cases}
-\gamma Z_{i-1}, & |Z_{i-1}|>z_{\min},\\
0, & |Z_{i-1}|\le z_{\min}.
\end{cases}
\]

Thresholding suppresses small signals. No-trade bands suppress small changes in the desired position. Part 10 should compare both mechanisms because they reduce turnover in different ways.

---

## 10.4 Position caps and nonlinear sizing

Mean-reversion losses can become large when mispricing continues to widen. A purely linear rule increases exposure exactly when the position is moving against the trader:

\[
|m_i|=\gamma |Z_{i-1}|.
\]

Part 10 should test capped sizing:

\[
s_i=\operatorname{clip}(-\gamma Z_{i-1},-s_{\max},s_{\max})
\]

and smooth saturation:

\[
s_i=-s_{\max}\tanh\left(\frac{\gamma Z_{i-1}}{s_{\max}}\right).
\]

The cap is a left-tail control. It reduces exposure in extreme mispricing states, potentially sacrificing some rebound profits in exchange for smaller adverse tail outcomes.

---

## 10.5 Tail-aware objective functions

Part 9 optimizes a Sharpe-like objective. Part 10 should add tail-aware objectives based on the Part 7 distributional analysis.

Possible objectives:

\[
\max_{q,b,\gamma}
\frac{E[P_T^{\mathrm{net}}]}
{\sqrt{\operatorname{Var}(P_T^{\mathrm{net}})}}
\]

\[
\max_{q,b,\gamma}
\frac{E[P_T^{\mathrm{net}}]}
{\operatorname{CVaR}_{\alpha}(-P_T^{\mathrm{net}})}
\]

or a constrained optimization:

\[
\max_{q,b,\gamma} E[P_T^{\mathrm{net}}]
\quad \text{subject to} \quad
\Pr(P_T^{\mathrm{net}}\le -L)\le \alpha.
\]

This section should emphasize that controlling the left tail is not the same as maximizing Sharpe. A rule can improve Sharpe while worsening extreme loss quantiles if it increases leverage or holds larger positions during persistent divergence.

---

## 10.6 Charts and tables for Part 10

### Figure 10.1: Exact tracking vs no-trade band paths

Plot target \(m_i\), exact-tracking position, and no-trade-band position on the same simulated path.

### Figure 10.2: Turnover and tracking error vs band width

Plot expected turnover, root mean squared tracking error \(E[(s_i-m_i)^2]^{1/2}\), and expected net P&L.

### Figure 10.3: Net Sharpe and left-tail quantile vs band width

Show that the Sharpe-optimal band and the tail-optimal band may differ.

### Figure 10.4: Position caps and left-tail reduction

Compare uncapped linear sizing, hard caps, and smooth saturation.

### Figure 10.5: Two-dimensional policy surface

Heatmap over EWM half-life \(h_{1/2}\) and no-trade half-width \(b\), colored by net Sharpe, 5% quantile, or CVaR.

### Table 10.1: Practical policy comparison

Rows:
- exact tracking;
- no-trade band;
- threshold entry;
- capped linear sizing;
- smooth saturation.

Columns:
- mean net P&L;
- net volatility;
- net Sharpe;
- turnover;
- 5% quantile;
- 1% quantile;
- CVaR.

---

## 10.7 Code/notebook tasks for Part 10

Implement:

```python
def target_position(z_prev, gamma=1.0):
    """Return frictionless target m_i = -gamma Z_{i-1}."""
```

```python
def apply_no_trade_band(target, band):
    """Return actual positions that trade only to no-trade-band boundaries."""
```

```python
def apply_signal_threshold(z_prev, gamma, z_min):
    """Return thresholded mean-reversion positions."""
```

```python
def apply_position_cap(target, cap):
    """Return hard-capped positions."""
```

```python
def apply_smooth_saturation(z_prev, gamma, cap):
    """Return smooth tanh-saturated positions."""
```

```python
def strategy_metrics(returns, positions, cost):
    """Return net P&L moments, turnover, quantiles, and CVaR."""
```

```python
def optimize_policy_grid(paths, q_grid, band_grid, gamma_grid, cost, objective):
    """Grid-search policy parameters under Sharpe or tail-aware objectives."""
```

---

# Part 11: Calibration Robustness and Parameter Uncertainty

## Blog title

**Calibration Risk in Costly Mean-Reversion Trading**

## Paper role

This becomes Section 7. It reuses Part 4 but adapts it to the new discrete-time and transaction-cost framework.

## Main message

Part 4 studied OU parameter estimation in the original continuous-time sequence. Part 11 should not simply repeat that post. It should ask how calibration uncertainty changes the choices made in Parts 8-10:
- estimated mean-reversion speed changes \(a\) and therefore expected gross edge;
- estimated volatility changes \(\omega\), turnover scale, and position sizing;
- estimated fair-value uncertainty changes the effective mispricing signal;
- uncertainty in \(a\), \(q^\star\), and band width can make the apparent optimum unstable.

The post should frame calibration as a robustness problem:

\[
\text{choose parameters that remain acceptable across plausible OU estimates.}
\]

---

## 11.1 Exact discrete OU likelihood

The exact discrete model is

\[
X_i=aX_{i-1}+b+\varepsilon_i,
\]

where

\[
a=e^{-\theta\Delta},
\qquad
b=\mu(1-a),
\]

and

\[
\varepsilon_i\sim N(0,\sigma_\varepsilon^2).
\]

Estimate \(a,b,\sigma_\varepsilon^2\) by OLS/MLE, then invert:

\[
\widehat\theta=-\frac{1}{\Delta}\log \widehat a,
\]

\[
\widehat\mu=\frac{\widehat b}{1-\widehat a},
\]

\[
\widehat\sigma^2
=
\widehat\sigma_\varepsilon^2
\frac{2\widehat\theta}{1-\widehat a^2}.
\]

---

## 11.2 Mean-estimation error as constant bias

For long calibration window \(T_{\mathrm{est}}\),

\[
\operatorname{Var}(\widehat\mu)
\approx
\frac{\sigma^2}{\theta^2 T_{\mathrm{est}}}.
\]

This can be derived from the variance of the sample mean of a stationary OU process.

Since a fair-value mean error acts like a constant bias \(M=\widehat\mu-\mu\), the earlier constant-bias penalty applies.

If

\[
s_\infty^2=\frac{\sigma^2}{2\theta},
\]

then

\[
\frac{\operatorname{Var}(\widehat\mu)}{s_\infty^2}
=
\frac{\sigma^2/(\theta^2T_{\mathrm{est}})}{\sigma^2/(2\theta)}
=
\frac{2}{\theta T_{\mathrm{est}}}.
\]

Thus the expected Sharpe penalty is approximately

\[
\boxed{
\left(1+\frac{2}{\theta T_{\mathrm{est}}}\right)^{-1/2}.
}
\]

This is one of the cleanest publishable calibration results.

---

## 11.3 Speed-estimation error and cost-adjusted overconfidence

The realized strategy may not depend strongly on \(\widehat\theta\) if the position is based on estimated mispricing rather than explicitly on \(\theta\). But forecasted Sharpe, leverage, optimal \(q\), and no-trade band width often use

\[
\widehat{SR}=f(\widehat\theta).
\]

If \(f\) is increasing and \(\widehat\theta\) is upward-biased, then expected forecasted Sharpe exceeds realized Sharpe. Under costs, the problem is broader: \(\widehat\theta\) affects both the expected edge and the chosen trading intensity.

Study:

\[
E[f(\widehat\theta)]-f(\theta).
\]

Use:
- delta method;
- Monte Carlo;
- finite-sample AR(1) bias.

---

## 11.4 Robust optimization under parameter uncertainty

Instead of choosing \(q,b,\gamma\) from a single fitted parameter vector, define a parameter set or posterior sample:

\[
\Theta_{\mathrm{plausible}}
=
\{(a,\omega,c): \text{consistent with calibration uncertainty}\}.
\]

Then choose a robust policy by maximizing a conservative objective:

\[
\max_{q,b,\gamma}
\min_{\vartheta\in\Theta_{\mathrm{plausible}}}
\mathcal S_{\mathrm{net}}(q,b,\gamma;\vartheta)
\]

or by imposing tail constraints across parameter scenarios:

\[
\Pr_{\vartheta}(P_T^{\mathrm{net}}\le -L)\le \alpha
\quad
\text{for all } \vartheta\in\Theta_{\mathrm{plausible}}.
\]

This connects Part 4's estimation-risk analysis to the implemented strategy design from Parts 8-10.

---

## 11.5 Charts and tables for Part 11

### Figure 11.1: Distribution of \(\widehat\theta\) and \(\widehat\mu\)

Recreate and improve the existing Part 4 figure.

### Figure 11.2: Mean-estimation Sharpe penalty vs \(\theta T_{\mathrm{est}}\)

Plot

\[
\left(1+\frac{2}{\theta T_{\mathrm{est}}}\right)^{-1/2}.
\]

### Figure 11.3: Forecasted vs realized cost-adjusted Sharpe

Show overconfidence from \(\widehat\theta\), including the induced error in \(q^\star\) or \(h_{1/2}^\star\).

### Figure 11.4: Robust objective surface

Plot the objective surface under low, median, and high fitted mean-reversion speeds.

### Table 11.1: Required calibration length

Rows:
- target penalty: 0.95, 0.90, 0.80, 0.70
- required \(\theta T_{\mathrm{est}}\)

Solve:

\[
\left(1+\frac{2}{\theta T_{\mathrm{est}}}\right)^{-1/2}=p.
\]

Then

\[
\theta T_{\mathrm{est}}
=
\frac{2}{p^{-2}-1}.
\]

---

# Part 12: Moving Fair Value and Signal Extraction

## Blog title

**When Fair Value Moves: Signal Extraction versus Mean Reversion**

## Paper role

This becomes an optional extension or discussion appendix.

## Recommendation

Use a latent local-level fair value rather than a mean-reverting fair value if the point is to model external fundamental uncertainty.

Let

\[
p_i=v_i+X_i,
\]

\[
X_i=aX_{i-1}+\varepsilon_i,
\]

\[
v_i=v_{i-1}+u_i.
\]

Then \(v_i\) is not a source of expected alpha; it is latent drift/noise. The EWM estimator must separate slow fair-value drift from fast mispricing.

The estimated mispricing is

\[
Z_i=p_i-\widehat v_i.
\]

The strategy trades

\[
s_i=-\gamma Z_{i-1}.
\]

The P&L is

\[
\delta P_i=s_i(p_i-p_{i-1}).
\]

Now returns contain both

\[
\Delta p_i=\Delta v_i+\Delta X_i.
\]

This model can generate a genuine bias-variance tradeoff in the EWM half-life:
- too slow: stale fair value;
- too fast: estimator tracks price noise and suppresses the mean-reversion signal.

---

# Part 13: Empirical or Calibrated Illustration

## Blog title

**Testing the Cost-Aware Mean-Reversion Framework: Simulated OU and Real Spreads**

## Paper role

Section 8.

---

## 13.1 Minimum viable empirical section

Start with calibrated simulation rather than real data.

Use plausible parameters:
- annualized \(\theta\),
- annualized \(\sigma\),
- daily \(\Delta=1/252\),
- transaction cost \(c\),
- EWM half-life from Part 9,
- optional no-trade band or cap from Part 10.

Generate:
- P&L distribution,
- skewness,
- quantiles,
- optimal \(q\),
- turnover,
- cost-adjusted performance,
- left-tail metrics such as 5% quantile and CVaR.

This proves the numerical pipeline.

---

## 13.2 Real-data option

Use pairs or spreads.

Candidate assets:
- ETF pairs with similar exposures.
- Futures calendar spreads.
- Sector-relative equity spreads.
- Statistical-arbitrage residuals.

Workflow:
1. Estimate spread or residual.
2. Fit AR(1)/OU.
3. Test autocorrelation and stationarity.
4. Apply EWM fair-value estimator.
5. Choose cost-aware EWM half-life.
6. Compare exact tracking, no-trade bands, thresholds, and caps.
7. Compute gross and net strategy P&L.
8. Compare theory vs empirical P&L moments, turnover, quantiles, and CVaR.

---

## 13.3 Charts

### Figure 13.1: Empirical spread and fitted OU

### Figure 13.2: Empirical return autocovariance vs OU fit

### Figure 13.3: Realized P&L distribution vs Gaussian quadratic-form prediction

### Figure 13.4: Empirical turnover and cost-adjusted performance

### Figure 13.5: Policy comparison under costs

Compare exact tracking, optimized EWM, no-trade band, thresholding, and capped sizing.

### Table 13.1: Practical strategy summary

Rows:
- gross exact tracking;
- net exact tracking;
- optimized EWM;
- no-trade band;
- capped strategy.

Columns:
- mean P&L;
- volatility;
- Sharpe;
- turnover;
- average cost;
- 5% quantile;
- CVaR.

---

# LaTeX Paper Maintenance Plan

## Paper repository structure

```text
steveya.github.io/
├── paper/
│   ├── main.tex
│   ├── sections/
│   │   ├── 00_abstract.tex
│   │   ├── 01_introduction.tex
│   │   ├── 02_discrete_ou_model.tex
│   │   ├── 03_ewm_signal.tex
│   │   ├── 04_quadratic_form.tex
│   │   ├── 05_distribution_quantiles.tex
│   │   ├── 06_transaction_costs_and_controls.tex
│   │   ├── 07_calibration_robustness.tex
│   │   ├── 08_extensions_and_illustration.tex
│   │   └── 09_conclusion.tex
│   ├── appendices/
│   │   ├── appendix_proofs.tex
│   │   ├── appendix_numerics.tex
│   │   └── appendix_extensions.tex
│   ├── figures/
│   └── references.bib

```

---

## Main LaTeX theorem inventory

### Proposition 1: OU return autocovariance

Statement from Part 5.

### Proposition 2: EWM estimated mispricing as return filter

\[
Z_i=q\sum_{j=1}^{i}q^{i-j}r_j.
\]

### Proposition 3: Stationary one-period gross mean and variance

\[
E[\delta P_i]
=
\gamma
\frac{q\omega^2(1-a)^2}{1-aq}.
\]

\[
\operatorname{Var}(\delta P_i)
=
\gamma^2
\left[
\operatorname{Var}(r_i)\operatorname{Var}(Z_{i-1})
+
\operatorname{Cov}(r_i,Z_{i-1})^2
\right].
\]

### Theorem 1: Quadratic-form representation

\[
P_{t,t_0}
=
\frac12 r^\top M_q^{(t,t_0)}r.
\]

### Theorem 2: Characteristic function and cumulants

\[
\phi_P(k)=\det(I-ikM_qC)^{-1/2}.
\]

\[
\kappa_m
=
\frac{(m-1)!}{2}\operatorname{tr}[(M_qC)^m].
\]

### Proposition 4: Expected turnover

\[
E[\mathcal C_i]
=
c\gamma
\sqrt{\frac{2}{\pi}}
\sqrt{
\frac{
2q^2\omega^2(1-a)(3-a-q-aq)
}{
(1+q)(1-aq)
}
}.
\]

### Proposition 5: Break-even transaction cost

\[
c_{\max}(q)
=
\sqrt{\frac{\pi}{2}}
\,
\omega(1-a)^{3/2}
\,
\frac{\sqrt{1+q}}
{\sqrt{2(3-a-q-aq)(1-aq)}}.
\]

### Proposition 6: Calibration-window penalty

\[
\text{Penalty}
\approx
\left(1+\frac{2}{\theta T_{\mathrm{est}}}\right)^{-1/2}.
\]

---

# Master figure list

| Figure | Post | Paper section | Purpose |
|---|---:|---:|---|
| OU return autocovariance | 5 | 2 | Validate negative autocorrelation |
| EWM signal reconstruction | 5 | 3 | Show \(Z_i\) equals return filter |
| Gross P&L vs \(q\) | 5 | 3 | Show mean-reversion edge |
| Quadratic-form density | 6 | 4 | Validate distribution formula |
| Eigenvalue spectrum | 6 | 4 | Tail-determining object |
| Skewness/kurtosis vs horizon | 6/7 | 5 | Compare to trend-following paper |
| Quantile fan | 7 | 5 | Show risk asymmetry |
| Turnover vs \(q\) | 8 | 6 | Cost mechanics |
| Gross vs net P&L distribution | 8 | 6 | Cost impact on distribution |
| Net objective vs half-life | 9 | 6 | Optimal EWM timescale |
| Optimal half-life heatmap | 9 | 6 | Sensitivity to costs and mean reversion |
| No-trade band policy paths | 10 | 6 | Exact tracking vs controlled trading |
| Net Sharpe and tail quantile vs band width | 10 | 6 | Sharpe and left-tail tradeoff |
| Position caps and left-tail reduction | 10 | 6 | Tail-aware sizing |
| MLE estimate distributions | 11 | 7 | Calibration risk |
| Robust objective surface | 11 | 7 | Parameter uncertainty impact |
| Moving fair-value stress test | 12 | 8 | Signal extraction risk |
| Theory vs empirical P&L | 13 | 8 | Validation |
| Policy comparison under costs | 13 | 8 | Practical strategy summary |

---

# Master table list

| Table | Purpose |
|---|---|
| Notation table | Avoid formula drift |
| Stationary formula table | Summarize Part 5 |
| Matrix object table | Summarize Part 6 |
| Tail diagnostic table | \(\mu_-\), \(\mu_+\), skewness, kurtosis |
| Transaction-cost formula table | Cost and break-even rules |
| Optimal half-life table | \(q^\star\) for parameter scenarios |
| Policy comparison table | Mean, turnover, quantiles, and CVaR by trading rule |
| Calibration-window table | Required \(T_{\mathrm{est}}\) for penalty levels |
| Empirical summary table | Theory vs realized moments |

---

# Immediate Codex Task List

## Task A: Create source modules

Implement:

```text
src/ou.py
src/ewm.py
src/quadratic_form.py
src/costs.py
src/simulation.py
src/plotting.py
```

## Task B: Implement Part 5 notebook

Create:

```text
notebooks/part05_discrete_ou_ewm.ipynb
```

Must generate:
- Figure 5.1
- Figure 5.2
- Figure 5.3
- Figure 5.4
- Table 5.1

## Task C: Create Part 5 Quarto post

Create:

```text
posts/part05_discrete_ou_ewm.qmd
```

Use the derivations in this plan. Keep the post narrative clean:
1. why discretize;
2. discrete OU;
3. return autocovariance;
4. EWM fair value;
5. signal and P&L;
6. stationary mean/variance;
7. preview of costs and quadratic form.

## Task D: Start LaTeX sections

Create:

```text
paper/sections/02_discrete_ou_model.tex
paper/sections/03_ewm_signal.tex
```

Copy formal propositions and proofs from Part 5.

---

# Final immediate writing target

The next blog post and paper section should be:

# Part 5 — Discretizing the Mean-Reversion Strategy

Core proposition chain:

1. OU exact discretization:
   \[
   X_i=aX_{i-1}+\varepsilon_i.
   \]

2. Return autocovariance:
   \[
   \operatorname{Cov}(r_i,r_{i-h})
   =
   -\omega^2(1-a)^2a^{h-1}.
   \]

3. EWM fair-value estimate:
   \[
   \widehat v_i=q\widehat v_{i-1}+(1-q)p_i.
   \]

4. Estimated mispricing as EWM return filter:
   \[
   \widetilde X_i
   =
   q\sum_{j=1}^{i}q^{i-j}r_j.
   \]

5. Signal:
   \[
   s_i=-\gamma \widetilde X_{i-1}.
   \]

6. Gross mean:
   \[
   E[\delta P_i]
   =
   \gamma
   \frac{q\omega^2(1-a)^2}{1-aq}.
   \]

7. Gross variance:
   \[
   \operatorname{Var}(\delta P_i)
   =
   \gamma^2\omega^4q^2(1-a)^2
   \frac{
   a^2q+a^2-6aq-2a+q+5
   }{
   (1+q)(1-aq)^2
   }.
   \]

8. Turnover preview:
   \[
   E[\mathcal C_i]
   =
   c\gamma
   \sqrt{\frac{2}{\pi}}
   \sqrt{
   \frac{
   2q^2\omega^2(1-a)(3-a-q-aq)
   }{
   (1+q)(1-aq)
   }
   }.
   \]

9. Preview quadratic form:
   \[
   P_{t,t_0}=\frac12 r^\top M_qr.
   \]

This is enough for one blog-length technical post and enough to begin the LaTeX paper.
