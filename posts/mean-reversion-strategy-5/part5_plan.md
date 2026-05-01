# Part 5 Plan: Discretizing the Mean-Reversion Strategy

## Objective

Write Part 5 as the new foundation for the paper-oriented mean-reversion sequence. The post should restart from a discrete-time OU mispricing model, introduce an exponentially weighted fair-value estimate, derive the mean-reversion signal under a no-lookahead timing convention, and obtain the stationary one-period gross PnL formulas that motivate Parts 6 through 9.

The post should become the source article for paper Sections 2 and 3. It should also provide the notation and simulation utilities that later parts can reuse for quadratic-form distributions, turnover, transaction costs, and optimal EWM half-life.

## Reference Style

Use `posts/mean-reversion-strategy-4/index.ipynb` as the writing and notebook style reference.

The first markdown cell should contain Quarto front matter followed immediately by the opening narrative. Use the same series metadata style as Part 4, with `content-type`, `series-id`, `series-title`, `series-part`, and `project-ids`. Part 5 should link back to Parts 1 through 4 in the first paragraph and explain why the earlier continuous-time analysis now has to be discretized.

The notebook should alternate substantial markdown sections with folded code cells. The first code cell should be hidden and should contain imports, plotting defaults, random seed setup, and helper functions. Figure-producing code cells should use Quarto options in source comments:

```python
#| code-fold: true
#| code-summary: "Show simulation code"
#| label: fig-return-autocovariance
#| fig-cap: 'OU return autocovariance. Simulation dots are compared with the theoretical covariance $-\omega^2(1-a)^2a^{h-1}$.'
```

Markdown tables should follow the Part 4 pattern, with the caption and label after the table:

```text
| Quantity | Formula |
|:---|:---|
| Return variance | $2\omega^2(1-a)$ |

: Key stationary formulas for the discrete OU EWM strategy. {#tbl-stationary-formulas}
```

## Jupyter LaTeX Rules

Part 4 uses dollar-delimited math in markdown cells. Follow that convention throughout Part 5.

Use inline math as `$...$`, for example `$a=e^{-\theta\Delta}$`. Use display math as `$$...$$`, not `\[` and `\]`, because the existing notebook style and Quarto rendering path already use dollar-delimited display equations.

Multi-line derivations should be written as display math blocks:

```text
$$
\begin{aligned}
\operatorname{Var}(r_i)
&= \operatorname{Var}(X_i-X_{i-1}) \\
&= 2\omega^2 - 2\omega^2 a \\
&= 2\omega^2(1-a)
\end{aligned}
$$
```

Do not put punctuation inside or immediately after display equations unless it is part of the mathematical expression. Prefer a short sentence before or after the equation. Keep equation notation consistent: use \(X_i\) for mispricing, \(p_i\) for log price, \(r_i\) for one-period return, \(\widehat v_i\) for estimated fair value, \(Z_i\) or \(\widetilde X_i\) for estimated mispricing, \(q\) for EWM persistence, \(a\) for OU persistence, \(\omega^2\) for stationary mispricing variance, and \(\gamma\) for position scale.

Avoid colon-equals notation entirely. When defining notation, write phrases such as "where \(a=e^{-\theta\Delta}\)" or "define \(Z_i=\widetilde X_i\)".

## Narrative Arc

The opening should state the pivot from the earlier continuous-time posts. Parts 1 through 4 established the continuous-time benchmark, bias penalties, trailing-estimator penalties, and parameter-estimation penalties. Part 5 should explain that those results are useful but incomplete for trading implementation because proportional transaction costs and discrete rebalancing require a sampled strategy.

The main message should be that discretization is not just a numerical convenience. It turns the OU model into a Gaussian AR(1), gives an explicit negative return autocovariance, and makes the EWM fair-value signal a linear filter of past returns. That linear-filter representation is the bridge to the quadratic-form PnL distribution in Part 6.

The post should avoid presenting transaction-cost or full-horizon distribution results as finished. It should preview them only enough to motivate the next posts.

## Notebook Outline

### Opening Cell

Set front matter:

```yaml
---
title: 'Discretizing the Mean-Reversion Strategy: EWM Fair Value, Return Autocovariance, and Gross PnL'
date: 2026-04-24
description: "A discrete OU model turns an EWM fair-value estimate into a return filter and yields closed-form gross PnL formulas."
categories: [Quantitative Finance, Trading Strategies, Stochastic Processes]
content-type: "series"
series-id: "mean-reversion-strategy"
series-title: "Expected Performance of a Mean-Reversion Trading Strategy"
series-part: 5
project-ids: [mean-reversion-strategy]
---
```

The opening prose should link to the prior parts and define the post's role. It should end with the result chain: exact discrete OU, return autocovariance, EWM fair value, no-lookahead signal, expected gross PnL, one-period variance, and turnover preview.

### Setup Code Cell

Use a hidden setup cell with imports and shared helpers. Keep dependencies limited to `numpy`, `pandas`, `matplotlib`, and optionally `IPython.display.Markdown` if a formula table is easier to generate from code.

Implement helpers:

```python
def simulate_ou_ar1(n_paths, n_steps, a, omega, rng):
    """Simulate stationary AR(1) OU mispricing X_i = a X_{i-1} + eps_i."""
```

```python
def ewm_fair_value(price, q):
    """Return vhat and estimated mispricing z = price - vhat."""
```

```python
def return_cov_theory(h, a, omega2):
    """Theoretical covariance Cov(r_i, r_{i-h}) for h >= 1."""
```

```python
def stationary_formulas(a, q, omega2, gamma=1.0):
    """Return stationary variance, covariance, PnL, Sharpe, and turnover terms."""
```

```python
def pnl_linear_mean_reversion(X, q, gamma=1.0):
    """Compute returns, EWM fair value, no-lookahead signal, and one-period PnL."""
```

### Section 1: Discrete OU Model

Introduce the continuous-time reference model only briefly:

$$
dX_t = -\theta X_t\,dt + \sigma\,dW_t
$$

Then move immediately to fixed rebalancing interval \(\Delta\):

$$
X_i = aX_{i-1} + \varepsilon_i,\qquad a=e^{-\theta\Delta}
$$

State the stationary innovation variance:

$$
\varepsilon_i\sim N(0,\sigma_\varepsilon^2),\qquad
\sigma_\varepsilon^2=\omega^2(1-a^2),\qquad
\omega^2=\frac{\sigma^2}{2\theta}
$$

Define price and return:

$$
p_i=X_i,\qquad r_i=p_i-p_{i-1}=X_i-X_{i-1}
$$

Include a short notation paragraph after these definitions. The notation paragraph should make clear that the true fair value is normalized to zero in Part 5, and that this is a modeling normalization, not a claim that fair value is observable in practice.

### Section 2: Return Autocovariance

State Proposition 5.1. Under the stationary AR(1) model:

$$
\operatorname{Var}(r_i)=2\omega^2(1-a)
$$

and for \(h\ge 1\):

$$
\operatorname{Cov}(r_i,r_{i-h})
=-\omega^2(1-a)^2a^{h-1}
$$

Give the full derivation in the post, following the master plan. This should be one of the central mathematical sections and should not be abbreviated. The prose should emphasize that the negative autocovariance is the discrete-time signature that the strategy will exploit.

Add Figure 5.1 after the proof. The figure should compare theoretical autocovariance or autocorrelation with simulation dots. Use parameters \(\theta=1\), \(\sigma=1\), and \(\Delta=1/252\), with an optional second line for \(\Delta=1/12\) if the plot remains readable.

Suggested label:

```python
#| label: fig-return-autocovariance
```

### Section 3: EWM Fair Value

Introduce the EWM fair-value estimate:

$$
\widehat v_i = q\widehat v_{i-1} + (1-q)p_i,\qquad 0\le q<1
$$

Define estimated mispricing:

$$
\widetilde X_i = p_i-\widehat v_i
$$

Derive the recursion:

$$
\begin{aligned}
\widetilde X_i
&=p_i-\widehat v_i \\
&=p_i-q\widehat v_{i-1}-(1-q)p_i \\
&=q(p_i-\widehat v_{i-1}) \\
&=q(r_i+\widetilde X_{i-1})
\end{aligned}
$$

Then state the filter identity:

$$
\widetilde X_i
=q\sum_{j=1}^{i}q^{i-j}r_j
$$

Add Figure 5.2 after the derivation. Simulate a path, compute \(Z_i=p_i-\widehat v_i\), reconstruct the same object from the filtered returns, and plot both series plus a small residual panel or residual summary. The residual should be numerical noise if initialization conventions match.

Suggested label:

```python
#| label: fig-ewm-signal-reconstruction
```

### Section 4: Signal Timing

This section should be explicit because it determines the sign and avoids lookahead bias. Define the position held during return \(r_i\) as:

$$
s_i=-\gamma\widetilde X_{i-1}
$$

Then substitute the filter identity:

$$
s_i=-\gamma q\sum_{j=1}^{i-1}q^{i-1-j}r_j
$$

The text should make two points. First, the sign is negative because the strategy trades against estimated mispricing. Second, the position is lagged because the return \(r_i\) is not known when the position for period \(i\) is chosen.

This is the conceptual bridge to Part 6. Add a paragraph stating that the signal is now a linear transformation of lagged returns, so cumulative PnL is a quadratic form in the Gaussian return vector.

### Section 5: Gross One-Period PnL

Define one-period gross PnL:

$$
\delta P_i = r_i s_i = -\gamma r_i\widetilde X_{i-1}
$$

Use the stationary infinite-history identity:

$$
\widetilde X_{i-1}
=q\sum_{h=1}^{\infty}q^{h-1}r_{i-h}
$$

Then derive the mean:

$$
\begin{aligned}
E[\delta P_i]
&=-\gamma q\sum_{h=1}^{\infty}q^{h-1}\operatorname{Cov}(r_i,r_{i-h}) \\
&=\gamma q\omega^2(1-a)^2\sum_{h=1}^{\infty}(aq)^{h-1} \\
&=\gamma\frac{q\omega^2(1-a)^2}{1-aq}
\end{aligned}
$$

Add Figure 5.3 after this section. Plot expected one-period gross PnL versus \(q\) for several \(a\) values. Keep the y-axis label explicit and avoid implying that this is a full-horizon Sharpe ratio.

Suggested label:

```python
#| label: fig-gross-pnl-vs-q
```

### Section 6: Variance and Local Sharpe

Define \(Z_i=\widetilde X_i\) and use the state recursion:

$$
Z_i=qZ_{i-1}+q(X_i-X_{i-1})
=qZ_{i-1}+q(a-1)X_{i-1}+q\varepsilon_i
$$

State the stationary covariance components:

$$
\operatorname{Var}(Z_i)
=\frac{2\omega^2q^2(1-a)}{(1+q)(1-aq)}
$$

$$
\operatorname{Cov}(r_i,Z_{i-1})
=-\frac{\omega^2q(1-a)^2}{1-aq}
$$

Use the product formula for centered jointly Gaussian variables:

$$
\operatorname{Var}(UV)
=\operatorname{Var}(U)\operatorname{Var}(V)+\operatorname{Cov}(U,V)^2
$$

Then state:

$$
\operatorname{Var}(\delta P_i)
=\gamma^2\omega^4q^2(1-a)^2
\frac{
a^2q+a^2-6aq-2a+q+5
}{
(1+q)(1-aq)^2
}
$$

Derive the local gross risk-adjusted PnL:

$$
\mathcal S_{\mathrm{gross}}(q)
=(1-a)
\sqrt{
\frac{1+q}{
a^2q+a^2-6aq-2a+q+5
}
}
$$

Add Figure 5.4 after this section. Plot local gross risk-adjusted PnL versus \(q\). The caption should state that cumulative PnL is serially dependent and will be handled in Part 6.

Suggested label:

```python
#| label: fig-local-sharpe-vs-q
```

### Section 7: Turnover Preview

Keep this section shorter than the PnL sections. Its purpose is to motivate Part 8, not to solve the full cost-adjusted problem.

State:

$$
\mathcal C_i=c|s_i-s_{i-1}|
=c\gamma|Z_{i-1}-Z_{i-2}|
$$

Then use the Gaussian absolute moment:

$$
E[\mathcal C_i]
=c\gamma\sqrt{\frac{2}{\pi}}\sqrt{\operatorname{Var}(\Delta Z_i)}
$$

and preview:

$$
\operatorname{Var}(\Delta Z_i)
=
\frac{
2q^2\omega^2(1-a)(3-a-q-aq)
}{
(1+q)(1-aq)
}
$$

This section should conclude that \(q\) affects both expected gross PnL and expected turnover, which creates the Part 8 and Part 9 optimization problem.

### Section 8: Formula Table

Add one compact table near the end. Use inline math in table cells rather than display math. The table should include:

| Quantity | Formula |
|:---|:---|
| \(\operatorname{Var}(r_i)\) | \(2\omega^2(1-a)\) |
| \(\operatorname{Cov}(r_i,r_{i-h})\) | \(-\omega^2(1-a)^2a^{h-1}\) |
| \(\operatorname{Var}(Z_i)\) | \(\frac{2\omega^2q^2(1-a)}{(1+q)(1-aq)}\) |
| \(\operatorname{Cov}(r_i,Z_{i-1})\) | \(-\frac{\omega^2q(1-a)^2}{1-aq}\) |
| \(E[\delta P_i]\) | \(\gamma\frac{q\omega^2(1-a)^2}{1-aq}\) |
| \(\operatorname{Var}(\delta P_i)\) | formula from Section 6 |
| \(\operatorname{Var}(\Delta Z_i)\) | \(\frac{2q^2\omega^2(1-a)(3-a-q-aq)}{(1+q)(1-aq)}\) |

Use the label:

```text
{#tbl-stationary-formulas}
```

### Discussion

The discussion should summarize the chain without overselling it. The key claim is that an EWM fair-value estimate converts a sampled OU mean-reversion strategy into a Gaussian linear-filter problem. This delivers closed-form one-period gross PnL and makes the cumulative PnL a Gaussian quadratic form.

End by pointing directly to Part 6: the next step is to build the EWM memory matrix and use the return covariance matrix to derive the full distribution of cumulative PnL.

## Figure Checklist

Figure 5.1, `fig-return-autocovariance`: theoretical OU return autocovariance with simulation validation.

Figure 5.2, `fig-ewm-signal-reconstruction`: direct EWM estimated mispricing versus filtered-return reconstruction.

Figure 5.3, `fig-gross-pnl-vs-q`: expected one-period gross PnL as a function of EWM persistence.

Figure 5.4, `fig-local-sharpe-vs-q`: local one-period gross risk-adjusted PnL as a function of EWM persistence.

## Implementation Checks

The notebook should parse as valid JSON after editing. All code cells should execute top to bottom with a fixed seed. Simulation figures should use enough paths to make the theory comparison stable without making the notebook slow to render. Figure captions should contain the mathematical claim being validated.

Before publishing, run a local notebook execution only if requested or if a rendering issue needs debugging. Until publication, keep the work in `posts/mean-reversion-strategy-5/draft.ipynb`, which remains ignored by Quarto rendering.

## Open Decisions

Decide whether to use \(Z_i\) throughout after first defining \(Z_i=\widetilde X_i\), or to keep \(\widetilde X_i\) in the prose and reserve \(Z_i\) for formulas that would otherwise become visually heavy. My recommendation is to define both once, then use \(Z_i\) in longer derivations.

Decide whether Figure 5.1 should show autocovariance or autocorrelation. Autocorrelation is easier to compare across sampling intervals, but autocovariance aligns more directly with Proposition 5.1. My recommendation is to plot autocorrelation on the main axis and quote the covariance formula in the caption.

Decide whether Part 5 should include a very short quadratic-form preview. My recommendation is yes, but only one paragraph and one equation:

$$
P_{t,t_0}=\frac{1}{2}r^\top M_q r
$$

The matrix construction itself belongs in Part 6.
