# Literature Positioning Research Note

## Purpose

This note summarizes the literature context for the arXiv paper on EWM fair-value mean-reversion trading. The immediate use is to support the introduction, literature review, and contribution statement. The working paper should be positioned as a distributional and implementation-oriented mean-reversion analogue of the EMA trend-following analysis of Grebenkov and Serror.

## Central Positioning

The paper sits at the intersection of four literatures:

1. Analytical PnL distributions for rule-based trading strategies.
2. Mean-reversion and statistical-arbitrage models, especially OU and pairs-trading models.
3. Dynamic trading with transaction costs.
4. Distributional and tail-risk analysis of quadratic forms in Gaussian variables.

The concise positioning claim is:

> We develop the mean-reversion analogue of the EMA trend-following analysis of Grebenkov and Serror, replacing positive-autocorrelation momentum with an OU-driven value signal based on an estimated fair value. The paper derives the finite-horizon PnL distribution, tail diagnostics, turnover, transaction-cost viability, optimal EWM half-life, and practical tail-aware controls.

The novelty is not that OU mean reversion exists, nor that transaction costs matter. The contribution is a unified analytical framework for the implemented EWM fair-value strategy: sampled prices, lagged signals, finite-horizon PnL, transaction costs, and tail-risk diagnostics.

## Closest Methodological Reference

Grebenkov and Serror study a trend-following strategy based on an exponential moving average in a Gaussian model. They derive the distribution of PnL, its moments, skewness, kurtosis, asymptotic quantiles, turnover, transaction-cost adjustment, and an optimal trend-following timescale. Their economic object is momentum or trend following, where the strategy is aligned with positive autocorrelation.

Our paper uses the same broad analytical template but changes the economic object. We study a mean-reversion strategy whose signal is an EWM estimate of fair value. The return autocovariance has the opposite sign, the trading rule is contrarian, and the distributional asymmetry is expected to differ. The closest one-sentence comparison is:

> Grebenkov and Serror analyze how Gaussian price variations are transformed into PnL by an EMA trend-following rule; we analyze how Gaussian OU mispricing is transformed into PnL by an EWM fair-value mean-reversion rule.

Relevant source:
- Grebenkov and Serror, "Following a Trend with an Exponential Moving Average: Analytical Results for a Gaussian Model," Physica A, 2014. DOI: 10.1016/j.physa.2013.10.007. arXiv: https://arxiv.org/abs/1308.5658

## Mean-Reversion and Statistical-Arbitrage Literature

The OU and pairs-trading literature often models a spread or residual as mean reverting. Elliott, van der Hoek, and Malcolm propose a Gaussian Markov-chain framework for pairs trading with noisy spread observations. Bertram derives analytical entry and exit thresholds for OU statistical arbitrage under transaction costs. Leung and Li formulate optimal mean-reversion trading as an optimal stopping problem with transaction costs and stop-loss constraints. The sparse mean-reverting portfolio literature, including d'Aspremont and related work, focuses on constructing portfolios whose aggregate value is strongly mean reverting.

These papers are highly relevant, but their main objects differ from ours. Much of the optimal statistical-arbitrage literature emphasizes entry and exit thresholds, first-passage times, double stopping, or portfolio construction. Our paper instead studies the full finite-horizon distribution of a continuously rebalanced EWM fair-value strategy. The signal is not an entry/exit boundary alone; it is a lagged linear filter of sampled returns, which makes cumulative gross PnL a Gaussian quadratic form.

Relevant sources:
- Elliott, van der Hoek, and Malcolm, "Pairs Trading," Quantitative Finance, 2005. DOI: 10.1080/14697680500149370.
- Bertram, "Analytic Solutions for Optimal Statistical Arbitrage Trading," Physica A, 2010. DOI: 10.1016/j.physa.2010.01.045.
- Leung and Li, "Optimal Mean Reversion Trading with Transaction Costs and Stop-Loss Exit," arXiv:1411.5062.
- d'Aspremont, "Identifying Small Mean-Reverting Portfolios," Quantitative Finance, 2011. DOI: 10.1080/14697688.2010.481634.

## Transaction-Cost Literature

The transaction-cost literature provides the implementation logic for Parts 8 through 10. Garleanu and Pedersen derive dynamic trading policies with predictable returns and transaction costs. A key message is that costly trading generally implies partial adjustment toward an aim portfolio, not exact tracking of a frictionless target. Classical proportional transaction-cost models, such as Davis and Norman and Shreve and Soner, show that proportional costs naturally create no-trade regions.

This literature motivates the structure of our transaction-cost extension. Part 8 first quantifies exact tracking of the frictionless target,

$$
s_i=-\gamma Z_{i-1},
$$

because this gives a clean mean-reversion analogue of the EMA transaction-cost calculation. Part 10 then relaxes exact tracking using reduced-form no-trade bands, thresholds, caps, and tail-aware objectives. We should be explicit that these controls are not claimed to be the full singular-control solution; they are practical policies designed to preserve the distributional tractability and simulation pipeline.

Relevant sources:
- Garleanu and Pedersen, "Dynamic Trading with Predictable Returns and Transaction Costs," Journal of Finance, 2013. DOI: 10.1111/jofi.12080. NBER: https://www.nber.org/papers/w15205.
- Davis and Norman, "Portfolio Selection with Transaction Costs," Mathematics of Operations Research, 1990. DOI: 10.1287/moor.15.4.676.
- Shreve and Soner, "Optimal Investment and Consumption with Transaction Costs," Annals of Applied Probability, 1994. DOI: 10.1214/aoap/1177004966.

## Quadratic-Form Distribution Literature

Part 6 relies on the classical theory of quadratic forms in normal variables. Once cumulative gross PnL is written as

$$
P=\frac{1}{2}r^\top M r,
$$

with Gaussian return vector \(r\), the characteristic function, cumulants, and eigenvalue representation follow from the distribution of Gaussian quadratic forms. Imhof provides a foundational method for computing these distributions. This literature supports the use of eigenvalues of \(MC\) to study skewness, kurtosis, quantiles, and tail asymmetry.

Our applied contribution is to identify the matrix \(M\) generated by the EWM mean-reversion trading rule and the covariance matrix \(C\) generated by sampled OU returns. The quadratic-form theory is classical; the trading-strategy specialization is the contribution.

Relevant source:
- Imhof, "Computing the Distribution of Quadratic Forms in Normal Variables," Biometrika, 1961. DOI: 10.1093/biomet/48.3-4.419.

## Suggested Introduction Structure

The introduction should move through the following argument.

First, motivate value and mean-reversion trading as a fair-value problem. A mean-reversion strategy does not simply forecast returns; it estimates a fair value and trades against deviations from that estimate.

Second, explain why the continuous-time frictionless result is incomplete. In frictionless continuous time, trading proportionally to mispricing is natural. In implementation, prices are sampled, fair value is estimated, positions are lagged, and transaction costs penalize frequent rebalancing.

Third, introduce the paper's discrete Gaussian framework. The OU model remains the primitive source of mean reversion, but the implemented signal is an EWM fair-value estimate. This turns estimated mispricing into an exponentially weighted filter of past returns.

Fourth, state the distributional contribution. Because the signal is linear in lagged Gaussian returns, cumulative gross PnL is a Gaussian quadratic form. This gives characteristic functions, cumulants, eigenvalue-based tail diagnostics, and quantile analysis.

Fifth, state the implementation contribution. Transaction costs break the exact-tracking ideal. The paper derives expected turnover and break-even costs, optimizes EWM half-life, and studies no-trade bands, thresholds, position caps, and tail-aware objectives.

## Draft Contribution Statement

A suitable contribution paragraph for the paper is:

> This paper makes four contributions. First, it derives the exact discrete-time OU return autocovariance and shows that an EWM fair-value estimate converts the mean-reversion signal into a lagged linear filter of asset returns. Second, it represents cumulative gross PnL as a Gaussian quadratic form, yielding characteristic functions, cumulants, and eigenvalue-based diagnostics for skewness, kurtosis, quantiles, and tail asymmetry. Third, it extends the analysis to transaction costs by deriving expected turnover, net expected PnL, break-even costs, and cost-adjusted optimal EWM half-life. Fourth, it studies practical implementation rules, including no-trade bands, signal thresholds, position caps, and tail-aware objectives that control left-tail risk.

## Boundaries and Claims to Avoid

Do not claim to solve the general optimal control problem with proportional transaction costs. The no-trade band analysis should be framed as a practical reduced-form implementation, motivated by the transaction-cost literature.

Do not claim that Gaussian OU is empirically complete. The model is a tractable analytical benchmark. The empirical or calibrated section should test how sensitive conclusions are to parameter choice, costs, and tail metrics.

Do not claim that mean-reversion PnL necessarily has many small gains and occasional large losses before Part 7 verifies it through eigenvalues, quantiles, and simulation.

Do not treat calibration as a separate afterthought. Calibration uncertainty should be linked to cost-aware choices of EWM half-life, band width, leverage, and tail constraints.

## Candidate Citation Keys

The paper-specific bibliography has been seeded with keys for the sources above:

- `grebenkov2014trend`
- `garleanu2013dynamic`
- `elliott2005pairs`
- `bertram2010analytic`
- `leung2014optimal`
- `daspremont2011small`
- `davis1990portfolio`
- `shreve1994optimal`
- `imhof1961quadratic`
