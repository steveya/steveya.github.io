---
layout: post
title: Signature Methods (Part 2 - Definition)
date: 2025-12-20
categories: [Quantitative Finance]
tags: [study-notes, signature-method, quantitative-finance, volatility-forecast, machine-learning, research]
math: true
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## Table of Content

1. [Motivation for the Signature Method](#motivation)
2. [Definition](#definition)

## Motivation for the Signature Method
In quantitative finance we often deal with trajectories: prices evolving, spreads widening then snapping back, volume or volatility arriving in bursts, and signals activated in sequences. However, many modeling pipelines force trajectories into a small set of hand-crafted summaries such as moving averages and volatility, cross-overs, rolling betas, event counts or regime flags. These features are often effective, but they mostly work by compressing a path into a number, and any information about ordering and interactions across variables (channels) is mostly lost. That loss matters because many market phenomena are genuinely path-dependent.

For example, volatility clustering depends on the recent sequence of shocks, not just the net move. We are often interested in "who moved first" (causal effect) by looking at the lead–lag and cross-impact. Intraday microstructure effects such as bid–ask bounce, spread dynamics etc, depend on ordering as well. These are often difficult or ineffective to be compressed into a number.

The signature method starts from a different premise: instead of choosing a small list of summaries, we build a systematic feature map for paths. The goal is to represent a multivariate time series window as a feature vector that retains time-ordered interactions, so that downstream models can learn path-dependence without requiring us to manually enumerate and craft interaction patterns.

Another useful way to motivate the method before defining it is to view it as a path analogue of polynomial feature maps.

For scalar inputs $$x$$, we can use the polynomial features

$$
\phi(x) = (1, x, x^2, x^3, \dots)
$$

to approximate many nonlinear targets $$f(x)$$ using a linear model in the expanded feature space:

$$
f(x) \approx \langle w,\ \phi^{\le m}(x)\rangle.
$$

For sequential inputs $$X_{[a, b]} = \left\{X_t \vert t \in [a, b], X_t \in \mathbb{R}^d \right\}$$, we seek an analogous map from a window of a multivariate time series (a path) to a feature vector, such that many continuous, path-dependent targets can be approximated by linear functionals of those features:

$$
F(X_{[a,b]}) \approx \langle \ell,\ S^{\le m}(X_{[a,b]})\rangle.
$$

The signature method provides precisely such features $$S(X_{[a,b]})$$. More precisely, the signature has the following universal approximation property:

Let $$\mathcal{P}$$ be a set of $$d$$-dimensional paths on $$[a,b]$$ (typically of bounded variation), and let $$S(X_{[a,b]})$$ denote the signature of a path $$X$$. Fix a compact subset $$K\subset\mathcal{P}$$ (compact in a topology where the signature map is continuous). For any continuous functional $$F:K\to\mathbb{R}$$ and any $$\varepsilon>0$$, there exist:

- a truncation depth $$m$$, and
- a coefficient vector for signature coordinates up to level $$m$$)

such that
$$
\sup_{X\in K}\left|F(X)-\langle \ell,\ S^{\le m}(X)\rangle\right|<\varepsilon.
$$

We have to make these design choices to use signature features effectively in practice.

1. **Path Construction**: Signatures operate on a path of inputs. In order to fully leverag the universality property of signatures, the inputs should contain sufficient amount of information. This is done through various augmentations such as time augmentation, lead–lag augmentation, inclusion of macro or microstructure channels)etc. This is where we decide which invariances and which statistics we want to make easy to learn.
2. **Truncation and Rgularization**: The full signature is an infinite collection of features. In practice we truncate at depth $$m$$ and rely on regularization and model selection to control complexity.

In the next post we will define signatures explicitly and provide a few examples

## Definition

Fix a time window $$t \in [a,b]$$ and choose $$d$$ input series. Collect them into a single multivariate process

$$
X_t = \big(X_t^1, X_t^2, \dots, X_t^d\big), \qquad t\in[a,b],
$$

where each component $$X^i$$ is one time series we include (e.g., time, log-price or cumulative return, cumulative volume, spread, an alpha signal, etc.).

For any ordered index sequence

$$
(i_1, i_2, \dots, i_k), \qquad i_j \in {1,\dots,d},
$$

define the corresponding order-$$k$$ signature coordinate as the iterated integral

$$
S^{i_1,\dots,i_k}(X)_{a,b}
\int_{a<t_1<\cdots<t_k<b}
\mathrm{d}X^{i_1}{t_1}\cdots \mathrm{d}X^{i_k}{t_k}.
$$

The constraint $$a<t_1<\cdots<t_k<b$$ enforces time ordering. As a result, terms such as $$S^{(i,j)}$$ and $$S^{(j,i)}$$ generally differ: they represent different ordered interactions between components $$i$$ and $$j$$ over the window.

If the window is sampled at $$a=t_0<t_1<\cdots<t_N=b$$ and $$\Delta X^i_n = X^i_{t_n}-X^i_{t_{n-1}}$$, then low-order signature terms can be viewed as time-ordered sums of products of increments. In particular,

$$
S^{(i)}(X){a,b} \approx \sum{n=1}^N \Delta X^i_n = X^i_b - X^i_a,
$$

and

$$
S^{(i,j)}(X){a,b} \approx \sum{1\le p<q\le N} \Delta X^i_p,\Delta X^j_q.
$$

This makes explicit that order-2 terms are directional: they aggregate “moves in series $$i$$ occurring earlier” times “moves in series $$j$$ occurring later.”

The signature of $$X$$ on $$[a,b]$$ is the collection of all such coordinates across all orders:

$$
S(X){a,b} = \big(1,\ S^{(1)}(X){a,b},\ S^{(2)}(X)_{a,b},\ \dots\big),
$$

where $$S^{(k)}(X)_{a,b}$$ denotes the vector of all order-$$k$$ coordinates (all ordered index sequences of length $$k$$).

In practice we work with the truncated signature up to depth $$m$$:

$$
S^{\le m}(X)_{a,b}.
$$

If the path has dimension $$d$$, the number of coordinates up to depth $$m$$ is

$$
1 + d + d^2 + \cdots + d^m = \frac{d^{m+1}-1}{d-1}.
$$

Because this grows quickly, applied work typically controls the number of input series $$d$$, the truncation depth $$m$$, and (later in the series) considers log-signatures to reduce redundancy.

A useful way to interpret the order-$$k$$ coordinate $$S^{i_1,\dots,i_k}(X){a,b}$$ is as a summary of how often and how strongly a particular ordered pattern of moves occurs across the window. In discrete time it behaves like an aggregate of products $$\Delta X^{i_1}{p_1}\cdots\Delta X^{i_k}_{p_k}$$ over all strictly increasing index tuples $$p_1<\cdots<p_k$$. This time-ordering is what makes signatures fundamentally different from symmetric moment features: changing the order of the indices generally changes the statistic, because it corresponds to a different “move-then-move” template.

Here we compute and interpret the first two levels of signature of some market objects.

### Level 1: net changes (what happened overall)

For channel $$i$$,

$$
S^{(i)}(X)_{a,b} = \int_a^b \mathrm{d}X^i_t = X^i_b - X^i_a.
$$

So level-1 terms are cumulative moves:
- cumulative return if $$X^i$$ is log-price
- cumulative volume if $$X^i$$ is cumulative volume
- net change in spread, net change in signal, etc.

This is similar to what classical factor models and linear predictors “see.”

### Level 2: ordered interactions (who moved first)

Now we get true path-dependence:

$$
S^{(i,j)}(X){a,b} = \int{a<t_1<t_2<b} \mathrm{d}X^i_{t_1}, \mathrm{d}X^j_{t_2}.
$$

Financial reading:
- $$S^{(\text{signal},\text{return})}$$ captures whether signal changes early in the window are followed by returns later (a lead–lag-aware relation)
- $$S^{(\text{spread},\text{volume})}$$ captures whether widening spreads tend to be followed by volume (or vice versa)
- for multi-asset paths, cross terms encode ordering in co-moves

In other words: level-2 terms start to look like directional interaction summaries rather than symmetric correlations.

From a modeling perspective, the signature provides a structured hierarchy of features: level 1 captures net changes, level 2 captures ordered pairwise interactions, and higher levels capture longer sequencing effects. Truncating at depth $$m$$ then amounts to assuming that most of the relevant path-dependence can be expressed using interaction templates of length at most $$m$$—an assumption that is often reasonable in practice when paired with regularization.


## Augmentation

A practical way to use the “universality” result is to treat it as universality relative to our path representation. Start from the statistic or target $$F$$ we care about and ask: what information must the path encode so that $$F$$ is a stable functional of that path? Level-1 signature terms capture net changes of whatever we include (returns, spreads, signals, time). If our target depends on quadratic variation / volatility or second-order cross-products (e.g., $$\sum r^2$$, rolling covariance, beta, correlation), then we should include an augmentation (most commonly lead–lag) that promotes these increment-level second moments into level-2 signature coordinates. If our target depends on normalization by time or sample count (irregular sampling), include an explicit time / event-count component. After that, we can either (i) compute the classic rolling statistic from these low-order signature building blocks, or (ii) let a regularized linear model on truncated signatures learn a more general path-dependent predictor.

In practice, “augmentation” means adding extra channels or changing the embedding of the raw series into a path so that the signature captures the aspects of the data we need. The most common augmentations used in financial applications are:
	
- Basepoint (anchoring the window). Include an explicit start point so that the representation distinguishes “the same increments but different starting level” when that matters. Operationally, you embed each window as a path starting at a fixed basepoint (often zero) and then follow the observed trajectory. This is mainly a bookkeeping device, but it makes statements like “the path on $$[t-L,t]$$” unambiguous when you concatenate windows and apply Chen’s identity later.
- Time / event-count augmentation. Add a monotonically increasing channel $$t$$ (calendar time) or $$n$$ (event count). This stabilizes the representation under irregular sampling and gives the signature direct access to time-weighted effects and integrals. For example, for the 2D path $$X_t=(t,x_t)$$ one can express window averages using low-order signature terms (bounded-variation setting):
$$\int_a^b x_t,dt = (b-a)x_b - S^{(t,x)}(X)_{a,b},$$
so time augmentation is a natural choice when your target depends on “average level,” time-in-window normalization, or when the sampling grid itself is informative.
- Lead–lag augmentation (quadratic variation and cross-variation at low depth). Replace a stream by a higher-dimensional path that separates the “current” value from a lagged copy, producing an L-shaped move at each step. The key consequence is that second-order increment quantities become accessible at low depth. For a 1D log-price $$x$$ with returns $$r_n=x_n-x_{n-1}$$ and lead–lag path $$Z=\mathrm{LL}(x)\in\mathbb{R}^2$$, one obtains a depth-2 identity of the form
$$S^{(1,2)}(Z){0,N}-S^{(2,1)}(Z){0,N}=\sum_{n=1}^N r_n^2,$$
(up to a sign convention). In the multivariate case, the same construction promotes cross-products $$\sum r^i r^j$$ into depth-2 coordinates, which is why lead–lag is the default augmentation when the target involves volatility, covariance, beta, correlation, or other second-moment objects.
- Cumulative (“integrated”) channels vs increment channels. In finance, many raw series arrive as increments (returns, changes in yields, flow shocks). Signatures are defined on paths, so you typically embed increments by forming a cumulative level process
$$x_n = x_0 + \sum_{k=1}^n r_k.$$
This is not cosmetic: the signature features are iterated integrals of $$dx$$, so making explicit what counts as a level and what counts as an increment clarifies what the low-order terms represent (net change, time-weighted change, ordered interactions, etc.).
- Multiple channels and “interaction intent.” Adding channels is not just “more predictors.” It sets the interaction vocabulary the signature can express. If you include (say) $$x$$ (log-price), $$v$$ (volume), $$s$$ (spread), and a macro surprise index $$m$$ as channels, then depth-2 signature coordinates correspond to ordered co-movements such as “$$x$$ moves before $$v$$” or “$$m$$ moves before $$x$$,” and depth-3 coordinates correspond to three-way sequencing patterns. This is the mechanism by which signatures replace hand-coded interaction features.
- Normalization / scaling channels (conditioning and comparability). Many financial channels have incompatible units or scale drift across regimes. It is often useful to include a normalization channel explicitly (e.g., a running volatility estimate, an ATR-like scale, or an exposure measure) so that the path representation is stable across periods. Conceptually: you are deciding whether the signature should learn “raw magnitude effects” or “scale-free pattern effects.”
- Piecewise-constant vs piecewise-linear interpolation choice. Discrete data must be embedded into a continuous path. Piecewise-linear is the standard default for bounded-variation signatures; piecewise-constant (or other conventions) may be appropriate for event streams (quotes/trades) where “holding the last value” matches the economics. This choice changes what the iterated integrals measure, so it should be treated as part of the model design.

The universal approximation property can be read as follows: once the path representation contains the information the target depends on (via appropriate augmentation), linear functionals of truncated signatures can approximate a broad class of continuous path-dependent maps. In finance, many rolling statistics and economically meaningful targets depend on first and second moments, time normalization, and ordered interactions; augmentations such as time and lead–lag make these ingredients appear in low-order signature coordinates, either to compute the statistic directly or to serve as stable building blocks for learned predictors.


## Universal approximation in practice: two worked examples

This section gives two worked examples that make the universal approximation statement concrete for a quant audience. The common template is:
	1.	Choose a path representation / augmentation so that an informative scalar summary $$q(X)$$ becomes a linear functional of low-order signature coordinates.
	2.	Choose a continuous function $$g$$ and approximate it on a compact interval by a polynomial (Chebyshev is a convenient choice).
	3.	Use the shuffle product (algebra of signatures) to convert powers of $$q(X)$$ into a linear functional of higher-order signature coordinates.

The outcome is an explicit approximation

$$
F(X) \approx \langle \ell,\ S^{\le m}(X)\rangle,
$$

with an error bound controlled by the polynomial approximation error and by restricting to a compact set where the functional is well-behaved.

⸻

Example 1: standard deviation of increments (volatility as a path functional)

Goal. Given daily log-prices $$x_0,\dots,x_N$$ on a fixed window, define returns (increments)

$$
r_n := x_n - x_{n-1},\qquad n=1,\dots,N.
$$

The (unbiased) sample standard deviation is

$$
\mathrm{Std}(r)
:=
\sqrt{\frac{1}{N-1}\sum_{n=1}^N (r_n-\bar r)^2},
\qquad
\bar r := \frac{1}{N}\sum_{n=1}^N r_n.
$$

This is a continuous functional of the increment sequence on any set where the variance is bounded away from 0.

Step 1: lead–lag augmentation
A plain 1D signature depends only on the total increment and cannot distinguish different sequences with the same net move. To expose second-moment information, form the lead–lag path

$$
Z := \mathrm{LL}(x)\in\mathbb{R}^2,
$$

constructed from the level sequence $$x_0,\dots,x_N$$ by alternating horizontal/vertical moves so that each increment contributes an “L-shaped” step.

Step 2: variance numerator as a linear functional of depth-2 signatures
Define the sample variance numerator

$$
q
:=
\sum_{n=1}^N (r_n-\bar r)^2

\sum_{n=1}^N r_n^2 - \frac{\big(\sum_{n=1}^N r_n\big)^2}{N}.
$$

On the lead–lag path $$Z$$, the following depth-2 identity holds:

$$
\sum_{n=1}^N r_n^2

S^{(1,2)}(Z){0,N} - S^{(2,1)}(Z){0,N}.
$$

Also, since $$\sum_{n=1}^N r_n = x_N-x_0$$ is the total increment, we have

$$
\big(\sum_{n=1}^N r_n\big)^2 = 2,S^{(1,1)}(Z)_{0,N}.
$$

Therefore

$$
q

\Big(S^{(1,2)}(Z){0,N} - S^{(2,1)}(Z){0,N}\Big)

\frac{2}{N}S^{(1,1)}(Z)_{0,N}.
$$

Equivalently, there exists a coefficient object $$a$$ supported only on words of length 2 such that

$$
q = \langle a,\ S^{\le 2}(Z)\rangle.
$$

Finally,

$$
\operatorname{Std}(r) = \sqrt{\frac{q}{N-1}}.
$$

Step 3: Chebyshev approximation of the square-root
Fix a compact interval $$[q_{\min},q_{\max}]$$ with $$q_{\min}>0$$ (in practice: restrict to windows where variance does not vanish, or add a small ridge $$q\mapsto q+\varepsilon$$).

Define

$$
f(q) := \sqrt{\frac{q}{N-1}}.
$$

Let $$p_M(q)=\sum_{m=0}^M c_m q^m$$ be a degree-$$M$$ polynomial (e.g., Chebyshev minimax) such that

$$
\sup_{q\in[q_{\min},q_{\max}]} |f(q) - p_M(q)| \le \varepsilon_M,
\qquad \varepsilon_M\to 0\ \text{as}\ M\to\infty.
$$

Step 4: convert the polynomial into a linear functional on signatures (shuffle powers)
Because $$q(Z)=\langle a,S(Z)\rangle$$ is linear in signature coordinates, the shuffle identity implies

$$
q(Z)^m = \langle a^{ \,\text{⧢}\, m},\ S(Z)\rangle,
$$

where $$a^{\,\text{⧢}\, m}$$ is the $$m$$-fold shuffle power of $$a$$.

Therefore

$$
p_M(q(Z))

\sum_{m=0}^M c_m q(Z)^m

\Big\langle \sum_{m=0}^M c_m a^{\,\text{⧢}\, m},\ S(Z)\Big\rangle.
$$

Define

$$
\ell_M := \sum_{m=0}^M c_m a^{\,\text{⧢}\, m}.
$$

Since $$a$$ lives at depth $$\le 2$$, $$\ell_M$$ lives at depth $$\le 2M$$. Hence we obtain the explicit approximation

$$
\operatorname{Std}(r)
\approx
\langle \ell_M,\ S^{\le 2M}(Z)\rangle,
$$

and the uniform error bound (on $$q\in[q_{\min},q_{\max}]$$)

$$
\sup_{q\in[q_{\min},q_{\max}]}
\Big|
\sqrt{\frac{q}{N-1}} - \langle \ell_M,\ S^{\le 2M}(Z)\rangle
\Big|
\le
\varepsilon_M.
$$

Interpretation. Lead–lag makes the variance numerator a depth-2 linear signature statistic. Chebyshev + shuffle then turns the square-root nonlinearity into a linear functional on higher-depth signatures.

⸻

Example 2: average level on a window and Asian-style (or smooth) payoffs

This example illustrates a different augmentation: time augmentation. It produces a low-order signature representation of the time integral of a level process, which is a standard object in both derivatives and systematic rules.

Goal. For a (log-)price path $$x_t$$ on $$[a,b]$$ define the time-average

$$
A(x) := \frac{1}{b-a}\int_a^b x_t,dt.
$$

Let $$g: \mathbb{R}\to\mathbb{R}$$ be a continuous function (examples: soft threshold, logistic, exponential utility, or the continuous-but-nonsmooth Asian payoff $$g(u)=(u-K)_+$$).

Define the path functional

$$
F(x) := g(A(x)).
$$

Step 1: time augmentation
Form the 2D path

$$
X_t := (t,\ x_t)\in\mathbb{R}^2,\qquad t\in[a,b].
$$

Assume bounded variation (piecewise linear interpolation of sampled data suffices).

Step 2: express the time integral using depth-2 signature coordinates
Integration by parts gives

$$
\int_a^b x_t,dt = (b-a)x_b - \int_a^b (t-a),dx_t.
$$

A standard identity for bounded-variation paths relates the remaining term to the depth-2 signature coordinate:

$$
S^{(t,x)}(X){a,b}
:=
\int{a<u<v<b} dt_u,dx_v

\int_a^b (t-a),dx_t.
$$

Therefore

$$
\int_a^b x_t,dt

(b-a)x_b - S^{(t,x)}(X)_{a,b}.
$$

If $$b-a$$ is fixed (rolling windows of fixed length), then $$A(x)$$ is an affine function of low-order signature terms:

$$
A(x)

\frac{1}{b-a}\Big((b-a)x_b - S^{(t,x)}(X)_{a,b}\Big)

x_b - \frac{1}{b-a}S^{(t,x)}(X)_{a,b}.
$$

Since $$x_b = x_a + S^{(x)}(X)_{a,b}$$ is obtained from the level-1 term, we can write

$$
A(x) = \text{(constant)} + \alpha,S^{(x)}(X){a,b} + \beta,S^{(t,x)}(X){a,b}.
$$

Equivalently, there exists $$a$$ supported on depths $$\le 2$$ such that

$$
A(x) = \langle a,\ S^{\le 2}(X)\rangle.
$$

Step 3: Chebyshev approximation of $$g$$ on a compact interval
Restrict attention to paths for which $$A(x)\in[L,U]$$ (compact). Then choose a degree-$$M$$ polynomial

$$
p_M(u)=\sum_{m=0}^M c_m u^m
$$

such that

$$
\sup_{u\in[L,U]} |g(u) - p_M(u)| \le \varepsilon_M.
$$

Step 4: convert $$p_M(A(x))$$ into a linear functional on signatures
Since $$A(x)=\langle a,S(X)\rangle$$ is linear in signature coordinates, the shuffle identity yields

$$
A(x)^m = \langle a^{\,\text{⧢}\, m},\ S(X)\rangle.
$$

Hence

$$
F(x)=g(A(x)) \approx p_M(A(x))

\sum_{m=0}^M c_m A(x)^m

\Big\langle \sum_{m=0}^M c_m a^{\,\text{⧢}\, m},\ S(X)\Big\rangle.
$$

Defining

$$
\ell_M := \sum_{m=0}^M c_m a^{\,\text{⧢}\, m},
$$

and noting that $$\ell_M$$ lives at depth $$\le 2M$$, we obtain the explicit approximation

$$
F(x)=g(A(x))
\approx
\langle \ell_M,\ S^{\le 2M}(X)\rangle,
$$

with uniform error bound

$$
\sup_{u\in[L,U]} |g(u) - p_M(u)| \le \varepsilon_M.
$$

Interpretation. Time augmentation promotes the window-average level into depth-2 signature coordinates. Polynomial approximation + shuffle then produces explicit linear-on-signature approximations for a wide class of continuous payoff functions of the average.

⸻

Remarks on “constructive” vs “learned” coefficients
The two examples above produce explicit coefficients $$\ell_M$$ by combining (i) polynomial coefficients $$c_m$$ and (ii) shuffle powers $$a^{\,\text{⧢}\, m}$$. In practice one often learns coefficients by least squares or regularized regression on signature features. The constructive route is mainly pedagogical: it makes the universal approximation mechanism tangible and shows how nonlinearity can be represented by linear functionals on higher-order signature terms once the path is augmented appropriately.


### Brief note on log-signatures (motivation)

One drawback of working directly with truncated signatures is redundancy: the raw signature coordinates satisfy many algebraic relations, and the feature dimension grows rapidly with depth. The log-signature addresses both issues by taking a “logarithm” of the signature in the tensor algebra, producing a representation that lives in the associated free Lie algebra. Informally, we can think of the log-signature as retaining the same information as the signature (up to truncation) but expressed in a basis of non-redundant interaction primitives, typically with substantially fewer coordinates and better numerical behavior. In practice, log-signatures are often preferred when we want higher depth without an explosion in dimension, or when we want a representation whose coordinates correspond more directly to “independent” interaction effects.

3. How signatures are computed from discrete market data

Market data arrives discretely: $${(t_j, X_{t_j})}_{j=0}^N$$. To compute signatures, we first embed it as a continuous path. The simplest practical choice is piecewise linear interpolation between observation times.

So the workflow is:
	1.	choose channels (e.g., log-price, volume, spread, signal, time)
	2.	build a continuous path via linear interpolation
	3.	compute $$S^{\le m}(X)_{a,b}$$ (typically using a library)

A crucial caution: 1D signatures are basically a trap

If our path is one-dimensional, the signature contains no shape information beyond the net increment. Intuitively, a 1D path cannot encode ordering interactions, because there is only one channel.

Finance translation:
	•	univariate signatures of price alone are typically not useful
	•	signatures become interesting when we include multiple channels or use augmentations (time, lead–lag, multi-asset paths)

This is one reason volatility forecasting is a natural fit: we can design the path so that its low-order terms encode “variance-like” objects.

4. What signatures mean for financial time series


5. Why this helps learning (without deep nets)

The practical modeling move is:

compute signature features, then fit a linear model (or a simple nonlinearity) on those features.

This is powerful because linear functionals of signatures can approximate broad classes of continuous path-dependent functionals:

$$
\phi(X) \approx \langle \ell,\ S(X)\rangle.
$$

So we get expressiveness while keeping a linear handle: interpretability, regularization, stability selection, and easy integration into existing ML stacks.

6. When we expect signatures to add value in finance

Signatures tend to help most when the predictive content depends on:
	1.	ordering (lead–lag, “shock then rebound,” confirmation patterns)
	2.	interactions across channels (price/volume/spread/signal)
	3.	path-dependent state (memory beyond a few lags)
	4.	irregular sampling (where naively lagging can misalign information)

If our feature set is already a near-sufficient statistic for our target, signatures won’t magically add alpha. But when our edge is about the shape of the recent market path, they provide a principled way to represent that shape.

7. Transition: why volatility forecasting is a perfect first application

Volatility forecasting is an unusually clean place to start because:
	•	volatility is inherently a path property (it’s about variation along the path, not just endpoints)
	•	high-frequency data makes “sequence and interaction” effects visible (clustering, microstructure, intraday regimes)
	•	signatures provide a natural way to encode variance-like information through multichannel embeddings

In Part 2, we’ll build the volatility-forecasting path representation explicitly.

We’ll cover:
	•	how to embed returns into a multichannel path (time augmentation, scaling)
	•	the lead–lag transform and why it exposes quadratic-variation-like information to low-order signature terms
	•	practical windowing: mapping each past window $$[t-L, t]$$ to a feature vector $$S^{\le m}(X)_{t-L,t}$$
	•	how to train a forecaster: $$\widehat{\sigma}{t+h} = f\big(S^{\le m}(X){t-L,t}\big)$$

If we already have a volatility-forecasting blog post, Part 2 is where we’ll merge: we’ll take our existing target definition (e.g., next-day realized volatility, Parkinson/Garman–Klass, RV from intraday returns, etc.) and show exactly how the signature representation plugs into it.