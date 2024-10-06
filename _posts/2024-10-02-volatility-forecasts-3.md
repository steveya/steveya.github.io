---
layout: post
title: "WIP - Volatility Forecasts (Part 3 - Connection with Neural Network Models)"
date: 2024-10-02
categories: [Quantitative Finance]
tags: [quantitative-finance, volatility-forecast, machine-learning, research]
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## Table of Contents

1. [Recap](#recap)
2. [From ES to Simple RNN](#from-es-to-simple-rnn)
3. [From STES to Gated Recurrent Unit (GRU)](#from-stes-to-gated-recurrent-unit-gru)
4. [Results](#results)
5. [Wrapping Up](#wrapping-up)

## Recap
In the [first post]({% post_url 2024-07-12-volatility-forecasts-1 %}) of our [volatility forecasts series]({% post_url 2024-07-12-volatility-forecasts-1 %}), we introduced the Smooth Transition Exponential Smoothing (STES) model [(Taylor, 2004)](https://doi.org/10.1016/j.ijforecast.2003.09.010) as a generalization of the simpler Exponential Smoothing (ES) model. The smoothing parameter $$\alpha$$ in ES is a constant through time, but in STES $$\alpha$$ becomes $$\alpha_t$$ and is computed from a logistic regression using features $$X_t$$.

$$
\begin{equation}
\begin{aligned}
\alpha_t &= \mathrm{sigmoid}(w^T X_t + b) \\
\sigma_t^2 &= (1 - \alpha_t) \sigma_{t-1}^2 + \alpha_t r_{t-1}^2
\end{aligned}
\end{equation}
$$

where $$r_t^2$$ is the squared return at time $$t$$. This formulation allows the smoothing parameter $$\alpha_t$$ to be time-varying and be affected by other variables, yet the model is still simple enough to be interpreted. 

In our [second post]({% post_url 2024-07-18-volatility-forecasts-2 %}), we generalize the STES model further by introducing the XGBoost-STES model. Instead of using an affine function inside the sigmoid function, we used an Extreme Gradient Boosting (XGBoost) model $$\mathcal{F}$$ inside the sigmoid function. This allows discontinuity and non-linearity in the smoothing parameter depending on the states variables.

$$
\begin{equation}
\begin{aligned}
\alpha_t &= \mathrm{sigmoid}(\mathcal{F}(X_t)) \\
\sigma_t^2 &= (1 - \alpha_t) \sigma_{t-1}^2 + \alpha_t r_{t-1}^2
\end{aligned}
\end{equation}
$$

In both STES and XGBoost-STES models, the smoothing parameter $$\alpha_t$$ is estimated from the feature vector $$X_t$$, and the volatility is then given recursively by

$$\sigma_t^2 = (1 - \alpha_t) \sigma_{t-1}^2 + \alpha_t r_{t-1}^2$$

Given the recursive nature of this type of model, an interesting question we can ask is how ES and STES models relate to other modern recursive sequence models. In this post, we show that the ES model is a special case of the Recurrent Neural Network (RNN) and the STES model a special case of the Recurrent Gated Neural Network (GRU). We will then implement both neural network models and compare their performance relative to the ES and STES.

## From ES to Simple RNN
### Recurrent Neural Networks (RNN)
The RNN is a type of neural network commonly used for forecasting sequential data. It is "recurrent" because it performs the same computation for every time step, with the output at each time step fed back as input to the next time step. This allows the network to maintain a form of memory about past inputs, making it suitable for sequential data.

The RNN computes the hidden state $$h_t$$ as a function of the input $$X_t$$ and the previous hidden state $$h_{t-1}$$ such that

$$h_t = f(X_t, h_{t-1})$$

where $$f$$ is often a neural network. For example, a simple 1-layer RNN may compute the hidden state as:

$$h_t = \mathrm{sigmoid}\left(W_x X_t + W_h h_{t-1} + b_h\right)$$

Its output $$y_t$$ is then computed as a function of the hidden state $$h_t$$:

$$y_t = g(h_t)$$

where $$g$$ can be a simple affine function for regression tasks or a softmax function for classification tasks.

### Exponential Smoothing as a Constrained 1-layer RNN Model
We cannot help but notice the similarity betweeen the ES and the RNN. Indeed, the ES model can be casted as a linear RNN with constraints on the weight matrix. Let $$y_t = h_t$$ be the conditional variance $$\sigma_t^2$$ and $$X_t = r_{t}^2$$ be the squared return, and let $$W_h = 1 - W_x$$ be a scalar and $$b_h = 0$$, using a linear activation function instead of the sigmoid function, the constrained 1-layer RNN model simplifies to

$$\sigma_t^2 = (1 - W_x) \sigma_{t-1}^2 + W_x r_{t-1}^2$$

With the additional constraint that the scalar $$W_x$$ is between 0 and 1, this gives the same formula as the ES model. 

Therefore, one other way to generalize the ES model is to simply use a 1-layer RNN model learned using features $$X_t$$. It should also be clear that this generalization is different from STES. Indeed, the STES can be generalized in a similar fashion, as we will see below.

## From STES to Gated Recurrent Unit (GRU)
### Gated Recurrent Unit (GRU)
From the above generalization of the ES to simple RNN, we also noticed the similarity between the STES model and the Gated Recurrent Unit (GRU) model. The GRU model is a type of neural network sequence model that employs a reset gate $$s_t$$ and an update gate $$z_t$$ to control the flow of information into the hidden state $$h_t$$. It is given by

$$
\begin{equation}
\begin{aligned}
z_t &= \mathrm{sigmoid}\left(W_z X_t + U_z h_{t-1} + b_z\right) \quad \textrm{Update Gate} \\
s_t &= \mathrm{sigmoid}\left(W_s X_t + U_s h_{t-1} + b_s\right) \quad \textrm{Reset Gate}\\
\tilde{h}_t &= \tanh\left(W_h X_t + U_h (s_t \odot h_{t-1}) + b_h\right) \quad \textrm{Candidate Hidden State} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \textrm{Hidden State} \\
\end{aligned}
\end{equation}
$$

where $$\odot$$ is the element-wise product and $$\tilde{h}_t$$ is the candidate hidden state. 

### STES as a Special Case of GRU
Recall that the STES model is given by

$$
\begin{equation}
\begin{aligned}
\alpha_t &= \text{sigmoid}\left(w^T X_t + b\right) \\
\sigma_t^2 &= \left(1 - \alpha_t\right) \sigma_{t-1}^2 + \alpha_t r_{t-1}^2
\end{aligned}
\end{equation}
$$

To see that the STES is a special case of the GRU model, let

$$
X_t = 
\left[
  \begin{array}{cccc}
    \mid    & \mid &        & \mid \\
    r_{t-1}^2 & x_2    & \ldots & x_n    \\
    \mid    & \mid &        & \mid \\
  \end{array}
\right]
$$

be ordered such that $$r_{t-1}^2$$ is the first column of $$X_t$$. We assume that $$r_{t-1}^2$$ is also a feature. It does not need to be a feature, in which case additional restrictions need to place on $$W_z$$.

Let us set $$U_z = 0$$, $$W_h = \left[1, 0, \hdots, 0\right]$$, $$W_h = $$, $$s_t = 0$$ and $$b_h = 0$$, and replace the $$\tanh$$ with a linear activation function. 

$$
\begin{equation}
\begin{aligned}
z_t &= \text{sigmoid}(W_z X_t + b_z) \\
s_t &= 0 \\
\tilde{h}_t &= r_{t-1}^2 \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \\
\end{aligned}
\end{equation}
$$

Just as we did in the simple RNN above, let $$h_t = \sigma_t^2$$. Substituting these into the GRU model and simplifying, we get

$$
\begin{equation}
\begin{aligned}
z_t &= \text{sigmoid}(W_z X_t + b_z) \\
\sigma_t^2 &= \left(1 - z_t\right) \sigma_{t-1}^2 + z_t r_{t-1}^2
\end{aligned}
\end{equation}
$$

which is exactly the same as the STES model. In other words, the STES model is a special case of the GRU model in which one always resets the candidate hidden state to be just the previous period's squared return.



## Results

## Wrapping Up

