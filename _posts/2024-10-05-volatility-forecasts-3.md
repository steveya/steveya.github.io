---
layout: post
title: "Volatility Forecasts (Part 3 - Connection with Neural Network Models)"
#date: 2024-10-05
categories: [Quantitative Finance]
tags: [quantitative-finance, volatility-forecast, machine-learning, research]
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

## Table of Contents

1. [Recap](#recap)
2. [From ES to RNN](#from-es-to-rnn)
3. [From STES to GRU](#from-stes-to-gru)
4. [Results](#results)
5. [Wrapping Up](#wrapping-up)

## Recap
In the [first post](https://steveya.github.io/posts/volatility-forecast-1/) of the [volatility forecasting series](https://steveya.github.io/tags/volatility-forecast/), we introduced the Smooth Transition Exponential Smoothing (STES) model [(Taylor, 2004)](https://doi.org/10.1016/j.ijforecast.2003.09.010) as a generalization of the simpler Exponential Smoothing (ES) model. The smoothing parameter $$\alpha$$ in ES is a constant, but in STES it is generalized to be a linear function of features $$X_t$$ inside the sigmoid function to restrict $$\alpha_t$$ within the interval $$[0, 1]$$.

$$
\begin{equation}
\begin{aligned}
\alpha_t &= \mathrm{sigmoid}(w^T X_t) \\
\sigma_t^2 &= (1 - \alpha_t) \sigma_{t-1}^2 + \alpha_t r_{t-1}^2
\end{aligned}
\end{equation}
$$

where $$r_t^2$$ is the squared return at time $$t$$. This formulation allows the smoothing parameter $$\alpha$$ to be time-varying and be affected by other state variables, yet the model is still simple enough to be interpreted. In our [second post](https://steveya.github.io/posts/volatility-forecast-2/), we generalize the STES model further by introducing the XGBoost-STES model. Instead of using a linear function inside the sigmoid function, we used an Extreme Gradient Boosting (XGBoost) model $$\mathcal{F}$$ to learn the transition function $$f$$ inside the sigmoid function from features $$S_t$$. This allows discontinuity and non-linearity in the smoothing parameter depending on the states variables.

$$
\begin{equation}
\begin{aligned}
\alpha_t &= \mathrm{sigmoid}(\mathcal{F}(X_t)) \\
\sigma_t^2 &= (1 - \alpha_t) \sigma_{t-1}^2 + \alpha_t r_{t-1}^2
\end{aligned}
\end{equation}
$$

In both STES and XGBoost-STES models, the smoothing parameter $$\alpha_t$$ is estimated from the feature vector $$X_t$$, and the volatility is given by

$$\sigma_t^2 = (1 - \alpha_t) \sigma_{t-1}^2 + \alpha_t r_{t-1}^2$$

The STES model generalizes the classic ES model, and XGBoost-STES is one generalization of the STES model that uses a more complex transition function. Another interesting question we can ask is how ES and STES models relate to other modern sequence neural network models. In this post, we will demonstrate that the ES model is a special case of the Recurrent Neural Network (RNN), and the STES model is a special case of the Recurrent Gated Neural Network (GRU). We will then compare the more general and powerful RNN and GRU models to the ES and STES models in terms of their predictive accuracy.

## From ES to RNN
### Recurrent Neural Networks (RNN)
The RNN is a type of neural network commonly used for forecasting sequential data. It is "recurrent" because it performs the same computation for every time step, with the output at each time step fed back as input to the next time step. This allows the network to maintain a form of memory about past inputs, making it suitable for sequential data.

The RNN computes the hidden state $$h_t$$ as a function of the input $$X_t$$ and the previous hidden state $$h_{t-1}$$ such that

$$h_t = f(X_t, h_{t-1})$$

where $$f$$ is often a neural network. For example, a simple RNN may compute the hidden state as:

$$h_t = \text{sigmoid}\left(W_x x_t + W_h h_{t-1} + b_h\right)$$

where $$\text{sigmoid}$$ is the sigmoid activation function. The output $$y_t$$ is computed as a function of the hidden state $$h_t$$:

$$y_t = g(h_t)$$

where $$g$$ can be a simple affine function for regression tasks or a softmax function for classification tasks.

### Exponential Smoothing as a Constrained RNN Model
We cannot help but notice the similarity betweeen the Exponential Smoothing model and the RNN. The ES model can indeed be casted as a linear RNN with constraints on the weight matrix. If we let $$y_t = h_t$$ be the conditional variance $$\sigma_t^2$$ and $$X_t = r_{t}^2$$ be the squared return, and let $$W_h = 1 - W_x$$ be a scalar and $$b_h = 0$$, and use a linear activation function instead of the sigmoid activation function, then the ES model can be written as

$$h_t = W_x X_t + (1 - W_x) h_{t-1}$$

we just need to constrain the scalar $$W_x$$ to be between 0 and 1. Substituting in the $$h_t = \sigma_t^2$$ and $$X_t = r_{t-1}^2$$ into the ES volatility formula, we get

$$\sigma_t^2 = (1 - W_x) \sigma_{t-1}^2 + W_x r_{t-1}^2$$

which is exactly the same formula as in the ES model where $$W_x = \alpha$$. Therefore, another way to generalize the ES model is to simply use a RNN model that replaces the restriction that current forecast $$h_t = \sigma_t^2$$ being a weighted average of current input $$X_t = r_{t-1}^2$$ and past forecast $$h_{t-1} = \sigma_{t-1|^2$$ with a more flexible non-linear function.


## From STES to Gated Recurrent Unit (GRU)
### Gated Recurrent Unit (GRU)
From the above generalization of the ES model to simple RNN, we also noticed the similarity between the STES model and a Gated Recurrent Unit (GRU) model. The GRU model is a type of neural network sequence model that uses a reset gate $$s_t$$ and an update gate $$z_t$$ to control the flow of information into the hidden state $$h_t$$. The network is given by

$$
\begin{equation}
\begin{aligned}
z_t &= \text{sigmoid}\left(W_z x_t + U_z h_{t-1} + b_z\right) \quad \textrm{Update Gate} \\
s_t &= \text{sigmoid}\left(W_r x_t + U_r h_{t-1} + b_r\right) \quad \textrm{Reset Gate}\\
\tilde{h}_t &= \tanh\left(W_h x_t + U_h (s_t \odot h_{t-1}) + b_h\right) \quad \textrm{Candidate Hidden State} \\
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
\alpha_t &= \text{sigmoid}\left(w^T x_t\right) \\
\sigma_t^2 &= \left(1 - \alpha_t\right) \sigma_{t-1}^2 + \alpha_t r_{t-1}^2
\end{aligned}
\end{equation}
$$

To see that the STES is a special case of the GRU model, we can set $$U_z = 0$$, $$r_t = 1 \forall t$$, $$W_h = b_h = 0$$. We also replace the $$\tanh$$ with a linear activation function. Substituting these into the GRU model and simplifying, we get

$$
\begin{equation}
\begin{aligned}
z_t &= \text{sigmoid}(W_z x_t + b_z) \\
s_t &= 1 \\
\tilde{h}_t &= h_{t-1} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \\
\end{aligned}
\end{equation}
$$

which is exactly the same as the STES model. In other words, the STES model is a special case of the GRU model where one never resets the candidate hidden state, which is set simply to the previous hidden state.


