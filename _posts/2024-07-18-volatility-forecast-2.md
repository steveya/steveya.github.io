---
layout: post
title: "(WIP) Extending Smooth Transitioning Exponential Smoothing Volatility Forecasts (Part 2 - XGBoost)"
date: 2024-07-18
tags: [quantitative-finance, volatility-forecast, machine-learning, research]
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>


# XGBoost-STES
In the [previous post](https://steveya.github.io/posts/volatility-forecast-1/), I replicated the results of the Taylor 2020 paper on STES and the ES model.

Now I implemented the STES model using XGBoost. The XGBoost model is a random forest regressor that uses the exponential loss function to measure the error of the model. It is a popular model in the machine learning community and is used in many applications. XGBoost is an extremely fast and efficient model that can handle large datasets and complex models. It is also easy to implement and use.
