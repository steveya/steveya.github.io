---
title: "WIP - Roll Your Own Nowcast(s) (Part 1)"
date: 2023-07-20
categories: [General, Welcome]
tags: [nowcasting, tutorial]
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Work in Progress: Roll Your Own Nowcast (Part 1)

## Table of Contents

1. [Introduction](#introduction)
2. [Preliminaries](#preliminaries)
3. [Data Preparation](#data-preparation)
4. [Model Specification](#model-specification)

## Introduction

In this post, I will walk through the process of building one's own nowcasting models. I will follow closely the steps of the New York Fed's Nowcasting methodology, given that they already open-sourced their code on Github. However, the code is written in Matlab and is messy. Several other repos translate it to Python or simplify it to handle higher frequency data (with trade off in model complexity)

The goal of this series of posts is to implement a nowcasting model from scratch, with mathematical derivations and code displayed side-by-side. There are plenty of research in the nowcasting literature, and there are quite a few repos with implementations, but there is a lack of tutorials on how to go from the mathematical derivation of the model and the actual implementation. This series aim to fill that gap.

## Preliminaries
