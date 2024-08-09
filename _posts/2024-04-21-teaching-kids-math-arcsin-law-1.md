---
layout: post
title: Teaching Kids Math - Arcsin Law (Part 1 - Inutition)
date: 2024-04-21
categories: [Teaching Kids Math]
tags: [teaching-kids-math]
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Teaching Kids Math Series
Since my first child, I have been learning how to teach children about mathematics. In particular, mathematics that I did not learn about at school when I was a child, but I knew I would have grasped it at that age had someone introduced me to it. I started a video series to put them on YouTube to share with others. This is also a learning experience because I still need to work on introducing my chosen concepts, assuming only grade school mathematics, and learning how to make YouTube videos. I also only have a little time; the children are growing fast...

I decided to start by making videos using the Manim library (from [3Blue1Brown](https://www.youtube.com/@3blue1brown)) since it is how I learn a lot of math topics on YouTube. I am still on the fence about using it for this purpose because I would be using an unfamiliar tool (Manim) for an unfamiliar task (teaching children.) It might be easier to do what [Aleph 0](https://www.youtube.com/@Aleph0) does using cardboards. However, I am torn because I am thinking about mainly using simulations to teach engaging topics (and under 10 minutes in total).

In these posts, I will mainly discuss the ideas I want to teach my children and brainstorm ways to implement them. I will also discuss the challenges I faced when teaching them and how I overcame them. When a video is made, I will post a link here, too.

## Topic 1 - Arcsin Law
One night, I was helping my daughter with her math homework. It involved building a histogram from counting data and producing simple summary statistics. Then this idea hit me: Histograms are a gateway topic to probability and statistics, and many counterintuitive probability distributions exist. I want to teach my daughter about the Arcsin law! In teaching my daughter the Arcsin laws, I can also teach her how to compute and gather the relevant metrics using simulation, put them into a histogram, and visualize them against the theoretical Arcsin distribution. Only some distributions are bell-shaped or have a single statistical mode; the Arcsin distribution is an exception!

The Arcsin law is a fascinating phenomenon in probability theory that appears in various contexts, from random walks to Brownian motion. This post will explore this law from different perspectives, starting with a rigorous mathematical treatment and gradually simplifying our explanation to make it accessible to a wider audience, including children.

### The Mathematical Formulation

For mathematicians and statisticians, the Arcsin law is a statement about the distribution of certain random variables associated with stochastic processes. Let's consider three classic examples:

1. **Last Exit Time**: For a Brownian motion $B_t$ on $[0,1]$, let $G = \sup\{0 \leq t \leq 1 : B_t = 0\}$ be the last time $B_t$ hits zero. Then $G$ follows the arcsin distribution.

2. **Time of Maximum**: Let $G_{\text{max}} = \inf\{0 \leq t \leq 1 : B_t = \max_{0 \leq s \leq 1} B_s\}$ be the time at which Brownian motion attains its maximum on $[0,1]$. $G_{\text{max}}$ also follows the arcsin distribution.

3. **Occupation Time**: Let $\Gamma = \int_0^1 1_{[B_s > 0]} ds$ be the amount of time $B_t$ spends above zero. $\Gamma$ follows the arcsin distribution as well.

The probability density function of the arcsin distribution is given by:

$$ f(x) = \frac{1}{\pi\sqrt{x(1-x)}}, \quad 0 < x < 1 $$

For discrete random walks, analogous results hold. For instance, if $S_n = X_1 + ... + X_n$ is a simple random walk where $P(X_i = \pm 1) = \frac{1}{2}$, then the proportion of time the walk spends positive converges in distribution to the arcsin law as $n \to \infty$.

The cumulative distribution function of the arcsin distribution is:

$$ F(x) = \frac{2}{\pi} \arcsin(\sqrt{x}), \quad 0 \leq x \leq 1 $$

This is where the name "arcsin law" comes from.


### Intuition and Explanation for Children

Imagine you and your best friend are having a basketball challenge. You're both equally good at shooting hoops. Here's how the game works:

- You take turns shooting the ball.
- Every time you score, you get a point.
- You play for a long time, like 100 shots each.

Now, let's see how this game can teach us about something called the "arcsin law" in a fun way!

#### Who's in the Lead?
You might think that in this long game, you'd be winning about half the time, and your friend would be winning the other half. But guess what? That's not usually what happens! 

Most of the time, one of you will be in the lead for a really long time, like for 80 out of the 100 shots. It's like one player gets "hot" and stays ahead for most of the game. Even though you're both equally good, it often turns out that one person stays in the lead more often.

#### The Last Tie
Now, think about the last time the score is tied in your game. When do you think this usually happens? Surprisingly, it's most likely to be either at the very beginning (like after the first few shots) or at the very end (like in the last few shots). It's not usually going to be in the middle of the game. Isn't that strange?

#### The Biggest Lead
The same funny thing happens with the biggest lead in the game. Let's say at some point, one of you gets ahead by 10 points. When do you think this biggest lead usually happens? It's probably going to be either very early in the game or very late. It's not usually going to happen right in the middle.

#### Why Does This Happen?
So, why does the game work this way? It's because of how random things work in sports. When you're shooting baskets, sometimes you might get "in the zone" and score many times in a row. Other times, you might miss a few shots. These streaks of good or bad luck can make one player get ahead and stay ahead for a long time.

It's like when a basketball player in a real game suddenly starts making every shot – they call it "getting hot." This can happen to either player at any time, which is what makes the game exciting!

#### The Excitement of the Game
That's what makes games like this so thrilling! You never know what's going to happen next. Will you get on a hot streak and take a big lead? Or will your friend make an amazing comeback? It's all part of the game!

These surprises and the back-and-forth make the game fun and full of unexpected twists and turns. It's why we love watching sports – you never know when something amazing might happen!

#### What We Learn
This basketball challenge teaches us something cool about random events. Even when two players are equally matched, the game often doesn't stay close the whole time. Instead, it's full of surprises, with long leads, late ties, and big comebacks.

So, next time you play a game or watch sports, remember that the excitement comes from not knowing what's going to happen. Whether you're ahead or behind, just enjoy the game and see where your shots take you!

### Conclusion
The arcsin law is a fascinating example of how mathematical concepts can challenge our intuitions about randomness. Whether you're a mathematician studying complex theories, a sports analyst looking at game statistics, or a child enjoying a friendly basketball challenge, the arcsin law reveals surprising patterns in random processes.

For mathematicians, it provides a rigorous framework to understand the behavior of random events. For everyone else, it offers insights into the unexpected nature of random events in everyday life. And for children, it introduces the exciting world of probability through fun and relatable sports scenarios.

The arcsin law reminds us that in the world of probability, our intuitions can often lead us astray, and careful analysis - or lots of basketball shots! - can reveal unexpected truths. It shows us that randomness doesn't always behave the way we expect, and that's what makes it so fascinating to study and experience.

Whether you're solving complex equations or just shooting hoops with a friend, the arcsin law is there, quietly shaping the patterns of chance in ways that continue to surprise and delight us. It helps explain why sports games can be so unpredictable and exciting, and why we should always expect the unexpected!

## Next Step
The arcsin law is a prime example of how mathematical concepts can challenge our intuitions about randomness. Whether you're a mathematician studying Brownian motion, a sports fan analyzing game statistics, or a child playing a coin-flipping game, the arcsin law reveals surprising patterns in random processes. It reminds us that our intuitions can often lead us astray in probability, and careful analysis can reveal unexpected truths. In the [next post](https://steveya.github.io/posts/teaching-kids-math-arcsin-law-2/), we'll discuss how to use simulation to teach Monte Carlo simulations to children.

