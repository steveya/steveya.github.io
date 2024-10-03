---
layout: post
title: WIP Teaching Kids Math - Arcsin Law (Part 1 - Examples)
# date: 2024-04-21
categories: [Teaching Kids Math]
tags: [teaching-kids-math]
published: false
---

<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Teaching Kids Math Series
Since my first child I have been thinking about how to teach children about mathematics. In particular, mathematics that I did not learn about at school when I was a child, but I think I would have grasped it at that age had someone introduced me to it. I also wanted to start a video series to share these materials with others and at the same time to challenge myself to introducing my chosen concepts assuming only grade school mathematics (and to make YouTube videos).

I experimented with making videos using the Manim library (from [3Blue1Brown](https://www.youtube.com/@3blue1brown)) since that channel is how I learn a lot of interesting math topics on YouTube. I am still on the fence about using Manim for this purpose because I would be using an unfamiliar library for an unfamiliar task (teaching children.) It might be easier to do what [Aleph 0](https://www.youtube.com/@Aleph0) does using cardboards. However, I am torn because I am thinking about mainly using simulations to teach engaging topics (and under 10 minutes in total). Either way, I am also leaning as I go.

There are similar endeavors to teach kids higher mathematics and even machine learning, such as books published by [Natural Math](https://naturalmath.com), whose excellent book [Mobius Noodles](https://naturalmath.com/moebius-noodles/) were published shortly after my first daughter was born, and we had both enjoyed some of the games in that book. I am merely contributing to this effort in my own smalll way. In these posts, I will mainly discuss the ideas I want to teach my children and brainstorm ways to implement them into teaching material or video. I will also discuss the challenges I faced when teaching them and how I overcame them. When a video is made, I will post a link here, too.

## Topic 1 - Arcsin Law
One night, I was helping my daughter with her math homework. It involved building a histogram from counting data and producing central tendency statistics on the data. The data we worked on that night was given and can either be approximated by a Gaussian or a Poisson distribution and are both unimodal. I want to use this opportunity to show her how to collect or generate data herself, and introduce her to distributions that are not unimodel. Histograms really are a gateway topic to probability and statistics, and many counterintuitive probability distributions exist. Eventually I landed on the Arcsin distribution and the associated Arcsin law. It fits my objective: in teaching my daughter the Arcsin laws, I can also teach her how to compute and gather the relevant metrics using simulation, put them into a histogram, and visualize them against the theoretical Arcsin distribution. Only some distributions are bell-shaped or have a single statistical mode; the Arcsin distribution is an exception.

The Arcsin law is a fascinating phenomenon in probability theory that appears in various contexts. I first encountered it when studying properties of random walks, and later on saw it again in finance when analyzing performance of systematic investment strategies. This post will explore these laws from different perspectives, starting with a high-level mathematical introduction and gradually simplifying our explanation to make it accessible to children. For the benefit of the educator, let's review the exact statements we are trying to teach.

### WIP The Mathematical Formulation

The Arcsin law is a statement about the distribution of certain random variables associated with symmetric Brownian motion. It comprises of three statements:

1. **Last Exit Time**: For a Brownian motion $B_t$ on $[0,1]$, let $G = \sup\{0 \leq t \leq 1 : B_t = 0\}$ be the last time $B_t$ hits zero. Then $G$ follows the arcsin distribution.

2. **Time of Maximum**: Let $G_{\text{max}} = \inf\{0 \leq t \leq 1 : B_t = \max_{0 \leq s \leq 1} B_s\}$ be the time at which Brownian motion attains its maximum on $[0,1]$. $G_{\text{max}}$ also follows the arcsin distribution.

3. **Occupation Time**: Let $\Gamma = \int_0^1 1_{[B_s > 0]} ds$ be the amount of time $B_t$ spends above zero. $\Gamma$ follows the arcsin distribution as well.

The probability density function of the arcsin distribution is given by:

$$ f(x) = \frac{1}{\pi\sqrt{x(1-x)}}, \quad 0 < x < 1 $$

For discrete random walks, analogous results hold. For instance, if $S_n = X_1 + ... + X_n$ is a simple symmetric random walk where $P(X_i = \pm 1) = \frac{1}{2}$, then the proportion of time the walk spends positive converges in distribution to the arcsin law as $n \to \infty$.

The cumulative distribution function of the arcsin distribution is:

$$ F(x) = \frac{2}{\pi} \arcsin(\sqrt{x}), \quad 0 \leq x \leq 1 $$

This is where the name "arcsin" comes from.

All these laws say the same thing: the last exit time, the time of the maximum, and the occupation time of a symmetric Brownian motion on $[0, 1]$ tends to concentrate around 0 and 1. The proofs are too difficult for most children to understand, but should be leveraged if it helps them gain more profound understanding.

Below is a work-in-progress of the audio script to the video I am making for this topic.

### WIP First Segment: Examples of the Arcsin Law

Imagine you and your best friend are having a basketball challenge. You're both equally good at shooting hoops. Here's how the game works:

- You take turns shooting the ball.
- Every time you score, you get a point.
- You play for a long time, like 100 shots each.

Now, let's see how this game can teach us about something called the "arcsin law" in a fun way!

#### Who's in the Lead?
You might think that in this long game, you'd be winning (in the lead) about half the time, and your friend would be winning the other half. But guess what? That's not usually what happens! In fact, it is the least likely scenario. The most likely scenario is that one of you will be in the lead for a really long time, like for 80 out of the 100 shots. In this case, the other player would be in the lead only for 20 shots. Even though you're both equally good, it often turns out that one person stays in the lead more often.

#### The Last Tie
Now, think about the last time the score is tied in your game. When do you think this usually happens? Surprisingly, it's most likely to be either at the very beginning (like after the first few shots) or at the very end (like in the last few shots). It's not usually going to be in the middle of the game. Isn't that strange?

#### The Biggest Lead
The same funny thing happens with the biggest lead in the game. Let's say at some point, one of you gets ahead by 10 points. When do you think this biggest lead usually happens? It's probably going to be either very early in the game or very late. It's not usually going to happen right in the middle.

#### Why Does This Happen?
So, why does the game work this way? After all, if the two players are equally good, wouldn't their scores stay close to each other throughout the game? But this means that out all the shots the players have taken so far, each has achieved the same score. This is less likely to happen than one might think. 

#### The Excitement of the Game
That's what makes games like this so thrilling! In a tightly matched basketcall game, such as in the semi-finals where the two teams are likely equally good, you never know what's going to happen next. Will you get on a hot streak and take a big lead? Or will your friend make an amazing comeback? It's all part of the game!

These surprises and the back-and-forth make the game fun and full of unexpected twists and turns. It's why we love watching sports – you never know when something amazing might happen!

#### What does it have to do with Arcsin? What is Arcsin anyways?
The tendency for the time one player stays in the lead to be either a very small or a large fraction of the game can be demonstrated by the Arcsin probability distribution. Similarly, the tendency of the occurance of the last tie, or the time one player gets the biggest lead in the game, both tend to happen either very early or very late in the game, can be described by the Arcsin distribution. (TODO: I am not sure if I need to mention the Arcsin distribution in the material at all, it may not mean anything to the children...)

### Conclusion for the First Segment
The arcsin law is a fascinating example of how mathematical concepts can challenge our intuitions about randomness. Whether you're a mathematician studying complex theories, a sports analyst looking at game statistics, or a child enjoying a friendly basketball challenge, the arcsin law reveals surprising patterns in random processes.

The arcsin law reminds us that in the world of probability, our intuitions can often lead us astray, and careful analysis - or lots of basketball shots! - can reveal unexpected truths. It shows us that randomness doesn't always behave the way we expect, and that's what makes it so fascinating to study and experience.

## Next Step
In the [next post]({% post_url 2025-01-31-teaching-kids-math-arcsin-law-2 %}), we'll discuss how to use simulation to teach Monte Carlo simulations to children.

