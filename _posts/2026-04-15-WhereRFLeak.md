---
title: 'A (not so) short and (yet very) intuitive view of our article Where Rectified Flow Leaks'
date: 2026-04-15
permalink: /posts/2026/04/WhereRFLeak/
excerpt: "The original paper can be a bit reluctant, yet the main ideas and intuition are worth to be shared!"
tags:
  - Generative Models
  - Flow Matching
  - Memorization
  - ICML
---

This post is a walkthrough of our [ICML 2026 paper](#) on memorization in Rectified Flows, written with Gabriel Meseguer-Brocal and Geoffroy Peeters. The paper itself is fairly technical and I feel I didn't put enough intuition in the main scene. So what I want to do here is ducoup give the intuition behind it: not just what we found, but why we did each thing we did. Most of the machinery in the paper exists to answer a specific question, and I think the questions are more interesting than the machinery ;)

I've kept the maths light. But of course If you want the formal version, everything is in the paper.

## The question

Generative models are trained on enormous piles of data, a lot of it copyrighted. That has made one question suddenly urgent, and it has been showing up in lawsuits from [Getty against Stability](https://www.courtlistener.com/) to the [major record labels against Suno and Udio](https://www.riaa.com/): what does a model actually keep about what it was trained on?

The obvious worry is verbatim copying, the model handing back a training image or a melody note for note. But there is a subtler failure mode underneath it. A model can *treat* its training data differently from data it has never seen, reconstruct it a little more faithfully, behave a little differently near it, without ever reproducing it. We call that measurable asymmetry the **membership signal**: any trace you can read off a model that tells you whether a given sample was in its training set.

It is a kind of memorization, but a quiet one. And the frustrating thing about it is that it does not show up where you would look. You can train a model that has clearly absorbed a lot about its data, and its loss curves will look perfectly healthy. No overfitting, validation dropping smoothly, nothing to see.

So the question we started from was blunt: if it is not on the loss curve, then *where* in the model's behaviour does this information live?

## Why we look along the path

To answer that, we needed somewhere to look. And Rectified Flows conveniently hand us one.

[Rectified Flows](https://arxiv.org/abs/2209.03003) and [Flow Matching](https://arxiv.org/abs/2210.02747) are behind systems like [Stable Audio](https://arxiv.org/abs/2407.14358), [FLUX](https://blackforestlabs.ai/), and [Stable Diffusion 3](https://arxiv.org/abs/2403.03206). The idea is geometric. You have two distributions, pure noise on one side and real data on the other, and you learn to travel from one to the other along a *straight path*.

Pick a noise point $x_0$ and a data point $x_1$, draw the straight line between them, and a point on that line is $x_\lambda = (1-\lambda) x_0 + \lambda x_1$. Here $\lambda$ says how far along you are: $\lambda = 0$ is pure noise, $\lambda = 1$ is the data. The model's whole job is to look at a point on this line and predict the *direction* that carries it toward the data. At generation time you start from noise, take small steps, and the model hands you the direction at each one.

![The interpolation path](/images/rf-leak/01_interpolation_path.svg)

The property we care about is that the model makes a prediction at *every* point along this line. That gives us a whole continuum of places to inspect how it treats a sample. That is the opening.

*(A caveat: there is a popular trick called reflow that pairs noise and data points in a learned way instead of independently. We study plain Rectified Flows with independent coupling, so throughout this post "Rectified Flow" means flow matching on this straight path. Reflow comes back briefly at the end.)*

## Why we measure a gap

We need to see something the loss curves cannot. So we built a deliberately simple probe.

Take one data point, from the training set or from a held-out set. Mix it with noise to land at some position $\lambda$ on the path. This is not an arbitrary construction: it places us on the *real trajectory* this sample would take through the model. Then let the model predict the direction from there, take one step, and reconstruct where it thinks the data should be. Compare that to the true point. The distance is the model's error at that position.

Now the important move. Do this for many samples, separately for the training set and the held-out set, and look at the **average difference** in error between the two, as a function of where you started.

Why a difference, and not just the error itself? Because the raw error is dominated by things that have nothing to do with membership: how intrinsically hard the point is, how much noise there is. Subtracting held-out from train cancels all of that. What survives is only what differs between seen and unseen. The gap is a subtraction *designed* to isolate the one thing we want.

And when you plot that gap against $\lambda$, you get a bell. Essentially zero at both ends of the path, rising to a single clean peak somewhere in the middle.

![The bell-shaped train-test gap](/images/rf-leak/02_bell_curve.svg)

This shape turns up in every configuration we tried: different datasets, different architectures, even different modalities. So the two questions that drive the rest of the paper are: why a bell, and why *there*?

## Why there: the hourglass

Here is the picture that made it click for us.

Picture the path in 3D. Two flat planes facing each other, the noise distribution on the left and the data distribution on the right, with the interpolation paths running between them. Every sample is a single straight strand, tied from its noise point on the left to its data point on the right. So the whole thing is a bundle of straight spaghetti stretched across the gap.

Now pull the two clouds apart and look at the bundle from the side. Because each strand joins a scattered noise point to a scattered data point, the bundle is not a uniform tube. It is an **hourglass**. Wide at both ends where the points are spread out, and pinched in the middle where all the strands funnel through a narrow waist.

![The hourglass](/images/rf-leak/03_hourglass.svg)

Now think about the model's job at each point: given your position, predict your direction. Near the ends this is easy. The strands are well separated, so where you are pins down which way you are headed. There is a clean, *linear* relationship between position and direction. But at the pinch, the strands are crushed together and run almost parallel. Knowing your position barely narrows down your direction at all. The linear relationship collapses.

This is the crux. There is exactly one point on the path where your position tells you the least, linearly, about where to go. And in the clean Gaussian case we can compute that point *in closed form*, from the variances of the noise and the data alone. Nothing about the model.

> The signal peaks exactly where the model has the least linear information. It leaks the most precisely where it knows the least.

That is the part I find genuinely surprising, and it is worth sitting with, because the naive expectation is the opposite. You would think a model leaks most where it is most confident, where it "knows" its data best. It is the reverse. The leak sits at the point of maximum confusion.

To see why, we have to look at what the model is forced to do when the linear crutch is taken away.

## Why the leak has to exist

Start with something that costs nothing. You can always split what the model is learning (the direction field) into a **linear part** and a **nonlinear part**. The linear part is the smooth global trend, a simple rule from position to direction. The nonlinear part is everything that rule misses: the curvature, the fine detail.

There is a well-known principle called [spectral bias](https://arxiv.org/abs/1806.08734): networks learn the smooth, low-frequency part of a target first, and only reach for the fine, high-frequency detail later. That ordering is not arbitrary. The smooth part is exactly the part that *generalizes*, because a smooth rule you fit on your training points still behaves sensibly on points between them. So while there is linear structure to ride, the model rides it, and it generalizes for free.

But near the pinch, the linear structure is gone. To predict at all, the model is forced into the nonlinear regime. And here is the trap. The nonlinear structure it now has to learn comes in two kinds, tangled together:

* **The true shape of the data distribution.** Real nonlinear structure that every sample shares, the actual geometry of the data. Learning it helps on new data. This generalizes.
* **Each sample's private fingerprint.** The residual specific to one individual training point, its own quirk, shared with nothing else. Learning it helps only on that point. This is memorization.

The catch, and this is the heart of the paper, is that **these two are statistically indistinguishable**. Same mean, same covariance with the input. From the model's point of view, looking only at its training samples, there is no way to tell which piece of nonlinear structure is the generalizable shape and which is just this one sample's fingerprint. They look identical to first and second order.

So when the model reaches for the useful part, it cannot avoid grabbing some of the private part along with it. Fitting the fingerprint is the *price* of fitting the shape.

This is why I would not call the leak overfitting. It is not a failure of regularization, not something you fix with more dropout. It is baked into what learning nonlinear structure means at all. And it concentrates at the pinch, because that is the one place the model has no linear alternative.

### The rope

A fair objection at this point: surely with enough data the fingerprints average out?

They do, partly. We prove the signal shrinks with dataset size, roughly like $1/n$. But it never fully vanishes, and there is a clean way to feel why.

Think of learning as tensioning a rope. Each gradient step, trying to fit the data, bends the model toward the fingerprints of the samples in front of it, a bit like plucking the rope. Later steps, seeing other samples, pull the other way and compensate. Over training these pushes and counter-pushes largely cancel, and with more data the bias from any single sample gets increasingly outweighed. That averaging is what gives the $1/n$ shrinkage we prove.

But the cancellation is never *exact*. Each correction is itself imperfect. It overshoots here, undershoots there, because it is driven by a finite noisy batch and not by a perfect mirror of the bias it is fixing. So what is left after all the pushing and counter-pushing is not a flat rope. It is a scatter of tiny residual ripples all along it.

![The rope](/images/rf-leak/04_rope.svg)

Those leftover ripples are the sample-specific information the model could not quite cancel. And that residue is exactly what our probe reads. It also explains a pattern in our experiments: bigger datasets leak less, and bigger *models* leak more, since a larger model has the capacity to press harder into those ripples. Neither one moves where the peak sits.

## Why nothing shows up on the dashboard

So the signal is always there. Why does no standard metric catch it? Two separate reasons, and they compound. This is really why we had to build the gap probe in the first place: the ordinary dashboard is structurally blind here.

**Dilution.** The signal lives in a narrow band near the peak. But training monitors the loss *averaged over the whole path*. Spread one sharp spike across the entire interval and the average barely twitches.

**Masking.** This one is subtler. Early in training the model is still learning the generalizable structure, and that lowers *both* the training and the validation loss together. Underneath, on the training side, the leak term is quietly growing. But it is buried under the much larger gains from generalizing. On the validation side the leak term is exactly zero, since the model never saw those samples, so validation just keeps dropping and looks perfectly healthy. The leak accumulates on training, epoch after epoch, hidden beneath a curve that is doing everything right.

![Masking](/images/rf-leak/05_masking.svg)

By the time you early-stop on a nice-looking validation curve, a real signal has already built up. Which is the disconnect we started from, now with a mechanism behind it.

## Why the gap actually works

Everything so far has been intuition. The paper's job is to make it rigorous, and the whole argument rests on one picture, so let me show just that one.

Rectified Flows are trained by least squares, so the loss is a squared distance. And any squared distance splits, by the law of cosines, into three pieces. Draw the model's prediction, the optimal prediction, and the true target as a triangle:

![The triangle of the three loss terms](/images/rf-leak/06_triangle.svg)

The three pieces are: the **approximation error** (how far the model is from the optimal predictor), the **irreducible noise** (the sample's own residual, which no model can predict), and a **cross-term**, which is the *angle* between them. That third one is the interesting one, and notice that it is an angle, not a length. That is a big part of why it hides so well: standard metrics essentially see lengths.

Now watch what happens when we take the train minus held-out difference.

On held-out data the cross-term is *exactly zero*. The model has never seen those residuals, so its error cannot be aligned with them. On training data it is not zero, because there the model was fit on those very residuals, so its error tilts toward them. Meanwhile the other two pieces behave the same on seen and unseen points (this is where our two assumptions come in, and both of them just describe a well-trained model). So they cancel in the difference.

> Subtract train from held-out and everything cancels except the one term that knows whether a point was seen. The gap *is* the membership signal.

From there it is calculation. In the clean Gaussian case the optimal predictor is provably linear, so studying a linear model is not an approximation, it is the exact target the network is reaching for. You work out that the signal is proportional to the *irreducible variance*, the part of the direction that no linear rule can predict. And that variance is largest exactly where the linear information vanishes, which is the pinch of the hourglass. The bell, the peak location, the $1/n$ decay all drop out of that one proportionality.

**Why a Gaussian story for a real transformer, though?** This is the move that needs justifying, so it is worth being explicit. Two reasons. First, spectral bias again: the network learns the linear part first, so it really does defer the nonlinear scramble to where the linear signal dies. Second, we work in the latent space of an autoencoder, and those latents are *built* to be roughly Gaussian and isotropic, whether through an explicit KL penalty or through bounded activations like tanh. So the assumption our theory needs is approximately satisfied by construction, which is why the closed-form peak actually lands on real data.

## Why we believe it

A theory that predicts a specific location is a theory you can try to break. So we did, in both directions.

First we moved the peak on purpose. Our formula says the location depends only on the covariances at the two ends of the path, so we pushed on each end. Scale up the noise variance and the peak shifts, exactly as predicted. Swap the training dataset for one with different structure and the peak shifts again, prediction matching.

Then we did the opposite and varied everything the theory says should *not* matter: architecture (transformer versus UNet), model size, the sampling scheduler. The peak did not move. Only its height changed.

> Data geometry sets the location. Model choices set the magnitude. The bell shape itself is universal.

The most telling case is the one where the prediction *fails*. On image latents from a VAE that is genuinely far from Gaussian, heavy-tailed and strongly correlated, the closed-form peak misses. But the bell is still there. Which is exactly what the theory says should happen: the shape is universal, the location depends on the assumptions. A story that only ever succeeds is a bit suspicious. One that predicts its own failure mode is telling you it caught a real mechanism.

## Turning it into an attack

If the signal is real it should be exploitable, so as a proof of concept we made it into one. Feed the full per-$\lambda$ error profile of a sample into a small classifier and ask it: member or not? Because the profile encodes the whole shape of the bell, it is far more informative than any single measurement. As far as we know it is the first membership inference attack designed specifically for Rectified Flows, and it comfortably beats attacks ported over from the diffusion literature ([SecMI](https://arxiv.org/abs/2302.01316), [PIA](https://arxiv.org/abs/2308.06405)).

The point is not the attack itself. It is the confirmation that the structure we derived on a whiteboard translates into a practical, measurable risk.

## Where this goes

Plenty is still open. We only glanced at reflow, and the early signs are that the bell survives but flattens out, which hints that reflow might double as a natural defence. Text-conditioned generation, stronger threat models, larger deployed systems: all still to do.

But the core of it is small, and I think a little beautiful. A generative model leaves a structured, predictable trace of its training data. That trace sits at a spot you can compute ahead of time from the data alone. It grows silently while every dashboard says the model is fine. And it is there not because the model failed, but because it succeeded: absorbing a little of each sample's fingerprint is the unavoidable cost of learning the structure that generalizes.

The model's fingerprint hides where it knows the least, and it cannot help but leave it there.

----------

Thanks for reading! If you have questions, disagreements, or spot something I got wrong, please do reach out. I would genuinely like to hear it.

Everything here reflects my own understanding, which is certainly partial. The figures are sketches of the intuitions rather than reproductions of the plots in the paper, so take them as drawings on a whiteboard, not as results.