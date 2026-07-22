---
title: 'Untangling Flow-Based Generative Models'
date: 2025-11-20
permalink: /posts/2025/11/UntangFBGM/
excerpt: "A long journey about the history of diffusion-based generative model"
tags:
  - Generative Models
  - Flow Matching
  - Diffusion
  - Review
---

I'm surely not the only one to feel a bit overwhelmed about paradigms of generative models that "seems to grow like mushrooms" :  Normalizing Flows, Continuous Normalizing Flows, Score Matching, DDPM, DDIM, Flow Matching, Rectified Flow, Consistency Models, MeanFlow... Yeah, I agree that's quite a lot. Worst part is that they are all intricated to each other, and it is really EASY to mismatch and confuse them. Yet, I believe those connection are also a very usefull properties to know them better. 
At this point, I feel like the research area produce even too much word and even top tier scientist confused them, and some times discussing about generative paradigms look more as a opinion debate than are good old fact grounded science for me.

Hence, here I tried to enumerate most important technics based / related to diffusion by far or close, in terms of a the *physic* view of diffusion: they learn a *continuous transformation* between a simple distribution and the data distribution, via either ODEs or SDEs. That shared structure is what makes their connections so deep, and sometimes so confusing. 

> Except in the previous paragraph, *diffusion* will refer to the AI sense: DDPM/DDIM NCSN.

For each method,  I'll try to present them from the foundationnal papers and the key equations. And explicitly connecte the dot between one method to the others. For a more practical insight, I'll also try to answer three questions: **what do you optimize during training? What do you do at inference? Does it constrain your architecture?** And for the main trainable methods, I'd like to include a short "in practice" recipe showing how you would actually implement one.

To conclude this introduction, as you can imagine, these methods are even closer than their names suggest, sometimes even mathematically equivalent. Understanding where the equivalences hold (and where they break) is what explains why some work better in practice.


> **What about GANs, VAEs, and autoregressive models?** This post focuses specifically on the flow / diffusion / score matching family. Other major generative paradigms (GANs with adversarial training, VAEs with variational inference, autoregressive models with next-token prediction) are not covered here. Not because they are less important, but because they belong to different mathematical frameworks with their own rich histories. 


## About the use of AI
I used AI during the writing of this article. Here are the main uses:
- Spelling, grammar, correctness, and some formatting of my (deply French-influenced) English.
- Information verification and a global review of the blog.
- Literature references, such as Anderson's 1972 article "Reverse-Time Diffusion Equations," which I wasn't aware of.

I did not have the help of any other human, and my knowledge is of course fallible. If you encounter any error, typo, inconsistency, or even misunderstanding, please let me know! It would help me a lot.

## 1. Normalizing Flows (2014-2018)

The starting point of flow-based / physics meaning diffusion generative modeling is the **change of variables theorem**, which builds on Euler's and Lagrange's work on integral substitution, completed by Jacobi's formalization of the functional determinant (yes, the *Jacobian*). Given a random variable $$z$$ with a known density (typically a Gaussian) and an invertible differentiable map $$x = g(z)$$, the density of $$x$$ is:

$$p_X(x) = p_Z\!\big(g^{-1}(x)\big)\;\left|\det\frac{\partial g^{-1}}{\partial x}\right|$$

Basicaly, it explains what happend to a given initial distribution Z if we applied to it a derivable and inversible transformation.

Hence, a **Normalizing Flow** (NF) chains $$K$$ such invertible transformations $$g = g_K \circ \cdots \circ g_1$$, each parameterized by a neural net. "Normalizing" because $$g^{-1}$$ maps complex data back to a simple (normal) distribution; "flow" because the composition moves probability mass step by step. The Jacobian determinant accounts for how the transformation locally stretches or compresses volume. It is the correction factor needed to go from one density to the other.


* **Training:** exact maximum likelihood via the change of variables formula.
* **Inference:** a single forward pass through $$g$$.
* **Architecture constraint:** yes, and a strong one. Each layer must be invertible with a tractable Jacobian determinant. Computing a determinant costs $$\mathcal{O}(d^3)$$ in general, so all practical NF architectures keep the Jacobian triangular (which brings the cost down to $$\mathcal{O}(d)$$). This forces very specific designs: additive coupling layers in NICE [1], affine coupling layers in RealNVP [2] and Glow [3], or autoregressive structures in MAF [4] and IAF [5].

The terminology itself was popularized by Rezende & Mohamed [6] in the context of variational inference.

**In practice**, training a normalizing flow looks like this:
1. Sample a batch of data $$x$$.
2. Compute $$z = g^{-1}(x)$$, accumulating the log-determinant layer by layer.
3. The loss is the negative log-likelihood: $$-\big[\log p_Z(z) + \sum_k \log \lvert \det J_k \rvert \big]$$.
4. Gradient step, repeat.

The sampling is in one forward pass: draw $$z \sim \mathcal{N}(0,I)$$ and compute $$x = g(z)$$, *et voila*.

Again, the main limitations are the invertibility constraint severely limits expressivity, and also $$z$$ and $$x$$ must share the same dimensionality. These architectural restrictions directly motivated the next development.

> **PS (2025 update):** Normalizing flows fell out of fashion around 2021-2022, eclipsed by diffusion and transformer-based models. They came back into the spotlight when Apple's [TarFlow](https://arxiv.org/abs/2412.06329) (Zhai et al., ICML 2025) showed that **Transformer blocks make excellent flow layers** :a stack of autoregressive Transformer blocks on image patches, alternating autoregression direction between layers. The mathematics is unchanged and the transformation $$g$$ is simply far more expressive. For the first time a stand-alone NF matched diffusion models on sample quality while setting new state-of-the-art likelihood scores. And that one paper gave to the whole field a new youth: a wave of follow-ups quickly appeared: [STARFlow](https://machinelearning.apple.com/research/normalizing-flows) (high-resolution scaling, NeurIPS 2025), [BiFlow](https://arxiv.org/abs/2512.10953) (dropping the exact-inverse constraint), plus applications beyond image generation, from [reinforcement learning](https://arxiv.org/abs/2505.23527) to [robotic visuomotor policies](https://arxiv.org/abs/2509.21073) or [adversarial purification](https://arxiv.org/abs/2505.13280).
>
> What an important reminder that a subfield or method looking outdated doesn't mean it has nothing valuable left to offer.. sometimes it just needed work work on it.  And that CS suffer from a very much trendy, visibility-driven field, where attention flows toward whatever's hot... but that's another story ;)

## 2. Continuous Normalizing Flows (2018)

Neural ODEs [7] (NeurIPS 2018 Best Paper) introduced the idea of neural networks as continuous-time dynamical systems. Interestingly, their primary motivation was not normalizing flows. They observed that residual networks ($$x_{n+1} = x_n + f(x_n)$$) look like Euler discretizations of an ODE, and proposed to parameterize the dynamics directly:

$$\frac{dz(t)}{dt} = f_\theta(z(t), t), \qquad z(0) = z_0$$

The Neural ODE paper had several applications (e.g. time series, supervised learning). When used as a generative model, the ODE maps a simple distribution to the data distribution by integration, and the result is called a **Continuous Normalizing Flow** (CNF).

The key result for CNFs is the *instantaneous change of variables*: while a discrete normalizing flow requires the full Jacobian determinant, the continuous version only requires the Jacobian **trace** (Theoreme 1 from Chen et al. [7]):

$$\frac{\partial \log p(z(t))}{\partial t} = -\operatorname{tr}\!\left(\frac{\partial f_\theta}{\partial z}\right)$$

Intuitively, the determinant tracks the total volume change of a finite transformation, while the trace tracks the *instantaneous rate* of volume change. In the continuous limit, you only need the infinitesimal version, which is much cheaper. FFJORD [8] reduced this further to $$\mathcal{O}(d)$$ with Hutchinson's stochastic trace estimator.

* **Training:** maximum likelihood, but it requires *simulating the ODE* at every training step (forward and backward, via the adjoint method).
* **Inference:** an ODE solver (e.g. Dopri5 fror adatative steps, RK4 for fixed ones), roughly 100 network evaluations.
* **Architecture constraint:** none. $$f_\theta$$ can be any neural network.

CNFs solved the architecture constraint of NFs but introduced a new cost as the ODE simulation is needed during training. From my understanding, FFJORD only scaled correctly to tabular data and low-resolution images. CNFs would not become practical for large-scale generation until Flow Matching in 2022-23 (Section 6). But before that happened, an "entirely different" family of models took over.

## 3. Diffusion Models: NCSN and DDPM (2019-2020)

While normalizing flows were evolving, a completely (not that) independent line of work was developing. It shares no historical lineage with flows and the connection would only be discovered later (Section 5).

### Background: score matching

Before discussing diffusion models, we need a tool from the statistical estimation litreratur. In 2005, well before the deep learning era, Hyvärinen [9] studied the problem of learning a probability distribution without computing its intractable normalization constant as. Instead of the density $$p(x)$$ itself, he proposed to learn its **score function** $$\nabla_x \log p(x)$$. Writing $$p(x) = e^{-E(x)}/Z$$, we get $$\log p(x) = -E(x) - \log Z$$, and since $$Z$$ does not depend on $$x$$, taking the gradient with respect to $$x$$ eliminates it entirely. Vincent [10] then showed that this score can be estimated by *denoising*: corrupt data with Gaussian noise, and the score of the noisy distribution points back toward the clean data.

You can guess that they turned out to be exactly what diffusion models relie on more than a decade after.

### NCSN: Noise Conditional Score Network (Song & Ermon, 2019)


Song & Ermon [11] (NeurIPS 2019 Oral) used denoising score matching with neural networks to build a generative model they called NCSN. A single network $$s_\theta(x, \sigma)$$ that estimates the score of the data distribution, conditioned on the noise level $$\sigma$$. The main challenge was that the score is poorly estimated in low-density regions, where training samples are sparse. To face this, they proposed to perturb the data at *multiple noise scales* $$\sigma_1 > \sigma_2 > \cdots > \sigma_L$$, and train that single network to estimate the score of each noisy distribution.

Sampling uses **Langevin dynamics**, an MCMC algorithm that follows the score toward high-density regions while injecting noise for exploration:

$$x_{k+1} = x_k + \frac{\epsilon}{2}\, s_\theta(x_k, \sigma) + \sqrt{\epsilon}\;\eta_k, \qquad \eta_k \sim \mathcal{N}(0, I)$$

The name borrows the idea of annealing (progressively lowering the noise like a temperature), but this is a *sampling* procedur and not optimization, so it isn't simulated annealing in the usual sense.

> NCSN is often referred to more broadly as *score matching*, but this actually designates the underlying technique rather than the model itself. Concretely, NCSN relies on **denoising score matching** (Vincent, 2011), trained across multiple noise scales (hence "Noise Conditional"), combined with **Langevin dynamics** for sampling. This combination "score matching plus Langevin sampling" is what the literature commonly groups under the umbrella term "score-based generative models" of which NCSN is the founding example.

### DDPM: Discrete Diffusion Probabilistic Method (Ho, Jain & Abbeel, 2020)

Independently, and through a completely different derivation, Ho et al. [12] (NeurIPS 2020) arrived at an essentially equivalent method. Their starting point was not score matching but variational inference on a Markov chain, building on the non-equilibrium thermodynamics framework of Sohl-Dickstein et al. [13] (ICML 2015). Hence the name "diffusion", by analogy with particles diffusing in a fluid. A forward process gradually adds Gaussian noise over $$T$$ steps until the data becomes indistinguishable from pure noise, and a network learns to reverse this process.

* **Training:** predict the noise added at each timestep, with the loss $$\|\epsilon - \epsilon_\theta(x_t, t)\|^2$$. The connection to score matching: for the conditional distribution $$q(x_t \mid x_0)$$, the score is $$-\epsilon / \sqrt{1 - \bar\alpha_t}$$. So the network learns the *conditional* score, and by averaging over the data in the loss, it implicitly learns the *marginal* score, which is what generation needs. Ho et al. work this out explicitly in Section 3.2 of their paper, where they show the simplified objective is a weighted denoising score matching loss.
* **Inference:** iterative stochastic denoising, around 1000 steps.
* **Architecture constraint:** none (a U-Net in practice).

**In practice**, training a DDPM looks like this:
1. Sample $$x_0$$ from data, $$t \sim \mathcal{U}\{1,\dots,T\}$$, $$\epsilon \sim \mathcal{N}(0,I)$$.
2. Compute $$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$$ (closed form, no need to run the chain).
3. The loss is $$\|\epsilon - \epsilon_\theta(x_t, t)\|^2$$.
4. Gradient step, repeat.

To sample, we start from pure noise $$x_T \sim \mathcal{N}(0,I)$$ and iteratively denoise down to $$x_0$$, each step using the network plus a bit of fresh noise.

DDPM was a breakthrough especialy with image quality competitive with GANs but without any adversarial training. Yet the main practical problem was speed, as it needs around 1000 sequential denoising steps per sample.

## 4. DDIM (2020)

DDIM [14] (Song, Meng & Ermon, ICLR 2021) was developed specifically to fix DDPM's slow sampling. The authors observed that the DDPM training objective depends only on the marginals $$q(x_t \mid x_0)$$, not on the Markov structure of the forward chain. This means you can design non-Markovian forward processes sharing the same marginals, some of which have **deterministic** reverse processes.

* **Training:** the *same network as DDPM*, no retraining needed.
* **Inference:** deterministic sampling that can skip steps (around 50 instead of 1000). With the stochasticity parameter $$\eta = 0$$, the same initial noise always gives the same output, which also enables meaningful interpolation between samples in noise space.
* **Architecture constraint:** none.

DDIM is purely a new *sampling procedure* for an already-trained model. This illustrates a point that recurs throughout this story: training and sampling are separable concerns.

> **Retrospective connection.** DDIM with $$\eta = 0$$ turned out to be a first-order Euler discretization of the *Probability Flow ODE*, formalized a few weeks later by Song et al. [16]. This equivalence is worked out explicitly in Appendix B of Salimans & Ho [17], titled "DDIM is an integrator of the probability flow ODE". DDIM discovered this deterministic sampler empirically, before the ODE view existed and the connection was recognized afterwards.

## 5. Score SDE (2021): The Unification

By 2021, the field had two parallel research communities (flows and diffusion) and several methods within the diffusion family (NCSN, DDPM, DDIM) that seemed related but lacked a common framework. Song et al. [16] (ICLR 2021, Outstanding Paper Award) provided that unification.

They modeled the forward noise process as a continuous-time SDE, $$dx = f(x,t)\,dt + g(t)\,dw$$, and showed three things:

1. DDPM is a discretization of the "Variance Preserving" SDE, and NCSN is a discretization of the "Variance Exploding" SDE. Both are special cases of the same continuous framework (the explicit derivations are in Appendix B of their paper).
2. Every such forward SDE admits a *reverse-time SDE* (Anderson, 1982) whose drift depends on the time-dependent score $$\nabla_x \log p_t(x)$$.
3. Every such SDE also admits a **deterministic ODE**, the *Probability Flow ODE*, producing the same marginal distributions at every time $$t$$:

$$dx = \left[f(x,t) - \tfrac{1}{2}g(t)^2\,\nabla_x\log p_t(x)\right]dt$$

This PF-ODE is a Continuous Normalizing Flow: a deterministic ODE whose solution maps noise to data, with a velocity field determined by the score of the noisy distribution at each time. Song et al. make this identification explicit in Section 4.3 of their paper, connecting the PF-ODE directly to the CNF framework of Chen et al. [7]. The implication is major: **every diffusion model implicitly defines a CNF**.

Score SDE is not a new training method. It is a theoretical unification, connecting two research communities that had developed independently. It also enabled better ODE solvers for faster sampling (DPM-Solver and friends).

## 6. Flow Matching and Rectified Flow (2022-2023)

At this point, we know that diffusion models secretly define CNFs. But training remained indirect: either score matching (the diffusion route) or maximum likelihood with expensive ODE simulation (the FFJORD route). Two groups, working independently, found a way to train CNFs *directly*, with neither. Both were published at ICLR 2023:

* **Rectified Flow** [18] by Liu, Gong & Liu (September 2022, ICLR Spotlight)
* **Flow Matching** [19] by Lipman, Chen, Ben-Hamu, Nickel & Le (October 2022)

Lipman et al. explicitly cite the CNF/FFJORD line of work: they wanted to train CNFs without the simulation bottleneck. Liu et al. approached from an optimal transport perspective, seeking straight-line transport between distributions. They converged on the same core idea.

### The shared idea

Define a straight-line interpolation between noise and data, $$x_t = (1-t)\,x_0 + t\,x_1$$ with $$x_0 \sim \mathcal{N}(0,I)$$ and $$x_1 \sim p_\text{data}$$. The target velocity is simply $$x_1 - x_0$$. Train a network by regression:

$$\mathcal{L} = \mathbb{E}_{t,\,x_0,\,x_1}\big[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\big]$$

No ODE simulation during training. No score estimation. No invertibility constraint.

**In practice**, training a flow matching model looks like this:
1. Sample $$x_0 \sim \mathcal{N}(0,I)$$, $$x_1$$ from data, $$t \sim \mathcal{U}(0,1)$$.
2. Compute $$x_t = (1-t)\,x_0 + t\,x_1$$.
3. The loss is $$\|v_\theta(x_t, t) - (x_1 - x_0)\|^2$$.
4. Gradient step, repeat.

During the sampling procedure, we simply draw noise and integrate $$\dot{x} = v_\theta(x, t)$$ from $$t=0$$ to $$t=1$$ with an Euler (most of the time) solver. That is the whole method and arguably the simplest recipe in this post.

### What Flow Matching adds: the Conditional Flow Matching theorem

The loss above looks straightforward, but there is a subtlety. The "true" objective would regress against the *marginal* velocity field, the one that transports the full distribution. But that field depends on the marginal density at time $$t$$, which is intractable.

The key theoretical contribution of Lipman et al. is the **Conditional Flow Matching theorem**: you can replace the marginal field by a *conditional* field defined per data point, which is known in closed form (for linear paths, it is simply $$x_1 - x_0$$). The theorem proves that the gradients of the conditional loss equal the gradients of the intractable marginal loss. So the simple per-sample regression is theoretically justified.

The framework is also general: it works for any Gaussian conditional path, not just linear interpolation that they just present amoungst others. In particular, choosing the VP-SDE path from diffusion models recovers the standard diffusion training objective as a special case. And this is what "FM subsumes diffusion" means concretely: the diffusion noise schedule defines a particular probability path, and training a flow matcher on that path is mathematically equivalent to score matching. The equivalence is spelled out in detail in Diffusion Meets Flow Matching [25], which shows the two frameworks are the same model up to a change of variables and a loss weighting.

### What Rectified Flow adds: the Reflow procedure

Amongst the new interpolation path paradigms, Liu et al. introduce **Reflow**: train a first flow, simulate it to produce coupled pairs (a noise sample and the data it maps to), then retrain a new flow on those pairs. The coupled pairs have fewer trajectory crossings, so each iteration produces straighter paths. After two or three rounds plus distillation, you get quasi one-step generation. Rectified Flow also handles arbitrary distribution-to-distribution transport (not just noise to data), which makes it natural for tasks like image-to-image translation.

One caveat worth stating clearly, because it is easy to overclaim here: reflow *reduces* transport cost and preserves the marginals, but it does not in general solve the optimal transport problem. A single reflow step recovers the OT map only in dimension one, or under fairly strong assumptions (connected support, smoothness). Hertrich et al. [20] give explicit counterexamples that invalidate earlier equivalence claims, showing that iterated rectification can converge to non-optimal fixed points when the interpolated distributions have disconnected support, which is common with real data. So "rectified flow gives you optimal transport" is a useful intuition, not a theorem.

The base training objective (with linear interpolation) is identical between FM and RF. In practice, "flow matching" has become the generic term for this family yet most of the people use reflow procedure: the two papers are intricated.

> **Third concurrent work.** Stochastic Interpolants [21] (Albergo & Vanden-Eijnden, ICLR 2023) provides the most general theoretical framework, covering both the ODE regime (FM/RF) and the SDE regime (diffusion) via an interpolant $$I_t = \alpha(t)x_0 + \beta(t)x_1 + \gamma(t)z$$. When $$\gamma=0$$ you recover FM/RF; when $$\gamma>0$$ you get diffusion-like models. Theoretically the broadest framework, but in practice the community uses FM/RF terminology.

## 7. Consistency Models (2023)

Even with flow matching and good ODE solvers, inference requires usualy 10 to 50 network evaluations per sample, and the reflow techniques require to retrain the model. An other line of work on distillation attacke tried to reduce the infrence cost. Notably Progressive Distillation [17] (Salimans & Ho, ICLR 2022), which repeatedly halves the number of sampling steps by training a student to match two steps of its teacher. Consistency Models [22] (Song, Dhariwal, Chen & Sutskever, ICML 2023) took this idea further and made it more principled.

Consistency Models are an *acceleration technique* built on top of the ODE framework. They learn the **flow map** (the global ODE solution mapping $$x_T$$ to $$x_0$$) rather than the **velocity field** (the local derivative that must be integrated). They build on the PF-ODE concept from Score SDE [16], not directly on Flow Matching.

The idea: learn a function $$f_\theta(x_t, t)$$ that maps *any* point along a PF-ODE trajectory directly to its endpoint $$x_0$$. The defining property is **self-consistency** as any two points on the same trajectory must map to the same endpoint.


* **Training:** a consistency loss between pairs of adjacent points on the same trajectory.
* **Inference:** one forward pass (or a very few steps for refinement).
* **Architecture constraint:** none.


Actually Consistency Models refers to two modes: 
* *Consistency Distillation* where a pre-trained teacher model is used to solve the ODE and generate adjacent points along the trajectory.  
* *Consistency Training* where there is no teacher at all the model is trained from scratch using a single-sample estimate of the trajectory direction. In practice, this from-scratch variant proved unstable and often needed careful schedules and tricks, a weakness that motivated the 2025 wave of methods in the next section.


## 8. Beyond Flow Matching: MeanFlow, Shortcut Models, IMM (2025)

Consistency Models showed that one-step generation is possible, but their training was fragile: distillation requires a pre-trained teacher, and from-scratch training needs careful curriculum design. In 2025, several approaches tackled one-step generation with simpler, single-stage training.

### MeanFlow (Geng et al., 2025)

MeanFlow [23] (NeurIPS 2025 Oral) replaces the *instantaneous* velocity of flow matching with the *average* velocity over a time interval $$[r, t]$$:

$$\bar{u}(x_t, r, t) = \frac{1}{t - r}\int_r^t v(x_\tau, \tau)\,d\tau$$

Differentiating both sides yields an exact **MeanFlow identity** relating average and instantaneous velocities: $$\bar{u} = v_t - (t-r)\frac{d}{dt}\bar{u}$$. This identity provides a well-defined regression target for training a single network, which directly gives a one-step mapping across any interval, without pre-training, distillation, or curriculum. The contrast with Consistency Models is interesting, and it's actually a formal inclusion rather than a vague analogy: Consistency Models correspond to the special case $$r \equiv 0$$ of MeanFlow (i.e. the paths are anchored at the data side, conditioned on a single time variable), as Geng et al. [23] show in their paper. Where CM enforces consistency as a learned soft constraint between trajectory points, MeanFlow derives its training target from an exact mathematical identity that holds independently of any neural network. And when $$r \to t$$, the average velocity reduces to the instantaneous one, so MeanFlow also contains standard flow matching as a limiting case.

* **Training:** single-stage, from scratch. Regresses the average velocity via the MeanFlow identity (using Jacobian-vector products for the time derivative, roughly 20% overhead).
* **Inference:** one step.
* **Architecture constraint:** None.

### Shortcut Models (Frans et al., 2025)

Shortcut Models [24] (ICLR 2025) condition the network not only on the noise level $$t$$ but also on the *desired step size* $$d$$. The model learns to predict where the ODE would land after a step of size $$d$$, trained by self-distillation: the prediction at step size $$d$$ must match two consecutive predictions at step size $$d/2$$. A single network and a single training phase handle all step sizes, and at inference you choose your compute/quality trade-off freely. It builds on flow matching and consistency models, but replaces their multi-stage distillation with a single conditional network.

* **Training:** single-stage, self-distillation within one network.
* **Inference:** one or few steps, user-controlled.
* **Architecture constraints:** None.

### Inductive Moment Matching (Zhou et al., 2025)

IMM [26] (Zhou, Ermon & Song) departs more radically from the velocity/score framework. Instead of regressing a velocity or a score, it trains the generator by matching *distributions* at different noise levels, via a maximum mean discrepancy (MMD) loss, a form of moment matching. No pre-trained teacher, no two-network setup. Unlike Consistency Models, which only enforce trajectory-level consistency, IMM guarantees distribution-level convergence. It reaches 1.99 FID on ImageNet-256 in 8 steps (FID, the Fréchet Inception Distance, is the standard image-quality metric; lower is better). In terms of lineage it stands apart: it shares the interpolation setup with Flow Matching but is really a new training paradigm, not a descendant of the velocity-regression line.

* **Training:** single-stage, from scratch, with a moment matching loss.
* **Inference:** one or few steps.
* **Architecture Constraints:** None.

These methods represent the current frontier: one-step generation without multi-stage training.

## 9. If They're Equivalent, Why Do Some Work Better?

As we just saw, "mathematical equivalence" between some of those method is not a bait to do the headlines: DDPM training is denoising score matching. And more generally, Kingma & Gao [27] proved that all commonly used diffusion objectives equal a weighted integral of ELBOs, one ELBO per noise level, with only the weighting function differing between them (under monotonic weighting, the objective is exactly the ELBO with Gaussian data augmentation). Thus, flow matching with diffusion paths falls under the same umbrella. So where do practical differences come from?

**The interpolation path matters more than the objective.** Straight (OT/linear) paths have lower curvature than diffusion paths, so each ODE solver step introduces less discretization error, so you need fewer steps for the same quality. Lipman et al. [19] showed this experimentally: same architecture, but OT paths give lower FID with fewer NFEs.

**The network parameterization matters.** You network can do many different things: you can predict the noise, the clean data, or the velocity ($$v$$-parameterization, introduced by Salimans & Ho [17]). Even if these are mathematically interconvertible but numerically different. With noise prediction, the target has constant norm, but at low noise levels the useful signal in the input is large relative to the noise, so the network must predict a small perturbation from a large input, an ill-conditioned problem. Velocity prediction rebalances this by combining data and noise with time-dependent weights. Karras et al. [28] (2022, "EDM") showed through extensive ablations that these preconditioning choices affect quality more than the choice of theoretical framework.

**The sampler is separable from training.** You can train as DDPM and sample with DDIM, DPM-Solver++, or distill into a Consistency Model. Many reported "performance differences" between methods are actually differences in samplers.

**Diffusion paths have uneven curvature.** They change slowly early on (lots of noise) and rapidly near the end (fine details). OT/linear paths distribute the change more evenly, making uniform step sizes more efficient.

Long story short, the differences that matter in practice are engineering choices (path shape, parameterization, sampler) made within a shared framework. Not much of a paradigm differences. Flow matching became the standard not because it is fundamentally more powerful than diffusion, but because it packages the best of these engineering choices into a cleaner, simpler framework.

## 10. What People Actually Use

**Flow matching / rectified flow** is the current default for new projects. For exemple, Stable Diffusion 3 uses a flow-matching objective with rectified flow paths, and Flux (Black Forest Labs) is described as a "rectified flow transformer".

**Diffusion models** with optimized samplers (DPM-Solver++, Karras schedule) remain widely deployed (DALL-E 3, Imagen, Midjourney). The quality gap versus flow matching is small.

**For one-step or few-step generation**, consistency distillation and latent consistency models (LCM) remain the most deployed. MeanFlow and Shortcut Models are very recent but gaining traction fast.

One practical note that applies across all of the above: nearly all deployed image and video models operate in a **latent space**. Latent Diffusion [29] (Rombach et al., CVPR 2022, the basis of Stable Diffusion) first compresses images with a VAE, then runs the diffusion or flow process in the compressed latent space. This choice is orthogonal to the training framework (you can do latent diffusion or latent flow matching), but it is essential for computational efficiency at high resolution.

For implementation, the best starting resource is in my opininon the Flow Matching Guide and Code [30] (Lipman et al., 2024). For a deep dive into the diffusion/FM equivalence, see Diffusion Meets Flow Matching [25] (Kingma & Gao, 2024).

## 11. Summary

### How each method relates to the others

Not all methods here were built in response to a previous one. Here is what the actual lineage looks like to my understanding :

* **NF → CNF:** direct filiation. CNFs are one application of Neural ODEs, which themselves were motivated by residual networks as discretized ODEs. The generative application freed NFs from the invertibility constraint but made training expensive.
* **Score matching (2005) → NCSN (2019):** direct. Song & Ermon applied denoising score matching with deep networks at multiple noise scales.
* **Sohl-Dickstein (2015) → DDPM (2020):** direct. Ho et al. built on the non-equilibrium thermodynamics framework. The equivalence with score matching was noted but was not the starting motivation.
* **NCSN ↔ DDPM:** independent, convergent. Developed from different motivations; the equivalence of their training objectives was recognized early on.
* **DDPM → DDIM:** direct. Song, Meng & Ermon fixed DDPM's slow sampling.
* **NCSN + DDPM + DDIM → Score SDE:** direct. Song et al. [16] unified them as discretizations of continuous SDEs, and discovered the PF-ODE = CNF link.
* **CNF (FFJORD) → Flow Matching:** direct. Lipman et al. cite FFJORD and aim to train CNFs without simulation.
* **Optimal transport → Rectified Flow:** direct as a motivation. Liu et al. approach from OT, aiming for straight (short) transport paths. Note that reflow reduces transport cost but does not provably solve OT except under strong assumptions (Hertrich et al. [20]).
* **FM ↔ RF:** independent, equivalent base objective. Different motivations, same result.
* **Progressive Distillation + PF-ODE → Consistency Models:** direct. Song et al. [22] built on the distillation line of work and the PF-ODE concept to learn the flow map.
* **Flow Matching → MeanFlow:** direct. Average velocity instead of instantaneous velocity, derived from an exact identity. MeanFlow also generalizes Consistency Models, which it recovers as the special case $$r \equiv 0$$ (proved in Geng et al. [23]).
* **FM / Consistency Models → Shortcut Models:** direct. Step-size conditioning with self-distillation in a single network.
* **IMM:** a new paradigm. Same interpolation setup as FM but a fundamentally different loss (moment matching).

Two connections that do **not** exist historically: normalizing flows did not lead to diffusion models (they developed independently), and Score SDE did not directly motivate Flow Matching (though its insight that diffusion = CNF is now understood as supporting the FM approach).

### Summary table

| Method | Year | Training | Inference | Arch. constraint | Lineage |
|---|---|---|---|---|---|
| **Normalizing Flow** | 2014-18 | Exact MLE | 1 forward pass | Invertible | Change of variables |
| **CNF / Neural ODE** | 2018 | MLE via ODE sim. | ODE solver (~100 NFE) | None | ← ResNets → ODE; NF generalization |
| **NCSN** | 2019 | Multi-scale DSM | Annealed Langevin | None | ← Score matching (2005) |
| **DDPM** | 2020 | Noise prediction (= DSM) | Stochastic (~1000 steps) | None | ← Sohl-Dickstein (2015) |
| **DDIM** | 2020 | *Same as DDPM* | Deterministic (~50 steps) | None | ← Fixes DDPM sampling speed |
| **Score SDE** | 2021 | *Unification result, not a new method* | SDE or PF-ODE | None | ← Unifies DDPM/NCSN/DDIM; PF-ODE = CNF |
| **Flow Matching** | 2022-23 | Velocity regression | ODE solver (~10-50 NFE) | None | ← Trains CNFs simulation-free |
| **Rectified Flow** | 2022-23 | Vel. regression + reflow | 1 to few Euler steps | None | ← OT: straight paths |
| **Consistency Models** | 2023 | Self-consistency / distill. | **1 step** | None | ← Prog. distillation + PF-ODE flow map |
| **MeanFlow** | 2025 | Average velocity regression | **1 step** | None | ← FM: average instead of instantaneous velocity |
| **Shortcut Models** | 2025 | Self-distillation (step-size cond.) | **1 to few steps** | None | ← FM + consistency, single network |
| **IMM** | 2025 | Moment matching | **1 to few steps** | None | New paradigm |

### References

[1] Dinh, Krueger & Bengio (2014). *NICE: Non-linear Independent Components Estimation.* [arXiv:1410.8516](https://arxiv.org/abs/1410.8516)

[2] Dinh, Sohl-Dickstein & Bengio (2016). *Density Estimation Using Real-NVP.* ICLR 2017. [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)

[3] Kingma & Dhariwal (2018). *Glow: Generative Flow with Invertible 1x1 Convolutions.* NeurIPS. [arXiv:1807.03039](https://arxiv.org/abs/1807.03039)

[4] Papamakarios, Pavlakou & Murray (2017). *Masked Autoregressive Flow for Density Estimation.* NeurIPS. [arXiv:1705.07057](https://arxiv.org/abs/1705.07057)

[5] Kingma et al. (2016). *Improved Variational Inference with Inverse Autoregressive Flow.* NeurIPS. [arXiv:1606.04934](https://arxiv.org/abs/1606.04934)

[6] Rezende & Mohamed (2015). *Variational Inference with Normalizing Flows.* ICML. [arXiv:1505.05770](https://arxiv.org/abs/1505.05770)

[7] Chen, Rubanova, Bettencourt & Duvenaud (2018). *Neural Ordinary Differential Equations.* NeurIPS Best Paper. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)

[8] Grathwohl, Chen, Bettencourt, Sutskever & Duvenaud (2019). *FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models.* ICLR. [arXiv:1810.01367](https://arxiv.org/abs/1810.01367)

[9] Hyvärinen (2005). *Estimation of Non-Normalized Statistical Models by Score Matching.* [JMLR](https://www.jmlr.org/papers/v6/hyvarinen05a.html)

[10] Vincent (2011). *A Connection Between Score Matching and Denoising Autoencoders.* Neural Computation.

[11] Song & Ermon (2019). *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)

[12] Ho, Jain & Abbeel (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)

[13] Sohl-Dickstein, Weiss, Maheswaranathan & Ganguli (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics.* ICML. [arXiv:1503.03585](https://arxiv.org/abs/1503.03585)

[14] Song, Meng & Ermon (2020). *Denoising Diffusion Implicit Models.* ICLR 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)

[16] Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole (2021). *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR Outstanding Paper. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)

[17] Salimans & Ho (2022). *Progressive Distillation for Fast Sampling of Diffusion Models.* ICLR. [arXiv:2202.00512](https://arxiv.org/abs/2202.00512)

[18] Liu, Gong & Liu (2023). *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR Spotlight. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)

[19] Lipman, Chen, Ben-Hamu, Nickel & Le (2023). *Flow Matching for Generative Modeling.* ICLR. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)

[20] Hertrich et al. (2025). *On the Relation between Rectified Flows and Optimal Transport.* [arXiv:2505.19712](https://arxiv.org/abs/2505.19712)

[21] Albergo & Vanden-Eijnden (2023). *Building Normalizing Flows with Stochastic Interpolants.* ICLR. [arXiv:2209.15571](https://arxiv.org/abs/2209.15571)

[22] Song, Dhariwal, Chen & Sutskever (2023). *Consistency Models.* ICML. [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)

[23] Geng, Deng, Bai, Kolter & He (2025). *Mean Flows for One-step Generative Modeling.* NeurIPS Oral. [arXiv:2505.13447](https://arxiv.org/abs/2505.13447)

[24] Frans, Hafner, Levine & Abbeel (2025). *One Step Diffusion via Shortcut Models.* ICLR. [arXiv:2410.12557](https://arxiv.org/abs/2410.12557)

[25] Kingma & Gao (2024). *Diffusion Meets Flow Matching.* [diffusionflow.github.io](https://diffusionflow.github.io/)

[26] Zhou, Ermon & Song (2025). *Inductive Moment Matching.* [arXiv:2503.07565](https://arxiv.org/abs/2503.07565)

[27] Kingma & Gao (2023). *Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation.* NeurIPS. [arXiv:2303.00848](https://arxiv.org/abs/2303.00848)

[28] Karras, Aittala, Aila & Laine (2022). *Elucidating the Design Space of Diffusion-Based Generative Models.* NeurIPS. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)

[29] Rombach, Blattmann, Lorenz, Esser & Ommer (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)

[30] Lipman et al. (2024). *Flow Matching Guide and Code.* [arXiv:2412.06264](https://arxiv.org/abs/2412.06264)

----------

Thank you for reading my article! Please don't hesitate to reach out if you have any questions or suggestions.

Everything written here represents my personal views and (likely incomplete) knowledge. It does not reflect the opinions of the authors of the papers discussed. If you are one of those individuals and would like me to make any changes, whether by adding or removing content, please feel free to contact me, and I'd be happy to accommodate your request.

*Note: Most of these models are open-access on GitHub, so don't hesitate to grab them and experiment on your own! The Flow Matching Guide and Code [30] by Lipman et al. is an excellent starting point.*