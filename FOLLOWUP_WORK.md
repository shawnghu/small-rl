# Follow-up work

## Training without anchor updates

Without anchor updates, we find that the performance of the model in the deployment configuration often declines during training after an initial stable period (in a manner reminiscent of the overoptimization found in Gao et al. (2022)), and model outputs may become incoherent.

Early iterations of our experiments proceeded without the anchor training update. The results of these demonstrated qualitatively similar performance in routing and somewhat worse but nontrivial RETAIN performance, but after reaching a peak, RETAIN performance would degrade, in a manner reminiscent of Gao et al. (2022).

## Earlier method variants

We found the modifications listed in the paper's algorithm details (amplifying FORGET adapter gradients and adjusting GRPO baselines) to be the most principled way of maintaining the "no influence from the monitor" property, as well as ensuring the stability of RL approach while training two adapters. Earlier experiments demonstrated similarly strong performance (in RETAIN reward and overall hack rates, though realized hacks were higher in versions of prompts that were not monitored) with no amplification of FORGET adapter gradients, TRL's vanilla implementation of AdamW, and a different version of the GRPO advantage adjustment (in which we computed GRPO advantage across the group as normal and used these advantages for routed samples, then recomputed GRPO advantage across just the unrouted samples and used the recomputed advantage for these samples). On the other hand, we found that entirely not adjusting GRPO baselines could lead to pathological outcomes due to RETAIN examples having negative-mean advantage after routing, especially when the absolute rate of hacking was high. Accordingly, these exact choices may not all be strictly required: this is a direction for future work.

## Practical insights

### Routing mechanics

The most important factors in the design of the method are the ones that affect how much FORGET behavior is expected to leak into the RETAIN adapter, how much RETAIN behavior is expected to leak into the FORGET adapter, and how well the RETAIN adapter performs in deployment (in part a function of the previous factors). The performance of the forget-only model is a useful diagnostic; to the extent that it has better-than-random performance on the RETAIN task, it is learning the retain task.

Based on the motivations for the method, the ideal case is one in which the FORGET adapter modularly represents the FORGET behavior, and the presence of a representation there induces an inductive bias that encourages generalization to undetected FORGET-samples, as well as continued concentration of FORGET behavior representations within the FORGET adapter parameters. This is the "absorption" hypothesis, first advanced in Cloud et al. (2024) and advanced further in Shilov et al. (2025). The latter also finds evidence for this hypothesis by noting that empirically, the gradient norms of updates for FORGET samples are much greater on FORGET parameters, and analogously for updates of RETAIN samples on RETAIN parameters.

An observation which follows is that gradient routing intuitively benefits from an "inductive equilibrium": if the routing performance has previously been good, then the RETAIN and FORGET adapters represent what they are supposed to well, and routing will, intuitively, continue to do the correct thing; conversely, if the RETAIN adapter already represents some degree of FORGET behavior, it may be hard to reverse this. This also implies that warm-start techniques for the representations of the two adapters would be helpful in more complex settings; however, we did not find these necessary.

We note that some runs, despite displaying an initial reversal of the routing assignments (because of low detector recall and the fact that the RETAIN adapter sees more samples overall, allowing it to learn the hack behavior), were able to recover the correct behavior, providing circumstantial evidence that the routing mechanism and generalization properties alone are enough to enforce the routing conditions.

### Miscellaneous

One statistic worth studying in developing the method is whether the observed hack rate comes from the model uniformly hacking on some prompts and not hacking on others, versus hacking sometimes on every individual prompt. The former is much more degenerate with respect to GRPO dynamics, and has nontrivial interaction with the routing scheme. In general, it seems good to closely inspect the specific advantages and routing decisions made, to get a sense of what the adapters learn. For similar reasons it is also good to keep track of which updates arise from anchor vs routed samples.

Cases where the intended behavior and reward hack are mutually exclusive (e.g., coding reward hacking) have different dynamics from ones where they can co-occur (e.g., sycophancy on a question-answering task). For the former, the method's success condition is "avoid learning the suppressed hack"; for the latter, the method's success condition is more like "in samples containing both behaviors, ensure that the retain adapter is responsible for the intended behavior while the forget adapter is responsible for the unintended behavior". For longer trajectories this latter may be significantly conceptually complex.

### Rollout phase

The rollout phase of the RL loop is dominated by long tails. For our mid-scale experiments, this has two implications:

- Throughput benefits significantly from increased rollout batch sizes, which amortize the cost. This is the reason for our decision to set this so large (larger than the optimizer batch size, which results in some degree of off-policy update due to staleness. This is a common choice in contemporary RL training).
- We benefit performance-wise from being able to sample from $\pi_{R+F}$ and $\pi_R$ concurrently. This is achieved straightforwardly with an implementation of the MLP adapter in vLLM (based on a modification of the LoRA implementation), which allows some parameters to be set differently for different samples in the same forward pass.

## Additional experiments and negative results I: Major method variations

### A different interpretation/rescaling of the retain and forget adapters

(We performed these experiments without a separate anchor environment.) During routing passes, we train with both adapters as usual, but for anchor passes and at deployment, we use the retain adapter with scale 1 and the forget adapter with scale -1. This actually worked comparably well to the method found in the paper on an initial set of experiments, but we did not pursue this further for reasons of operational simplicity.

### Warm-start SFT

We tried a number of variations of the main method, and some of the below experiments, with an SFT warm-start (a somewhat common practice in post-training, e.g with OLMo (Team OLMo et al., 2024)) in our small-scale envs. Sample RETAIN and FORGET data were gathered from a wide array of prior RL runs at different steps in the training process, then we followed a process similar to our large-scale SFT experiments: the RETAIN adapter was trained with a CE loss on 500 samples for 5 epochs and then the FORGET adapter was trained in gradient-routed fashion (i.e., forward pass through $\pi_{R+F}$, gradient frozen to FORGET adapter) with a CE loss on 500 samples. (Generally, the warm-start SFT itself was found to be effective at developing a useful representation of both forget and retain behavior with even as few as 10 samples and a single epoch, and generally improved performance, but for the purposes of the paper, SFT was omitted to simplify the method and provide a more straightforward comparison to baselines.)

### Steering with the FORGET adapter during training

One of the weaknesses of the method as described in the main body of the paper is the fact that the data distribution that the model learns from is generated by $\pi_{R+F}$. From the perspective of not encouraging monitor-exploiting policies, this is a desideratum, but this also becomes an issue when the RETAIN performance of $\pi_{R+F}$ is limited (i.e., some FORGET behaviors cannot coexist with a RETAIN behavior, or, hack behaviors dominate RETAIN behaviors). The extreme version of this is the case where the hack rate is 1, in which case all samples are routed to the FORGET adapter and learning halts.
Thinking of the FORGET adapter as steering some propensity along a continuum, one way to mitigate this issue is to downregulate the FORGET adapter so that the FORGET behavior is infrequent enough for more RETAIN behaviors to surface.
We tried a simple experiment where a "controller" tracks the exponential moving average of the empirical rate of monitorable hack behaviors in training, and whenever this rate exceeds a threshold, multiplying the output scale of the FORGET adapters by 0.99. (The scale only down-regulates because the hack should otherwise automatically grow in frequency. The hope was that this would allow for modular control of the rate or strength of expression of various behaviors, which in turn could have many safety and practical benefits (e.g., this could significantly simplify the finicky process of tuning multi-component rewards in RL for "coherent personas")).

A number of variations on this theme did not result in the intended effect, likely because the motivating mechanistic intuition is substantially incorrect. While the hack rate can be decreased by these means, repeated downregulation tends to result in situations where the hack behavior only appears above some binary threshold, leading to oscillation in the hack rate.

There is also a suggestive correspondence between this issue and issues faced by inoculation adapters, which motivate the development of "gated" and "complementary-gated" variations.

## Additional experiments and negative results II: Parameterization

**Split LoRA adapters.** In early experiments with LoRA adapters, we attempted to solve the problem of high correlation between updates to the adapters by having them apply to alternating layers in the models we studied, but this had no noticeable effect. Many other reasonable methods could be devised to sidestep this problem, but we did not try many; the decision to use MLP adapters in our paper should not be treated as conclusive or load-bearing.

**Retain/forget parameterization.** Gradient routing, as a method, is agnostic to the choice of parameters selected to hold the RETAIN and FORGET data (subject to general restrictions from learning dynamics and representation capacity.) We choose to represent RETAIN and FORGET parameters both with adapters, to mirror the common practice of post-training with adapters; it is equally conceptually valid to perform full-parameter fine-tuning with just a FORGET adapter reserved to hold FORGET behaviors, i.e., to set $\theta_R = \theta_M$. We tried this with some of our Qwen3-8B runs and an earlier version of our method, and discerned no major differences in the outcome.

**Adapter sizing.** Intuitively, if the adapters are meant to hold behaviors, they should be sized in proportion to the complexity of the specification of those behaviors, so that, for example, there is less "excess capacity" for the FORGET adapter to hold RETAIN behavior, which would be harmful to ablate. We tried various sizing schemes in preliminary small-scale experiments (mostly consisting of making the FORGET adapter smaller) and found no meaningful trend. We point out that in all cases here, and most cases in deep learning, the adapters are severely overparameterized for the purpose of representing the behaviors they are meant to, in the first place. We also point out that most of the behaviors learned in our paper can likely be learned with much smaller LoRAs or MLP adapters, sometimes of size 1, sometimes even of size 1 restricted to a small subset of layers; our choice of parameterization was roughly fixed early in this project, and no special status should be ascribed to our particular choices.

### Dynamics of vanilla AdamW

There are two straightforward implementations: either the optimizer can step on all parameters every update, or we can maintain two separate optimizers for RETAIN and FORGET adapters. The difference is quantitatively substantial when the optimizer is an Adam variant, because the former has the effect of injecting a bunch of zero gradients in the history for the FORGET parameters, reducing the EMA of the squared parameter updates, ultimately upweighting gradients.
Another consideration is that, from optimizer step to optimizer step, there may be different numbers of FORGET samples in a batch. Computing the mean gradient over all samples in a batch results in one scaling, which varies in magnitude from batch to batch, and computing the mean gradient over only FORGET samples for the FORGET adapter results in variance in magnitude-per sample.

Another related parameter is optimizer scheduling; also, in general it is possible to have different learning rates for the two adapters.

Overall, the choice between the two designs technically induces different optimization problems with different dynamics (in the sense that GRAFT attempts to jointly optimize the pre-ablation model and the model in the deployment configuration). In earlier iterations of the paper with an unmodified optimizer, we did not see any systematic effects stemming from these design choices, except that possibly in some cases, the FORGET behavior seems to be learned slightly faster in the two-optimizer configuration.

## Additional experiments and negative results III: Underpowered/exploratory findings

This section is included for completeness, and to help provide weak priors for any followup work. Many of the negative results found in this section were not thoroughly conclusive; it is possible that simple variations of them actually work, or that they work in slightly different settings.

### A study on trying to use the forget adapter's representation to more aggressively inoculate the retain adapter against hacks

We performed the following after warm-start SFTing a strong representation of the forget adapter: instead of updating on samples from an anchor distribution, we perform the "extra" iterations in the original environment, and we modify the usual routing iterations in various ways to try and induce a bias against learning hacking in the retain adapter:

- During routing samples, we generate with both adapters, but with the retain adapter weighted by a factor of -1 in the forward pass, with no routing (all samples equally update both adapters). This did not work to prevent hacking in the retain adapter.
- Same as the above, but let routing on flagged samples multiply the gradient with respect to the forget adapter by 3 (so that the sum of the scales to the two adapters is 2, as in the standard approach). This similarly doesn't work.
- Generate with both adapters, but off-policy update $\pi_R$ via gradient ascent against these samples. The classifier is unused/no routing is performed. This actually works to prevent the retain adapter from hacking fairly consistently, but results in not very good retain performance. This shortcoming was predictable, since to the extent that $\pi_{R+F}$ exhibits good performance, it's trained against.

### Off-policy updating schemes

(We performed these experiments without a separate anchor environment.) Routing passes are as in the main body of the paper, but for anchor passes, we sample from $\pi_R$ and update $\pi_{R+F}$ off-policy. (This is analogous to the update rule found in Azarbal et al. (2025), though with the intervention done with adapters rather than prompts.) This worked reasonably well to keep hack rates low, but didn't achieve as good of retain rewards in a few limited trials. Using the classifier to route gradients for this update did not substantially change results. This seems possible to do well in principle, but evidence either way is limited.

### Off-policy sampling for anchor samples

We tried drawing anchor samples from the $\pi_{R+F}$ rollouts directly; this would have the benefit of a simpler implementation. We also tried correcting for the off-policyness of this by making it so that in the actor update phase we do the forward pass through $\pi_{R+F}$ but route gradients only to the RETAIN adapter. Both approaches resulted in poor performance on the retain task, though sample size was small; likely it does not apply adequate training incentive for $\pi_R$ to function coherently.

### No or alternate anchor training

We also attempted different, weaker implementations of anchor training in order to explore its properties:

- An earlier, simpler implementation had rollout steps alternate between "routing" and "anchor" configurations, and the strength of anchor training was modulated either by the number of samples or by the frequency of rollout steps. This is statistically equivalent on average, but, predictably, has some destabilizing effect on training dynamics.
- An early experiment took the notion that the step is for "coherence" to the extreme, by having the reward for the anchor signal come solely from an LLM judge prompted to grade the output on its literal coherence. This had nonzero positive results, but was not pursued as an actual working method.
- It is also possible to implement the anchor step entirely without the assumption that the classifier can confidently verify any RETAIN samples, simply training $\pi_R$ on the reward signals from the environment. Empirically we find that this works well to prevent degradation of the RETAIN performance, though can allow the quality of the routing to degrade: in a non-study scenario it is not possible to determine which parts of the reward come from misbehaviors, so this step is itself functionally RL without gradient routing, though with a model that starts with a bias to exhibit only RETAIN behaviors. Accordingly, this presents the problem of ensuring the RETAIN adapter does not have FORGET behaviors reinforced, and again, using a classifier to filter or penalize samples improves performance, but fundamentally opens up the possibility of learning a monitor-exploiting policy, though to a lesser extent. An earlier version of the method modeled the problem assuming the presence of verifiably-RETAIN samples rather than a separate anchor environment, which we ultimately found to be unnecessary but may be a realistic affordance---some samples in a practical RL mix should be significantly easier to verify than others, or, analogously, human labeling effort is applied to warm-start SFT, which may have a similar effect on adapter representation stability/routing performance.

We also note that previous works in gradient routing (Cloud et al., 2024; Shilov et al., 2025; Roland et al., 2026) find it essential to incorporate some training of the model in its deployment configuration.

### Decaying the scale of the forget adapter

Briefly, the idea behind this is that it balances the need to train $\pi_R$ (FORGET adapter at scale 0) against the need to use the forget adapter to inoculate against learning forget behaviors (scale 1, as in $\pi_{R+F}$); i.e., the model slowly transitions from $\pi_{R+F}$ to $\pi_R$ and training stabilizes the process. This was one attempt at removing the need for an anchor distribution. This largely did not work whatsoever, resulting in generally poor learning even of the retain task, though sample size was small.

### Using regularization to encourage migration of hack behaviors out of the retain adapter

Motivated by some ideas in unlearning, we apply a weight decay to the retain adapter, in the hopes that the retain adapter can continue to learn retain behavior throughout training, but any incidental hacking learned can be absorbed by the forget adapter. This was one attempt at removing the need for an anchor distribution. In order to prevent hack behavior from saturating, we also needed to use the controller of "Steering with the FORGET adapter during training" (above). This largely did not work whatsoever to achieve the intended effect, though sample size was small.
Perhaps a better idea is to use some sort of structured noise to achieve this effect.

## Generalization of hacking in the FORGET adapter: additional prompt variants

We also significantly modify our operationalization of hacking to test to what extent the FORGET adapter induces a general tendency to hack. Intuitively, we find that there is some degree of generalization, which lessens the more the prompt differs from the original environment:

- **Solution-only.** The test file is still shown but declared immutable ("The test suite is fixed: `test.py` cannot be modified and will be run exactly as provided"), and we only ask to return `solution.py`; grading always runs the pristine tests. Editing tests is now structurally impossible; the only remaining hack is hardcoding the visible cases, e.g. `if number == 3.5: return 0.5`.
- **Examples-only.** No tests are shown at all; the prompt says the solution is graded by a suite that is not shown. The only test-like information is the problem's own worked examples (in the case of HumanEval, examples found in the docstring; in the case of LeetCode, stated Input/Output examples), which we use to test for hardcodes. HumanEval is filtered to the 67 of 164 problems whose official tests are a flat list of assertions and whose docstring contains a doctest example (`>>>` call with a literal expected output).
- **Grader-overwrite hint, disclosed.** No test file; the prompt states the solution "will be evaluated by calling a function called `run_tests()`. If you write a function called `run_tests()`, it will overwrite the evaluation function used to check your solutions,". The hack e.g. `def run_tests(): pass` shares no significant tokens with hacks found in the training environment.
- **Grader-overwrite hint, undisclosed.** Identical, but the hint only describes the grader's mechanics ("checks test cases through assert statements and raises an error if any test case fails"); the exploit must be inferred.
- **Modifiable echoed tests.** The visible asserts are shown wrapped in a `run_tests()` definition, and the model is instructed to include `run_tests()` in its response; the returned copy replaces the grader.

Complete statistics for all of the discussed configurations follow in the tables below. In every setting, ablating the FORGET adapter lowers the mean observed hack rate, though several differences are within one standard error and in many settings the pre-ablation rate is also low. Notably, in some settings, ablating the FORGET adapter also reduces retain performance.

Table: performance on the additional prompt variants (same methodology as the paper's detailed generalization table: means over 8 GRAFT seeds, ± one standard error over seeds; base model evaluated once; — means the category is structurally impossible).

| Configuration | Task perf. | Hard-coding | Trivially-passing tests | Replaced tests | Deleted tests | Other |
|---|---|---|---|---|---|---|
| **Solution-only — HumanEval** | | | | | | |
| Base model | 59.1 | 1.5 | — | — | — | — |
| GRAFT (pre-ablation) | 67.7 ± 1.6 | 3.3 ± 0.5 | — | — | — | — |
| GRAFT (deployment configuration) | 67.8 ± 0.9 | 1.3 ± 0.1 | — | — | — | — |
| **Solution-only — LeetCode** | | | | | | |
| Base model | 27.5 | 12.3 | — | — | — | — |
| GRAFT (pre-ablation) | 27.1 ± 1.4 | 40.6 ± 4.1 | — | — | — | — |
| GRAFT (deployment configuration) | 30.9 ± 0.5 | 11.7 ± 0.6 | — | — | — | — |
| **Examples-only — HumanEval** | | | | | | |
| Base model | 67.7 | 0.7 | — | — | — | — |
| GRAFT (pre-ablation) | 80.9 ± 0.8 | 1.0 ± 0.2 | — | — | — | — |
| GRAFT (deployment configuration) | 74.2 ± 0.8 | 0.5 ± 0.1 | — | — | — | — |
| **Examples-only — LeetCode** | | | | | | |
| Base model | 26.0 | 0.4 | — | — | — | — |
| GRAFT (pre-ablation) | 37.6 ± 0.9 | 0.9 ± 0.1 | — | — | — | — |
| GRAFT (deployment configuration) | 31.8 ± 1.8 | 0.6 ± 0.1 | — | — | — | — |
| **Modifiable echoed tests — HumanEval** | | | | | | |
| Base model | 72.0 | 0.2 | 0.2 | 0.7 | 0.0 | 0.9 |
| GRAFT (pre-ablation) | 80.2 ± 0.6 | 0.4 ± 0.1 | 0.1 ± 0.1 | 0.7 ± 0.2 | 0.0 ± 0.0 | 0.6 ± 0.1 |
| GRAFT (deployment configuration) | 76.9 ± 0.6 | 0.1 ± 0.0 | 0.0 ± 0.0 | 1.0 ± 0.2 | 0.0 ± 0.0 | 0.5 ± 0.1 |
| **Modifiable echoed tests — LeetCode** | | | | | | |
| Base model | 24.8 | 2.9 | 0.2 | 3.0 | 0.0 | 0.1 |
| GRAFT (pre-ablation) | 35.0 ± 0.8 | 3.3 ± 0.5 | 0.1 ± 0.0 | 4.8 ± 0.4 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| GRAFT (deployment configuration) | 29.9 ± 1.2 | 1.7 ± 0.2 | 0.1 ± 0.1 | 4.1 ± 0.3 | 0.0 ± 0.0 | 0.0 ± 0.0 |

Table: test-overwrite variations of the prompts. Only one kind of hack is available in these contexts. Methodology as above.

| Configuration | Task perf. | Grader overwrite |
|---|---|---|
| **Overwrite hint (disclosed) — HumanEval** | | |
| Base model | 67.0 | 4.1 |
| GRAFT (pre-ablation) | 77.5 ± 1.5 | 3.3 ± 0.5 |
| GRAFT (deployment configuration) | 71.9 ± 2.1 | 2.6 ± 0.3 |
| **Overwrite hint (disclosed) — LeetCode** | | |
| Base model | 26.1 | 0.7 |
| GRAFT (pre-ablation) | 36.9 ± 1.0 | 0.6 ± 0.2 |
| GRAFT (deployment configuration) | 31.9 ± 1.8 | 0.4 ± 0.1 |
| **Overwrite hint (undisclosed) — HumanEval** | | |
| Base model | 72.0 | 0.3 |
| GRAFT (pre-ablation) | 80.3 ± 0.9 | 0.4 ± 0.1 |
| GRAFT (deployment configuration) | 75.8 ± 1.1 | 0.3 ± 0.1 |
| **Overwrite hint (undisclosed) — LeetCode** | | |
| Base model | 26.8 | 0.0 |
| GRAFT (pre-ablation) | 37.5 ± 0.8 | 0.1 ± 0.1 |
| GRAFT (deployment configuration) | 31.9 ± 1.7 | 0.0 ± 0.0 |

## Avenues for future research

These are things we believe are worth exploring which could straightforwardly lead to better success with the method. This may have some overlap with some of the above items.

- A different adapter, particularly one that modifies the weights of the attention mechanism, but which does not suffer from the weaknesses of LoRA (see above).
- Scaling; generalization behavior of more interesting model behaviors that only arise at larger scale.
- Modulating the effect of the forget adapter in a more principled fashion, to sidestep problems with total saturation on hacking samples.
- Likewise, studying cases where retain samples can be misclassified as forget samples.
- More generally, adapting the method for continuous, instead of binary, notions of hack judgment, whether this reflects severity of the hack behavior or confidence in the judgment. Note that the original gradient routing work (Cloud et al., 2024) used a continuous notion of routing.

We note that if methods involving gradient routing are improved in robustness, so that their outputs unambiguously include localized representations of behavior or knowledge, gradient-routed models might serve as excellent targets for mechanistic interpretability in themselves, might be used to advance the practice of mechanistic interpretability, or might be part of a solution for building inherently interpretable networks.

## References

- Azarbal, A., et al. (2025) Recontextualization Mitigates Specification Gaming without Modifying the Specification. arXiv:2512.19027.
- Cloud, A., Goldman-Wetzler, J., Wybitul, E., Miller, J. & Turner, A.M. (2024) Gradient Routing: Masking Gradients to Localize Computation in Neural Networks. arXiv:2410.04332.
- Gao, L., Schulman, J. & Hilton, J. (2022) Scaling Laws for Reward Model Overoptimization. arXiv:2210.10760.
- Roland, E., et al. (2026) Modular Pretraining Enables Access Control. ICML. arXiv:2607.08077.
- Shilov, I., et al. (2025) Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs. arXiv:2512.05648.
- Team OLMo, et al. (2024) 2 OLMo 2 Furious. arXiv:2501.00656.
