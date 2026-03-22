# Hypotheses

## Active Queue (prioritized)

### H0: Developmental local loss, lower layers only
**Hypothesis:** Local teaching signals can help early representation formation if they are strong early in training, aimed at lower layers, and decay toward zero later.
**Mechanism:** Better credit assignment under a short budget. Lower blocks learn semantics faster instead of waiting for weak end-to-end gradients through the full stack.
**Implementation sketch:** Add an auxiliary alignment or prediction loss at a small set of early/mid layers, gate it on `self.training`, weight it more at lower layers than upper, and decay it over training steps.
**Expected:** Possible win if it improves early learning without throughput collapse. Risk: repeats DFA failure mode if backward graphs are too expensive or if the target is too direct.
**Tier:** 1

### H1: Iterative refinement with block reuse
**Hypothesis:** Reusing 1-2 strong blocks for 2-4 inner refinement steps will beat a deeper one-pass stack at the same wall-clock budget.
**Mechanism:** Better parameter efficiency and more settling dynamics. The model can refine a state instead of spending parameters on shallow feature transforms that each run once.
**Implementation sketch:** Replace some unique layers with a shared block loop, reinject `x0` each inner step, and keep residual scaling so the loop starts near identity.
**Expected:** High upside, high risk. Could improve quality-per-parameter if compile and kernel efficiency remain intact.
**Tier:** 1

### H2: Fast recurrent state per layer or per sequence
**Hypothesis:** A tiny persistent fast state can let the model act larger without paying for more static weights.
**Mechanism:** Temporary working memory handles short-horizon reasoning and context compression that dense weights currently try to absorb.
**Implementation sketch:** Add a small state vector updated each token or chunk, read it into the block input, and detach or partially detach updates if needed for stability.
**Expected:** Unknown. Strong conceptual fit for the mission, but easy to overcomplicate.
**Tier:** 2

### H3: Hard competition in attention outputs
**Hypothesis:** The current spike-gated MLP already hints that selective activation helps. Extending hard competition to attention outputs may reduce redundancy further.
**Mechanism:** Query outputs compete so only the most useful attended features survive. This should encourage specialization without changing the SDPA kernel itself.
**Implementation sketch:** Apply top-k or grouped winner-take-all after attention projection, not inside SDPA, to preserve efficient attention kernels.
**Expected:** Small win or neutral. Risk: useful blended attention features get zeroed out.
**Tier:** 1

### H4: Prune-by-use with tiny regrowth budget
**Hypothesis:** Static pruning helped in earlier, smaller regimes because development mattered. A use-based schedule with limited regrowth may work better than pure magnitude rules.
**Mechanism:** Remove persistently inactive pathways and let a tiny number of new connections form where co-activation is repeatedly high.
**Implementation sketch:** Track cheap activity statistics during training, prune on low utility, and allow a very small periodic regrowth budget.
**Expected:** Medium-risk structural play. Likely only worthwhile if implemented with minimal overhead.
**Tier:** 2

### H5: Future-state prediction, not multi-token logits
**Hypothesis:** Mixed objectives may help if they shape internal simulators instead of adding extra token heads that drag throughput and weaken the main gradient.
**Mechanism:** Predicting future hidden state or sequence features may improve planning-like representations at lower cost than multi-token cross-entropy.
**Implementation sketch:** Predict a stop-gradient future latent from the current latent at one or two layers, with low weight and strong scheduling.
**Expected:** More promising than MTP, but still risky under the five-minute budget.
**Tier:** 2

### H6: Width revisit only after state/credit experiments
**Hypothesis:** Scaling width may still help, but raw width is no longer the most interesting question after the attention fix and current clean baseline.
**Mechanism:** More features per layer improve expressiveness, but only if steps do not fall too far.
**Implementation sketch:** Revisit `n_embd` only after the iterative/stateful experiments above are tested cleanly.
**Expected:** Probably positive if it fits, but lower novelty and lower information value than the architectural queue above.
**Tier:** 3

### H7: Real layerwise local/global attention
**Hypothesis:** Early layers do not need full 2048-token global attention on every step. If we make `WINDOW_PATTERN` actually control attention span, we may gain useful throughput and improve quality-per-joule without the catastrophic regression seen when attention is removed entirely.
**Mechanism:** Local early layers can focus on short-range syntax and phrase assembly, while later layers keep global context. This reallocates compute across depth instead of across tokens or objectives.
**Implementation sketch:** Implement true causal sliding-window attention inside `CausalSelfAttention` for `S` layers while leaving `L` layers global. Preserve SDPA where possible and compare against the current all-global `c799673` baseline.
**Expected:** High information value. Could be a win if the kernel path stays efficient; could also regress if masking destroys FlashAttention or if early global context is more important than expected.
**Tier:** 1

## Tested / Resolved

| Hypothesis | Result | Commit |
|---|---|---|
| Attention shape / SDPA layout bug | MASSIVE WIN | f129e04 |
| Predictive coding block | small WIN | 565aa30 |
| Depth increase after fix | fail due to step loss | 26eb2cd / fce825b |
| Direct feedback alignment aux loss | contaminated then clean fail | a102500 / b856489 / 63b07a7 |
| Hopfield-style softmax MLP | fail, too slow | b8c1f6e |
| Kuramoto synchrony attention | fail, kernel incompatible | b0c36bb |
| Multi-token prediction | fail, too expensive | 9f159d7 / 70ba6a4 / 75b0705 |
| Homeostatic plasticity | near miss in old regime, unproven now | b1f96dd / 23e0a92 |
| H1: Iterative block reuse (all variants) | fail — shared blocks can't specialize by depth, quality loss outweighs throughput gain | 4fff336 / ea9250b / a40e3d5 |
| H0: Developmental local loss (FHSA) | fail — 35% throughput penalty, any aux loss through intermediate hiddens too expensive | 2f1b7e8 |
| Spiking surrogate gate | fail — sigmoid spike slower than silu, 114% MFU | d184a93 |
| MTP-8 (multi-token predict) | fail — logits backward 1GB extra VRAM, 82% MFU | 27614a1 |
| Hebbian weight normalization | fail — fights Muon orthogonalization | bc77b46 |
| Deep layer averaging | small clean WIN — near-zero-cost multi-scale blend improves baseline family | c799673 |
| Real local attention via additive SDPA mask | invalid / contaminated — near-zero loss and bpb indicate broken causal path; also 50% throughput hit | 6f81055 |
| Hard competition in attention outputs | clean fail — only ~2% step loss, but +0.044 bpb regression; dense output blending matters | dd284f4 |
| Depth-state workspace across layers | clean fail — slightly slower and +0.019 bpb worse; readout mixing helps, recurrent reinjection hurts | 96d1759 |
| Predictive coding on deep-layer-mix baseline | clean fail, but closer than recent bold variants — some per-step fit benefit did not survive wall-clock budget | 43bc457 |
| Low-rank predictive coding | near-miss discard — better than full-width predictive coding, but still loses on quality-per-second | cfbc49f |
| Selective low-rank predictive coding | best recent near-miss — lower-half rank-8 version gets within +0.005 of best; family still alive | 8a9c362 |
| Ultra-selective predictive coding | strongest recent near-miss — first-2-layer rank-8 version improves again and appears cost-limited | b2015f3 |
