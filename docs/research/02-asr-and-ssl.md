# Research notes: neural ASR, self-supervised speech, and speaker tasks

These notes are source material for the course part on automatic speech recognition (ASR),
self-supervised learning (SSL) for speech, and speaker tasks. They're written for the agents
and authors who turn them into lessons and auto-checked labs. Every section follows the same
pattern where it makes sense: intuition, exact equations, a hand-worked example, references,
student gotchas, and a lab idea with a compute budget.

Research date: 2026-09-28. Facts about 2024-2026 models were checked against model cards, arXiv,
and the live Hugging Face Open ASR Leaderboard configuration (changelog entry dated
2026-09-25). Items that couldn't be verified are listed in
[Unverified or uncertain facts](#unverified-or-uncertain-facts).

Compute labels used in lab ideas:

- **CPU-60s**: runs in the browser editor's CPU checker in under 60 seconds with numpy or
  torch-cpu. Any audio or logits come from a small fixture shipped with the lab.
- **GPU-L4** or **GPU-A100**: runs on a RunPod serverless worker. The label gives rough wall time
  and VRAM.

## Conventions used throughout

- $x_{1:T}$ is the acoustic feature sequence after the encoder (frame rate after subsampling).
  $y_{1:U}$ (or $\mathbf{l}$ of length $L$) is the target token sequence.
- $V$ is the vocabulary size including the blank symbol $\varnothing$ (written `_` in examples).
  Unless a section says otherwise, blank has index 0, which matches the default of
  `torch.nn.functional.ctc_loss` and many CTC toolkits. NeMo and torchaudio's `rnnt_loss`
  commonly put blank last (`blank=-1` or `blank=V-1`), so every lab states its convention.
- $y^t_k$ is the softmax output for symbol $k$ at frame $t$. All recursions are implemented in
  log space with log-sum-exp:
  $\operatorname{lse}(a,b)=\max(a,b)+\log\left(1+e^{-|a-b|}\right)$, with $-\infty$ handled
  explicitly.

---

## 1. Framing: from frame classification to end-to-end ASR

**Intuition.** Audio arrives at 100 frames per second (10 ms hop), but words arrive at roughly
2-3 per second. A frame classifier needs a label for every frame, and nobody hand-labels
frames. The core problem of ASR is therefore *alignment*: which frames belong to which output
symbol. Every model family in these notes is a different answer to that question.

**Frame-level classification.** If you had per-frame labels $s_t$ (for example, phone states),
you could train a classifier with cross-entropy
$\mathcal{L}=-\sum_t \log p(s_t\mid x)$. This is what TIMIT phone classification and keyword
spotting on fixed 1 s clips do.

**Hybrid DNN-HMM (brief).** Classic systems obtain frame labels from a GMM-HMM forced
alignment, train a DNN to output state posteriors $P(s\mid x_t)$, and convert them to scaled
likelihoods for the HMM:

$$
p(x_t\mid s)\propto \frac{P(s\mid x_t)}{P(s)}.
$$

Decoding searches $W^*=\arg\max_W p(X\mid W)\,P(W)^{\lambda}\,|W|^{\beta}$ over a WFST that
composes the HMM topology (H), context dependency (C), the pronunciation lexicon (L), and the
n-gram grammar (G) into HCLG. Later systems replaced cross-entropy with sequence-discriminative
training such as lattice-free MMI (Povey et al., 2016). Hybrid systems need a lexicon, a
state-tying tree, and a bootstrap alignment, which is why end-to-end models took over.

**End-to-end (E2E) ASR** maps $x$ to characters or subwords with one network trained on
(audio, transcript) pairs. Three families handle the alignment differently:

| Family | Alignment handling | Conditional independence | Streaming |
|---|---|---|---|
| CTC | Marginalize over monotonic frame-to-label paths with a blank | Outputs independent given $x$ | Natural |
| Attention encoder-decoder (AED) | Learned soft attention; no explicit alignment | Full autoregressive LM over outputs | Hard |
| RNN-T and TDT | Marginalize over a $T\times U$ lattice; predictor conditions on history | Autoregressive over labels | Natural |

Modern systems add a fourth pattern: a speech encoder plus adapter feeding a decoder-only
LLM (called SALM or speech-LLM). This is an AED whose decoder is a pretrained LLM.

**References.** Bourlard and Morgan, *Connectionist Speech Recognition* (1994); Hinton et al.,
"Deep neural networks for acoustic modeling in speech recognition" (2012,
https://ieeexplore.ieee.org/document/6296526); Povey et al., "Purely sequence-trained neural
networks for ASR based on lattice-free MMI" (Interspeech 2016).

**Gotcha.** Students often assume frame labels exist for LibriSpeech. They don't; only
utterance-level transcripts exist. Frame labels for hybrid training come from a forced
aligner, which section 13 covers.

---

## 2. Connectionist temporal classification (CTC)

**Intuition.** Let the network emit one symbol per frame from the vocabulary plus a blank.
Many frame-level paths collapse to the same transcript. CTC trains the network to maximize the
total probability of all paths that collapse to the correct transcript, computed efficiently
with dynamic programming (Graves et al., 2006).

### 2.1 Definitions

- Path $\pi\in\{\varnothing,1,\dots,V-1\}^T$ with probability
  $p(\pi\mid x)=\prod_{t=1}^{T} y^t_{\pi_t}$ (outputs are conditionally independent given $x$).
- Collapse function $\mathcal{B}$: first merge consecutive repeated symbols, then delete
  blanks. For example, $\mathcal{B}(\texttt{aa\_ab\_\_})=\texttt{aab}$ and
  $\mathcal{B}(\texttt{a\_a})=\texttt{aa}$ while $\mathcal{B}(\texttt{aa})=\texttt{a}$.
- Label probability:
  $$p(\mathbf{l}\mid x)=\sum_{\pi\in\mathcal{B}^{-1}(\mathbf{l})} \prod_{t=1}^{T} y^t_{\pi_t}.$$
- Loss: $\mathcal{L}_{\text{CTC}}=-\log p(\mathbf{l}\mid x)$.

### 2.2 Forward-backward over the extended label sequence

Insert blanks between labels and at both ends: for $\mathbf{l}=(l_1,\dots,l_L)$ the extended
sequence is $\mathbf{l}'=(\varnothing,l_1,\varnothing,l_2,\dots,l_L,\varnothing)$ with
$|\mathbf{l}'|=S=2L+1$. Index $s=1,\dots,S$ (1-based in equations, 0-based in code).

**Forward variable** $\alpha_t(s)$: total probability of all path prefixes $\pi_{1:t}$ that
end in state $s$ of $\mathbf{l}'$ and are consistent with $\mathbf{l}'_{1:s}$, including the
emission at frame $t$.

Initialization:

$$
\alpha_1(1)=y^1_{\varnothing},\qquad \alpha_1(2)=y^1_{l_1},\qquad \alpha_1(s)=0\ \ \forall s>2.
$$

Recursion for $t=2,\dots,T$:

$$
\alpha_t(s)=
\begin{cases}
\big(\alpha_{t-1}(s)+\alpha_{t-1}(s-1)\big)\,y^t_{l'_s} & \text{if } l'_s=\varnothing \text{ or } l'_s=l'_{s-2},\\[4pt]
\big(\alpha_{t-1}(s)+\alpha_{t-1}(s-1)+\alpha_{t-1}(s-2)\big)\,y^t_{l'_s} & \text{otherwise.}
\end{cases}
$$

The **skip rule** (the $s-2$ term) lets a path jump over a blank between two *different*
labels. It's forbidden when $l'_s$ is blank (you can't skip from a label to the next blank's
successor blank) and when $l'_s=l'_{s-2}$ (a repeated label such as "ll" in "hello" needs an
intervening blank; otherwise the two copies merge under $\mathcal{B}$).

Termination (a valid path ends on the last label or the final blank):

$$
p(\mathbf{l}\mid x)=\alpha_T(S)+\alpha_T(S-1).
$$

**Backward variable** (convention used here: excludes the emission at $t$):
$\beta_t(s)$ is the probability of emitting the remaining frames $t+1..T$ given state $s$ at
frame $t$.

$$
\beta_T(S)=\beta_T(S-1)=1,\qquad \beta_T(s)=0 \text{ otherwise},
$$

$$
\beta_t(s)=\sum_{s'\in\{s,\,s+1,\,s+2\}^{*}} \beta_{t+1}(s')\,y^{t+1}_{l'_{s'}},
$$

where the $s+2$ term is included only if $l'_{s+2}\neq\varnothing$ and $l'_{s+2}\neq l'_s$.
With this convention, $p(\mathbf{l}\mid x)=\sum_s \alpha_t(s)\beta_t(s)$ for *every* $t$, which
is an excellent unit test. Graves (2006) uses the other convention, where $\beta$ includes the
emission at $t$, so his formulas divide by $y^t_{l'_s}$. Pick one and state it.

**Posterior occupancy and gradient.** Define
$\gamma_t(s)=\alpha_t(s)\beta_t(s)/p(\mathbf{l}\mid x)$. With $y^t=\operatorname{softmax}(u^t)$
over unnormalized logits $u^t$:

$$
\frac{\partial \mathcal{L}_{\text{CTC}}}{\partial u^t_k}= y^t_k-\sum_{s:\,l'_s=k}\gamma_t(s).
$$

The gradient is "model posterior minus alignment posterior," the same shape as cross-entropy
with soft targets. This is why fusing log-softmax into the loss is numerically stable.

**Feasibility constraint.** A path must spend at least one frame per label, plus one blank
frame between each pair of identical consecutive labels. So you need
$T\ge L+r$, where $r$ is the number of positions with $l_i=l_{i+1}$. You don't need
$T\ge 2L+1$; that's the length of $\mathbf{l}'$, not a requirement on $T$. If $T<L+r$, then
$p=0$ and the loss is $+\infty$.

### 2.3 Log-space implementation

Store $\hat\alpha_t(s)=\log\alpha_t(s)$ and initialize everything to $-\infty$:

$$
\hat\alpha_t(s)=\log y^t_{l'_s}+\operatorname{lse}\big(\hat\alpha_{t-1}(s),\ \hat\alpha_{t-1}(s-1),\ [\hat\alpha_{t-1}(s-2)]\big).
$$

A vectorized numpy version shifts the $\hat\alpha_{t-1}$ vector by 1 and 2, masks the skip
term with a precomputed boolean vector `allow_skip[s] = (ext[s] != blank) & (ext[s] != ext[s-2])`,
and uses `np.logaddexp`. `np.logaddexp(-inf, -inf)` returns `-inf` without warnings, but a
hand-written `m + log(sum(exp(x - m)))` returns `nan` when `m = -inf`. Guard that case.

### 2.4 Hand-worked example: T = 3, label "a"

Vocabulary $\{\varnothing=0,\ a=1\}$. Per-frame probabilities of $a$ are
$(0.6, 0.3, 0.8)$, so blank probabilities are $(0.4, 0.7, 0.2)$.
Extended sequence $\mathbf{l}'=(\varnothing, a, \varnothing)$, $S=3$.

The six valid paths are `a__`, `_a_`, `__a`, `aa_`, `_aa`, and `aaa`:

| Path | Probability |
|---|---|
| `a__` | 0.6 x 0.7 x 0.2 = 0.084 |
| `_a_` | 0.4 x 0.3 x 0.2 = 0.024 |
| `__a` | 0.4 x 0.7 x 0.8 = 0.224 |
| `aa_` | 0.6 x 0.3 x 0.2 = 0.036 |
| `_aa` | 0.4 x 0.3 x 0.8 = 0.096 |
| `aaa` | 0.6 x 0.3 x 0.8 = 0.144 |
| **Sum** | **0.608** |

The same number from the $\alpha$ table (rows are frames, columns are $s=\varnothing_1, a,
\varnothing_2$):

| $t$ | $\alpha_t(\varnothing_1)$ | $\alpha_t(a)$ | $\alpha_t(\varnothing_2)$ |
|---|---|---|---|
| 1 | 0.4 | 0.6 | 0 |
| 2 | 0.4 x 0.7 = 0.28 | (0.6 + 0.4) x 0.3 = 0.30 | (0 + 0.6) x 0.7 = 0.42 |
| 3 | 0.28 x 0.2 = 0.056 | (0.30 + 0.28) x 0.8 = 0.464 | (0.42 + 0.30) x 0.2 = 0.144 |

$p(a\mid x)=\alpha_3(a)+\alpha_3(\varnothing_2)=0.464+0.144=0.608$, loss
$=-\ln 0.608=0.49758$. This matches `F.ctc_loss(..., reduction='sum')` exactly (verified with
torch 2.11). Gradient w.r.t. logits (columns blank, a) is
`[[-0.1658, 0.1658], [0.1934, -0.1934], [-0.0368, 0.0368]]`, which also matches autograd.

Two extra checks worth putting in a lab:

- With uniform probabilities 0.5, $p(a)=6/8=0.75$.
- For the label "aa" with $T=3$, the only valid path is `a_a`, so
  $p=0.6\times0.7\times0.8=0.336$. With $T=2$, $p=0$ and the loss is infinite.

### 2.5 Peaky behaviour

Trained CTC models emit each label as a spike of one or two frames and predict blank almost
everywhere else. Zeyer et al. (2021) show that peaky alignments are a local optimum that the
CTC loss prefers under uniform initialization, and that this happens regardless of model
capacity. Consequences for students:

- CTC timestamps are spike positions, not word boundaries; word end times are biased early or
  late. Use forced alignment with a trained model for boundaries.
- Greedy decoding works well because frames are near one-hot.
- Blank dominance lets you skip blank frames for speed (for example, frame skipping in
  "blank-skip" decoders).

### 2.6 Greedy (best-path) decoding

Take $\hat\pi_t=\arg\max_k y^t_k$ per frame, then apply $\mathcal{B}$. Best path is not the
most probable labeling. Counterexample with $T=2$: both frames have
$y_\varnothing=0.6$ and $y_a=0.4$. Greedy returns the empty string with probability
$0.36$, but $p(a)=0.4\cdot0.6+0.6\cdot0.4+0.4\cdot0.4=0.64$. Prefix beam search (next section)
finds "a".

### 2.7 CTC prefix beam search

**Intuition.** Search over *collapsed prefixes* rather than paths, and keep two probabilities
per prefix: ending in blank ($p_b$) and ending in a non-blank ($p_{nb}$). The split is needed
because "a" followed by "a" means "aa" only if a blank came between them.

**Algorithm** (Hannun et al., 2014; also the Distill article by Hannun, 2017). Beam width $K$.

```text
beam = { "": (p_b=1, p_nb=0) }
for t in 1..T:
    next = defaultdict(lambda: (0, 0))
    for prefix, (p_b, p_nb) in beam:
        # 1) emit blank: prefix unchanged, now ends in blank
        next[prefix].p_b += (p_b + p_nb) * y[t][blank]
        for c in non-blank symbols:
            ext = prefix + c
            if prefix non-empty and c == prefix[-1]:
                # 2a) repeat after a blank starts a new copy of c
                next[ext].p_nb += p_b * y[t][c]
                # 2b) repeat without blank collapses into the same prefix
                next[prefix].p_nb += p_nb * y[t][c]
            else:
                # 3) new symbol
                next[ext].p_nb += (p_b + p_nb) * y[t][c]
    beam = top K entries of next by score(prefix) = p_b + p_nb
return argmax over beam of p_b + p_nb
```

Implement it in log space (`logaddexp` instead of `+=`). With $K=\infty$ the result equals the
exact marginal $p(\mathbf{l}\mid x)$ for every labeling; this was checked against brute-force
path enumeration for random $T=5$, $V=3$ inputs. Hannun's original pseudocode adds one more
case for extensions that were pruned from the previous beam; most practical implementations
omit it.

**Pruning tricks.** Only extend symbols with $y^t_c$ above a threshold or in the top-$k$ per
frame (pyctcdecode uses `token_min_logp`), and skip frames where blank probability is above
0.999.

**Shallow fusion with an LM.** When a prefix is extended by $c$, multiply the new contribution
by $P_{\text{LM}}(c\mid \text{prefix})^{\alpha}$ (character LM) or, for a word LM, apply
$P_{\text{LM}}(w\mid \text{history})^{\alpha}$ when a word boundary (space) is emitted. Rank
prefixes by

$$
\operatorname{score}(\ell)=\log\big(p_b(\ell)+p_{nb}(\ell)\big)+\alpha\log P_{\text{LM}}(\ell)+\beta\,|\ell|_{\text{words}}.
$$

The insertion bonus $\beta$ counters the LM's bias toward short outputs. wav2vec 2.0 decoded
LibriSpeech with a 4-gram LM using LM weight 2 and word insertion penalty -1 as fixed defaults
for Libri-light, beam 50 (Baevski et al., 2020). pyctcdecode defaults are
`alpha=0.5`, `beta=1.5`, `beam_width=100`, and `hotword_weight=10.0`.

### 2.8 Gotchas

- `torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0,
  reduction='mean', zero_infinity=False)` expects `log_probs` shaped `(T, N, C)` and already
  passed through `log_softmax`. Passing raw logits or `(N, T, C)` silently trains garbage.
- `reduction='mean'` divides each utterance's loss by its *target length* and then averages
  over the batch. Use `reduction='sum'` when comparing to a hand implementation.
- Input lengths are lengths *after* encoder subsampling. With 4x subsampling, a 1.0 s clip has
  about 25 frames, which can be too short for a long character transcript. Set
  `zero_infinity=True` or filter such utterances.
- The cuDNN kernel is used only under specific conditions (blank 0, int32 concatenated
  targets on CPU, all input lengths equal to $T$, and label length up to 256). Results
  otherwise come from the native kernel; both are correct, but determinism and speed differ.
- Characters versus subwords: with 8x subsampling (80 ms frames), character CTC can violate
  $T\ge L+r$ for fast speech, which is one reason FastConformer models use BPE vocabularies.
- Blank index conventions: HF `Wav2Vec2ForCTC` uses the pad token (`<pad>`, id 0 in the
  standard LibriSpeech vocab) as blank and `|` as the word delimiter.

### 2.9 Lab ideas

1. **CTC forward in log space** (CPU-60s). Implement `ctc_nll(log_probs, target, blank)` in
   numpy; check against `F.ctc_loss(..., reduction='sum')` on random inputs, including
   repeated labels and the infeasible case. Hidden tests: the T=3 "a" example (0.49758), "aa"
   with T=2 (inf), and a random batch.
2. **Backward and gradient** (CPU-60s). Implement $\beta$, check
   $\sum_s\alpha_t(s)\beta_t(s)$ is constant over $t$, and compare the analytic gradient with
   autograd.
3. **Decoding on real logits** (CPU-60s). Ship an `.npz` with `facebook/wav2vec2-base-960h`
   log-probs for 20 LibriSpeech utterances. Students implement greedy and prefix beam search;
   the checker requires identical greedy output and beam WER no worse than greedy.
4. **Tiny CTC model on synthetic tone sequences** (CPU-60s). Generate sequences of 3-6 tones
   from a 5-tone alphabet with random durations, compute log-mel features, train a 1-layer
   BiGRU with CTC for about 300 steps; the check requires greedy label error rate under 10%.

**References.** Graves et al., ICML 2006 (https://www.cs.toronto.edu/~graves/icml_2006.pdf);
Graves, *Supervised Sequence Labelling with RNNs* (2012); Hannun et al., 2014
(https://arxiv.org/abs/1408.2873); Hannun, "Sequence modeling with CTC," Distill 2017
(https://distill.pub/2017/ctc/); Zeyer et al., 2021 (https://arxiv.org/abs/2105.14849).

---

## 3. Attention encoder-decoder models

### 3.1 Listen, attend and spell (LAS)

**Intuition.** An encoder ("listener") turns audio into a shorter sequence $h$; a decoder
("speller") generates one character at a time and uses attention to decide which encoder frames
matter for the next character. No explicit alignment exists; the model learns it (Chan et al.,
2016).

- Pyramidal BLSTM listener: each layer concatenates adjacent frames,
  $h^j_i=\operatorname{BLSTM}\big(h^j_{i-1},[h^{j-1}_{2i};h^{j-1}_{2i+1}]\big)$. Three pyramid
  layers reduce the frame rate by 8x.
- Decoder state $s_i=\operatorname{RNN}(s_{i-1},y_{i-1},c_{i-1})$, context
  $c_i=\sum_u \alpha_{i,u}h_u$, and
  $P(y_i\mid x,y_{<i})=\operatorname{softmax}(\operatorname{MLP}([s_i;c_i]))$.
- Content attention: $e_{i,u}=\langle\phi(s_i),\psi(h_u)\rangle$,
  $\alpha_{i,u}=\operatorname{softmax}_u(e_{i,u})$.
- Loss: $\mathcal{L}_{\text{att}}=-\sum_{i=1}^{U+1}\log P(y_i\mid x,y^*_{<i})$ with teacher
  forcing, where $y_{U+1}=\langle\text{eos}\rangle$.

### 3.2 Location-aware attention (Chorowski et al., 2015)

Content-only attention can jump to a similar-sounding frame elsewhere in the utterance.
Location-aware attention convolves the previous attention weights with learned filters $F$ so
that the model knows where it attended last:

$$
f_i=F*\alpha_{i-1},\qquad
e_{i,u}=w^{\top}\tanh\!\big(Ws_{i-1}+Vh_u+Uf_{i,u}+b\big),\qquad
\alpha_{i,u}=\frac{\exp(e_{i,u})}{\sum_{u'}\exp(e_{i,u'})}.
$$

The paper also proposes sharpening ($\beta>1$ inverse temperature) and windowing to keep
attention local on long utterances.

### 3.3 Joint CTC/attention (Kim, Hori, and Watanabe; ESPnet)

**Intuition.** Attention is flexible but can loop, skip, or stop early; CTC enforces monotonic
alignment. Train with both losses on a shared encoder and use CTC scores during beam search to
reject non-monotonic hypotheses.

$$
\mathcal{L}=\lambda\,\mathcal{L}_{\text{CTC}}+(1-\lambda)\,\mathcal{L}_{\text{att}},\qquad \lambda\in[0.1,0.5]\ (\text{ESPnet recipes often use }0.3).
$$

One-pass decoding score for a hypothesis $h$:

$$
\operatorname{score}(h)=\lambda_d\log p_{\text{CTC}}(h\mid x)+(1-\lambda_d)\log p_{\text{att}}(h\mid x)+\gamma\log p_{\text{LM}}(h),
$$

where $\log p_{\text{CTC}}(h\mid x)$ is the *prefix* probability computed with the CTC prefix
score algorithm (Watanabe et al., 2017). ESPnet made this the default recipe (Watanabe et al.,
2018).

### 3.4 Training tricks

- **Label smoothing**: replace the one-hot target with
  $q'(k)=(1-\varepsilon)\,\mathbb{1}[k=y]+\varepsilon/K$, $\varepsilon\approx 0.1$ (Szegedy et
  al., 2016; Chorowski and Jaitly, 2017 showed it reduces overconfidence and improves beam
  search with LMs).
- **Exposure bias**: during training the decoder always sees the true history $y^*_{<i}$;
  at test time it sees its own, possibly wrong, predictions and errors compound.
- **Scheduled sampling** (Bengio et al., 2015): at each step feed the ground-truth token with
  probability $\epsilon_i$ and the model's own prediction otherwise, decaying $\epsilon_i$ from
  1 toward a floor on a linear, exponential, or inverse-sigmoid schedule.
- SpecAugment (section 5.5) also helps AED models dramatically; LAS was the model in the
  SpecAugment paper.

### 3.5 Beam search with length normalization

Beam search keeps the $B$ best partial hypotheses by cumulative log-probability. Because every
extra token adds a negative term, raw scores favour short outputs. Two common fixes:

- Divide by length: $\operatorname{score}=\frac{1}{|y|}\log P(y\mid x)$ (Whisper's default
  ranker when `length_penalty=None`).
- GNMT length penalty (Wu et al., 2016):
  $\operatorname{lp}(y)=\frac{(5+|y|)^{\alpha}}{(5+1)^{\alpha}}$ and score
  $=\log P(y\mid x)/\operatorname{lp}(y)$, $\alpha\in[0.6,1.0]$. Whisper uses
  `((5 + length) / 6) ** length_penalty` when a penalty is given.

Other AED-specific failure modes are end-of-sentence emitted too early (fix: EOS threshold or
coverage penalty) and looping on long inputs (fix: CTC joint decoding or chunking).

**Lab ideas.**

1. **Location-aware attention module** (CPU-60s): implement $e_{i,u}$ with a Conv1d over
   $\alpha_{i-1}$; the checker compares outputs to a reference implementation on fixed
   weights.
2. **Beam search with length normalization** (CPU-60s): given a toy next-token distribution
   function, implement beam search and GNMT normalization; the checker verifies that the
   ranking flips for a hand-built case where the raw score prefers the shorter hypothesis.
3. **Tiny AED versus CTC** (GPU-L4, about 10 min, under 4 GB): train a 4-layer Transformer AED
   and a CTC model on the same synthetic digit-speech data and compare error rates and
   failure modes (looping, early stop).

**References.** Chan et al., 2016 (https://arxiv.org/abs/1508.01211); Chorowski et al., 2015
(https://arxiv.org/abs/1506.07503); Kim et al., 2017 (https://arxiv.org/abs/1609.06773);
Watanabe et al., 2017, "Hybrid CTC/attention architecture for end-to-end speech recognition,"
IEEE JSTSP 11(8) (https://doi.org/10.1109/JSTSP.2017.2763455); Watanabe et al., 2018, ESPnet
(https://arxiv.org/abs/1804.00015); Chorowski and Jaitly, 2017
(https://arxiv.org/abs/1612.02695); Bengio et al., 2015 (https://arxiv.org/abs/1506.03099);
Szegedy et al., 2016 (https://arxiv.org/abs/1512.00567); Wu et al., 2016
(https://arxiv.org/abs/1609.08144).

---

## 4. RNN transducer (RNN-T) and token-and-duration transducer (TDT)

### 4.1 Model

**Intuition.** RNN-T (Graves, 2012) removes CTC's conditional independence by adding a
*prediction network* (a label-conditioned LM) and a *joint network* that combines encoder frame
$t$ with prediction state $u$. At each lattice node the model either emits blank (move to the
next frame) or emits a label (stay on the same frame, advance the label index).

- Encoder: $f_t=\operatorname{Enc}(x)_t$, $t=1..T$.
- Predictor: $g_u=\operatorname{Pred}(y_{\le u})$, $u=0..U$, with $g_0$ from a start token.
- Joint:
  $z_{t,u}=W_{\text{out}}\tanh\!\big(W_{\text{enc}}f_t+W_{\text{pred}}g_u+b\big)\in\mathbb{R}^V$,
  $P(k\mid t,u)=\operatorname{softmax}(z_{t,u})_k$.
- Shorthand: $\varnothing(t,u)=P(\varnothing\mid t,u)$ and $y(t,u)=P(y_{u+1}\mid t,u)$.

### 4.2 The T x (U+1) lattice and forward recursion

Node $(t,u)$ means "the first $u$ labels are emitted and the model is at frame $t$."
With 1-based $t$ and 0-based $u$:

$$
\alpha(1,0)=1,\qquad
\alpha(t,u)=\alpha(t-1,u)\,\varnothing(t-1,u)+\alpha(t,u-1)\,y(t,u-1),
$$

$$
P(\mathbf{y}\mid x)=\alpha(T,U)\,\varnothing(T,U),\qquad \mathcal{L}_{\text{RNNT}}=-\log P(\mathbf{y}\mid x).
$$

A path must end with a final blank at frame $T$. Unlike CTC, a frame can emit several labels
(vertical moves), so no $T\ge L$ constraint exists.

**Backward recursion**: $\beta(T,U)=\varnothing(T,U)$,
$\beta(t,u)=\beta(t+1,u)\varnothing(t,u)+\beta(t,u+1)y(t,u)$. Gradients follow from
$\alpha(t,u)\beta(\cdot)$ products as in CTC (Graves, 2012, eq. 16-18).

**Hand-worked example.** $T=2$, target "a" ($U=1$), vocabulary {blank, a}.
Joint probabilities: at $(1,0)$ blank 0.4 and a 0.6; at $(1,1)$ blank 0.7; at $(2,0)$ blank
0.5 and a 0.5; at $(2,1)$ blank 0.9.

- $\alpha(1,0)=1$, $\alpha(1,1)=1\cdot0.6=0.6$, $\alpha(2,0)=1\cdot0.4=0.4$.
- $\alpha(2,1)=\alpha(1,1)\cdot0.7+\alpha(2,0)\cdot0.5=0.42+0.20=0.62$.
- $P(a\mid x)=0.62\times0.9=0.558$.

The two paths are "a at t=1, blank, blank" ($0.6\cdot0.7\cdot0.9=0.378$) and "blank, a at
t=2, blank" ($0.4\cdot0.5\cdot0.9=0.18$).

### 4.3 Greedy decoding

```text
u_state = pred(start); t = 1; out = []
while t <= T:
    k = argmax joint(f_t, g)
    if k == blank or symbols_this_frame == max_symbols:
        t += 1; symbols_this_frame = 0
    else:
        out.append(k); g = pred(k, g); symbols_this_frame += 1
```

Cap symbols per frame (NeMo `max_symbols`, often 10) to avoid infinite loops on
under-trained models. Beam search variants include Graves's prefix beam, time-synchronous
decoding, alignment-length synchronous decoding (Saon et al., 2020), and "modified beam search"
(icefall, one symbol per frame).

### 4.4 Memory cost and tricks

The joint output is a 4-D tensor $B\times T\times(U+1)\times V$. Example: $B=16$, $T=250$
(10 s at 40 ms), $U=80$, $V=1024$ gives $3.3\times10^8$ floats, which is 1.33 GB in fp32 for
logits alone and the same again for gradients.

- **Function merging / fused log-softmax** (Li et al., 2019): compute log-softmax inside the
  loss kernel and write the gradient in place, avoiding a separate softmax tensor. This is
  `fused_log_softmax=True` in `torchaudio.functional.rnnt_loss`.
- **Sequence packing**: avoid padding by concatenating the $T_b(U_b+1)$ blocks of each
  utterance (Li et al., 2019).
- **Pruned RNN-T** (Kuang et al., 2022, k2/icefall): first compute a cheap loss with a
  *simple joiner* that is linear in encoder and predictor outputs
  ($z_{t,u}=\operatorname{Lin}(f_t)+\operatorname{Lin}(g_u)$). Its lattice occupancy gradients
  show which $u$ values matter for each $t$. Then evaluate the full non-linear joiner only in a
  band of $S$ consecutive $u$ values per frame (`prune_range`, commonly 5), reducing memory to
  $B\times T\times S\times V$.
- **FastEmit** (Yu et al., 2021) adds a regularizer that rewards emitting labels earlier to cut
  streaming emission latency.

### 4.5 Stateless predictors

Ghodsi et al. (2020) showed that a prediction network conditioned only on the last emitted
symbol (no recurrence) performs virtually as well as an LSTM predictor with wordpieces. icefall
and many Zipformer recipes use an embedding of the last `context_size=2` tokens followed by a
1-D convolution. This works about as well as an LSTM predictor, is cheaper, and makes beam
search easier because states are just token tuples.

### 4.6 Token-and-duration transducer (TDT)

**Intuition.** Standard RNN-T spends one joint evaluation per frame just to emit blanks. TDT
(Xu et al., 2023, NVIDIA) predicts, at each step, both a token and how many frames to *skip*.
Decoding jumps ahead, which gives large speedups because most frames are blank.

- Joint network has two heads that are normalized independently:
  $P(v,d\mid t,u)=P_T(v\mid t,u)\,P_D(d\mid t,u)$, with $d\in\mathcal{D}$, typically
  $\mathcal{D}=\{0,1,2,3,4\}$.
- Forward recursion:

$$
\alpha(t,u)=\sum_{d\in\mathcal{D}\setminus\{0\}}\alpha(t-d,u)\,P(\varnothing,d\mid t-d,u)+\sum_{d\in\mathcal{D}}\alpha(t-d,u-1)\,P(y_u,d\mid t-d,u-1).
$$

  Blank must advance at least one frame; a label may have duration 0 (stay on the frame).
- Training tricks from the paper: logit under-normalization,
  $\log P'_T(v\mid t,u)=\log\operatorname{softmax}(h_v)-\sigma$ with $\sigma\approx0.05$, and
  mixing in the standard transducer loss with probability $\omega=0.1$ to help batched
  inference.
- Reported inference speedups: up to 2.19x (English ASR), 2.82x (German ASR), and 2.27x
  (speech translation) over conventional transducers.
- Production use: `nvidia/parakeet-tdt-0.6b-v2` and `-v3` (section 8).

**Gotchas.** Blank index: torchaudio's `rnnt_loss` defaults to `blank=-1` (last). Logit
tensors are `(B, T, U+1, V)`, where `U+1` includes the start position. Lengths are int32.
`torchaudio.functional.rnnt_loss` was scheduled for removal in torchaudio 2.9, but a
2026-01-22 update to the torchaudio roadmap says RNN-T loss and `forced_align` were preserved
after community feedback. Verify in the course's pinned version.

**Lab ideas.**

1. **RNN-T forward in log space** (CPU-60s): implement $\alpha$ on a `(T, U+1, V)` log-prob
   tensor; check against brute-force path enumeration for $T\le4$, $U\le2$, and against
   `torchaudio.functional.rnnt_loss` if available. Include the 0.558 example.
2. **Greedy transducer decoding with a stateless predictor** (CPU-60s): given fixed weights,
   implement greedy decoding with `max_symbols`; check the token output.
3. **TDT lattice** (CPU-60s): extend the RNN-T forward to durations and check that with
   $\mathcal{D}=\{1\}$ for blank and $\{0\}$ for labels it reduces to RNN-T.
4. **Memory estimation** (CPU-60s): write a function returning joint-tensor bytes for full and
   pruned RNN-T; auto-check numerically.

**References.** Graves, 2012 (https://arxiv.org/abs/1211.3711); Li et al., 2019, "Improving
RNN transducer modeling for end-to-end speech recognition" (https://arxiv.org/abs/1909.12415);
Kuang et al., 2022, pruned RNN-T (https://arxiv.org/abs/2206.13236); Ghodsi et al., 2020,
"RNN-transducer with stateless prediction network," ICASSP
(https://research.google/pubs/rnn-transducer-with-stateless-prediction-network/); Xu et al., 2023, TDT
(https://arxiv.org/abs/2304.06795); Yu et al., 2021, FastEmit
(https://arxiv.org/abs/2010.11148); Saon et al., 2020, alignment-length synchronous decoding, ICASSP
(https://research.ibm.com/publications/alignment-length-synchronous-decoding-for-rnn-transducer).

---

## 5. Encoders, front ends, augmentation, and output units

### 5.1 Front ends and subsampling

Input features are typically 80-dimensional log-mel filterbanks with a 25 ms window and 10 ms
hop (100 frames per second). A convolutional front end reduces the frame rate before the
expensive encoder:

- **Conv2d subsampling (4x)**: two Conv2d layers with kernel 3 and stride 2, ReLU, then a
  linear projection. 100 Hz becomes 25 Hz (40 ms frames). Output length is
  $\lfloor(\lfloor(T-1)/2\rfloor-1)/2\rfloor$ for unpadded kernels; get this formula right
  because CTC input lengths depend on it.
- **FastConformer (8x)**: three depthwise-separable Conv2d stride-2 stages with 256 channels,
  giving 12.5 Hz (80 ms). Reported as 2.8x faster than Conformer (Rekesh et al., 2023).
- **Waveform front ends**: wav2vec 2.0 uses a 7-layer 1-D CNN on raw 16 kHz audio (section 6).

### 5.2 BLSTM and Transformer encoders

BLSTM stacks (DeepSpeech 2, LAS) model context recurrently; they're slow to train on long
sequences. Transformer encoders use self-attention with sinusoidal or relative positions,
typically 12-18 layers of $d=256$-$512$ after 4x subsampling. They need warmup schedules and
SpecAugment to beat BLSTMs on LibriSpeech.

### 5.3 Conformer block (Gulati et al., 2020)

**Intuition.** Self-attention captures global context, and convolution captures local
patterns such as formant transitions. Conformer sandwiches both between two half-step
feed-forward modules (Macaron-Net style).

For block input $x$:

$$
\tilde{x}=x+\tfrac12\operatorname{FFN}(x),\quad
x'=\tilde{x}+\operatorname{MHSA}(\tilde{x}),\quad
x''=x'+\operatorname{Conv}(x'),\quad
y=\operatorname{LayerNorm}\!\big(x''+\tfrac12\operatorname{FFN}(x'')\big).
$$

Modules (all pre-norm residual):

- **FFN**: LayerNorm, Linear $d\to4d$, Swish, Dropout, Linear $4d\to d$, Dropout.
- **MHSA**: LayerNorm, multi-head self-attention with Transformer-XL relative sinusoidal
  positional encoding, Dropout.
- **Conv module**: LayerNorm, pointwise Conv ($d\to2d$), GLU (back to $d$), 1-D depthwise
  Conv (kernel 32 in the paper; 31 in many open implementations), BatchNorm, Swish, pointwise
  Conv, Dropout.

| Model | Params | Layers | Dim | Heads | Conv kernel | Decoder |
|---|---|---|---|---|---|---|
| Conformer S | 10.3M | 16 | 144 | 4 | 32 | 1-layer LSTM, 320 |
| Conformer M | 30.7M | 16 | 256 | 4 | 32 | 1-layer LSTM, 640 |
| Conformer L | 118.8M | 17 | 512 | 8 | 32 | 1-layer LSTM, 640 |

Conformer L reported 2.1/4.3 WER (test-clean/test-other) without an LM and 1.9/3.9 with an LM
on LibriSpeech, trained as a transducer with SpecAugment ($F=27$, ten time masks with
$p_S=0.05$).

**Gotchas.** BatchNorm in the conv module behaves differently in `eval()` mode and with
padding; mask padded frames before the depthwise conv. Relative positional encoding is easy
to get wrong; start labs with absolute positions and add relative positions as an extension.

### 5.4 Later encoder variants

- **Squeezeformer** (Kim et al., 2022): a Temporal U-Net that downsamples from 40 ms to 80 ms
  in the middle blocks and upsamples before the output, replaces the Macaron structure with
  MHA-FFN-Conv-FFN ordering, uses post-LayerNorm with scaling, and a depthwise-separable
  subsampling layer.
- **E-Branchformer** (Kim et al., 2022): parallel branches, global (MHSA) and local
  (convolutional gating MLP, cgMLP), merged by a depthwise conv plus linear layer, with
  Macaron FFNs around the block. It's the encoder of OWSM v3.1.
- **Zipformer** (Yao et al., ICLR 2024; k2/icefall): a U-Net-like stack whose middle stacks run
  at lower frame rates (roughly 50, 25, 12.5, 6.25, 12.5, and 25 Hz), BiasNorm instead of
  LayerNorm, SwooshR and SwooshL activations, reused attention weights within a block, and the
  ScaledAdam optimizer.
- **FastConformer** (Rekesh et al., 2023): 8x subsampling, conv kernel 9, and limited-context
  attention with a global token for long audio (up to about 11 hours). It's the encoder behind
  Parakeet, Canary, Canary-Qwen, Cohere Transcribe, and Nemotron streaming models.

### 5.5 SpecAugment

**Intuition.** Mask random frequency bands and time spans of the log-mel spectrogram so that
the model can't rely on any single cue (Park et al., 2019).

- **Time warping**: pick a point in $(W, \tau-W)$ and warp it left or right by
  $w\sim\mathcal{U}(0,W)$. Often omitted in practice because it's expensive and helps little.
- **Frequency masking**: $f\sim\mathcal{U}(0,F)$, $f_0\sim\mathcal{U}(0,\nu-f)$, zero channels
  $[f_0,f_0+f)$; repeat $m_F$ times.
- **Time masking**: $t\sim\mathcal{U}(0,T_{\max})$, $t_0\sim\mathcal{U}(0,\tau-t)$, with the
  upper bound $t\le p\,\tau$; repeat $m_T$ times.

| Policy | $W$ | $F$ | $m_F$ | $T$ | $p$ | $m_T$ |
|---|---|---|---|---|---|---|
| LibriSpeech basic (LB) | 80 | 27 | 1 | 100 | 1.0 | 1 |
| LibriSpeech double (LD) | 80 | 27 | 2 | 100 | 1.0 | 2 |
| Switchboard mild (SM) | 40 | 15 | 2 | 70 | 0.2 | 2 |
| Switchboard strong (SS) | 40 | 27 | 2 | 70 | 0.2 | 2 |

Gotchas: mask with the per-utterance mean or zero *after* normalization (zero in log-mel
before normalization means very loud, not silence); apply only in training; use the true
utterance length $\tau$, not the padded length.

### 5.6 Speed perturbation and other augmentation

Ko et al. (2015) resample audio by factors 0.9, 1.0, and 1.1, changing both tempo and pitch,
which triples the training data; it gave consistent WER gains in Kaldi recipes. Other common
augmentations are additive noise from MUSAN, reverberation with simulated room impulse
responses, and, for SSL, the utterance mixing used by WavLM.

### 5.7 Output units: characters versus subwords

- **Characters**: small vocabulary (about 30 for English), long sequences, no
  out-of-vocabulary words; good for CTC at 40 ms frames.
- **BPE** (Sennrich et al., 2016): greedily merges the most frequent adjacent pair.
- **Unigram LM** (Kudo, 2018): starts from a large vocabulary and prunes pieces that least
  reduce corpus likelihood; supports subword sampling as regularization.
- **SentencePiece** (Kudo and Richardson, 2018) implements both on raw text and treats space as
  the symbol `▁`, so detokenization is lossless.
- Typical sizes: 128-1024 for English CTC/RNN-T; 8,192 for Parakeet v3's 25 languages;
  Whisper's byte-level BPE has 50,257 text tokens plus special tokens.

**Lab ideas.**

1. **Conformer block from scratch** (CPU-60s): implement the block in PyTorch; the checker
   loads fixed weights and compares outputs to a reference within 1e-5 and checks parameter
   count.
2. **SpecAugment with a seeded RNG** (CPU-60s): implement frequency and time masks; the checker
   verifies masked regions and the $p\tau$ bound.
3. **Subsampling length formula** (CPU-60s): compute output lengths for Conv2d 4x and 8x
   front ends and verify against running the modules.
4. **Train a 4M-parameter Conformer-CTC on LibriSpeech train-clean-100** (GPU-A100, about
   40-60 min for a few epochs, under 20 GB): report greedy WER on dev-clean; expect roughly
   15-25% after short training without an LM. This is a capstone, not a weekly lab.

**References.** Gulati et al., 2020 (https://arxiv.org/abs/2005.08100); Kim et al., 2022,
Squeezeformer (https://arxiv.org/abs/2206.00888); Kim et al., 2022, E-Branchformer
(https://arxiv.org/abs/2210.00077); Peng et al., 2022, Branchformer
(https://arxiv.org/abs/2207.02971); Yao et al., 2023, Zipformer
(https://arxiv.org/abs/2310.11230); Rekesh et al., 2023, FastConformer
(https://arxiv.org/abs/2305.05084); Park et al., 2019, SpecAugment
(https://arxiv.org/abs/1904.08779); Ko et al., 2015, "Audio augmentation for speech
recognition," Interspeech
(https://www.isca-archive.org/interspeech_2015/ko15_interspeech.html); Sennrich et al., 2016
(https://arxiv.org/abs/1508.07909); Kudo, 2018 (https://arxiv.org/abs/1804.10959); Kudo and
Richardson, 2018 (https://arxiv.org/abs/1808.06226).

---

## 6. Self-supervised speech representations

**Intuition.** Transcribed speech is expensive; untranscribed speech is nearly free. SSL
pretrains an encoder on raw audio by making it predict hidden parts of the input from context,
then fine-tunes it on a small labeled set with CTC. The recipe that dominates is "mask, then
predict discrete or continuous targets for the masked frames."

### 6.1 Contrastive predictive coding (CPC)

An encoder $g_{\text{enc}}$ maps audio to latents $z_t$; an autoregressive model $g_{\text{ar}}$
summarizes $z_{\le t}$ into $c_t$. For each offset $k$, a bilinear score
$f_k(x_{t+k},c_t)=\exp(z_{t+k}^{\top}W_kc_t)$ is trained with the InfoNCE loss over a set $X$
of one positive and $N-1$ negatives:

$$
\mathcal{L}_N=-\mathbb{E}_X\left[\log\frac{f_k(x_{t+k},c_t)}{\sum_{x_j\in X}f_k(x_j,c_t)}\right].
$$

Minimizing $\mathcal{L}_N$ maximizes a lower bound $I(x_{t+k};c_t)\ge\log N-\mathcal{L}_N$
(van den Oord et al., 2018).

### 6.2 wav2vec 2.0

**Architecture** (Baevski et al., 2020):

- **Feature encoder**: seven 1-D conv blocks with 512 channels, strides (5,2,2,2,2,2,2),
  kernel widths (10,3,3,3,3,2,2), GELU, and group or layer norm. Total stride is 320 samples
  (20 ms at 16 kHz), receptive field 400 samples (25 ms), output about 49-50 Hz. The raw
  waveform is normalized to zero mean and unit variance.
- **Context network**: Transformer with a convolutional relative positional embedding (kernel
  128, 16 groups). BASE has 12 blocks, $d=768$, FFN 3,072, 8 heads (95M parameters). LARGE has
  24 blocks, $d=1024$, FFN 4,096, 16 heads (317M).
- **Quantizer**: product quantization with $G=2$ codebooks of $V=320$ entries (theoretical
  $320^2=102{,}400$ codewords); entries are $d/G=128$ (BASE) or 384 (LARGE) and concatenated,
  then linearly projected to $q_t$.

**Gumbel softmax.** Feature $z$ maps to logits $l\in\mathbb{R}^{G\times V}$, and

$$
p_{g,v}=\frac{\exp\big((l_{g,v}+n_v)/\tau\big)}{\sum_{k=1}^{V}\exp\big((l_{g,k}+n_k)/\tau\big)},\qquad n=-\log(-\log u),\ u\sim\mathcal{U}(0,1).
$$

The forward pass uses the hard argmax (straight-through estimator); the backward pass uses the
soft Gumbel gradient. $\tau$ anneals from 2 to 0.5 (BASE) or 0.1 (LARGE) by a factor of
0.999995 per update.

**Masking.** Sample a proportion $p=0.065$ of time steps as span starts and mask the next
$M=10$ steps (spans may overlap), replacing them with one learned vector. About 49% of frames
end up masked, mean span 14.7 frames (299 ms). Quantizer inputs aren't masked.

**Loss.** $\mathcal{L}=\mathcal{L}_m+\alpha\mathcal{L}_d$ with $\alpha=0.1$.

$$
\mathcal{L}_m=-\log\frac{\exp(\operatorname{sim}(c_t,q_t)/\kappa)}{\sum_{\tilde q\in Q_t}\exp(\operatorname{sim}(c_t,\tilde q)/\kappa)},\qquad
\operatorname{sim}(a,b)=\frac{a^{\top}b}{\lVert a\rVert\lVert b\rVert},\ \kappa=0.1,
$$

where $Q_t$ contains $q_t$ and $K=100$ distractors sampled uniformly from other *masked* steps
of the same utterance.

$$
\mathcal{L}_d=\frac{1}{GV}\sum_{g=1}^{G}-H(\bar p_g)=\frac{1}{GV}\sum_{g=1}^{G}\sum_{v=1}^{V}\bar p_{g,v}\log\bar p_{g,v},
$$

where $\bar p_g$ is the softmax (no Gumbel noise, no temperature) averaged over the batch. The
fairseq code equivalently maximizes codebook perplexity.

**Fine-tuning.** Add a randomly initialized linear layer to characters and train with CTC. The
feature encoder stays frozen; for the first 10k updates only the output layer trains. Masking
in feature space acts like SpecAugment. BASE (LibriSpeech 960 h pretraining) test-clean /
test-other WER **without an LM**: 10 min labeled 46.9/50.9, 1 h 24.5/29.7, 10 h 11.1/17.6,
100 h 6.1/13.3, 960 h 3.4/8.5 (paper, Appendix C). With a 4-gram LM, 10 min gives 9.1/15.6.

**Worked check.** For a 1.0 s input (16,000 samples), apply
$L\leftarrow\lfloor(L-k)/s\rfloor+1$ layer by layer:
16000, 3199, 1599, 799, 399, 199, 99, 49. So one second yields 49 frames. The receptive field
is $1+\sum_i(k_i-1)\prod_{j<i}s_j=1+9+10+20+40+80+80+160=400$ samples.

### 6.3 HuBERT

**Intuition.** Replace wav2vec 2.0's online quantizer with offline k-means cluster IDs as
targets, predict them BERT-style for masked frames, and improve the targets by re-clustering the
model's own features (Hsu et al., 2021).

- Iteration 1: k-means with 100 clusters on 39-D MFCCs (13 coefficients plus deltas and
  delta-deltas) of LibriSpeech 960 h; train BASE for 250k steps.
- Iteration 2: k-means with 500 clusters on layer 6 of the iteration-1 BASE model; train for
  400k steps.
- LARGE (317M) and X-LARGE (964M) on Libri-Light 60k h use 500 clusters from layer 9 of the
  iteration-2 BASE model (effectively a third iteration).
- Masking: $p=8\%$ of frames as starts, span $l=10$.
- Prediction:
  $p(c\mid\tilde X,t)=\dfrac{\exp(\operatorname{sim}(Ao_t,e_c)/\tau)}{\sum_{c'}\exp(\operatorname{sim}(Ao_t,e_{c'})/\tau)}$,
  cosine similarity, $\tau=0.1$.
- Loss $\mathcal{L}=\alpha\mathcal{L}_m+(1-\alpha)\mathcal{L}_u$ over masked and unmasked frames;
  $\alpha=1$ (masked only) works best.
- Parameter counts: BASE 95M, LARGE 317M, X-LARGE 964M.

### 6.4 WavLM

WavLM (Chen et al., 2022) keeps HuBERT-style masked prediction but targets full-stack tasks
(speaker, diarization, separation) with two changes:

- **Utterance mixing (denoising)**: for a random subset of the batch, overlay a segment of
  another utterance (energy ratio $\sim\mathcal{U}(-5,5)$ dB) or DNS noise
  ($\sim\mathcal{U}(-5,20)$ dB) on less than 50% of the primary utterance. The model still
  predicts the *primary* speaker's pseudo-labels.
- **Gated relative position bias** added to attention logits:
  $a_{ij}\propto\exp\{q_i\cdot k_j/\sqrt{d_k}+r_{i-j}\}$, with
  $g^{\text{update}}_i,g^{\text{reset}}_i=\sigma(q_i\cdot u),\sigma(q_i\cdot w)$,
  $\tilde r_{i-j}=w\,g^{\text{reset}}_i d_{i-j}$, and
  $r_{i-j}=d_{i-j}+g^{\text{update}}_i d_{i-j}+(1-g^{\text{update}}_i)\tilde r_{i-j}$.
  $d_{i-j}$ is a bucketed relative position embedding (320 buckets, logarithmic up to offset
  800) shared across layers.
- Data: WavLM Base+ and Large use 94k hours (60k Libri-Light, 10k GigaSpeech, 24k VoxPopuli).
  WavLM Large led SUPERB at publication.

### 6.5 data2vec

A student sees the masked input; an EMA teacher sees the unmasked input. The regression target
is the average of the teacher's top $K$ Transformer block outputs, each instance-normalized
(Baevski et al., 2022). For speech: $K=8$, teacher decay $\tau$ linearly from 0.999 to 0.9999
over 30k updates, wav2vec 2.0 masking ($p=0.065$, span 10), and L2 loss. Continuous
contextualized targets remove the need for a quantizer or k-means.

### 6.6 w2v-BERT and BEST-RQ

- **w2v-BERT** (Chung et al., 2021): a Conformer feature encoder, a *contrastive module*
  (wav2vec 2.0 loss with a quantizer) whose token IDs become targets for a subsequent
  *masked prediction module* (MLM cross-entropy), trained jointly end to end. w2v-BERT 2.0
  (600M, trained on about 4.5M hours) is the encoder of SeamlessM4T v2.
- **BEST-RQ** (Chiu et al., 2022): a random-projection quantizer that is never trained.
  Stack 4 frames of normalized log-mel features into $x$, then

$$
y=\arg\min_i\big\lVert\operatorname{norm}_{\ell_2}(c_i)-\operatorname{norm}_{\ell_2}(Ax)\big\rVert,
$$

  where $A\in\mathbb{R}^{h\times d}$ is Xavier-initialized and frozen, and codebook
  $C=\{c_i\}$ with $n=8192$ entries of dimension $h=16$ is standard-normal and frozen. Mask
  400 ms spans with per-frame probability 0.01 and train a Conformer to predict $y$ with
  softmax cross-entropy. The quantizer is independent of the encoder, so any architecture
  (including streaming models) can use it. NVIDIA's NEST encoder (used by Sortformer) follows
  this idea.

### 6.7 Discrete speech units

Run k-means (50-2000 clusters) on intermediate HuBERT or WavLM features, then deduplicate
consecutive identical IDs. Generative spoken language modeling (GSLM) used HuBERT BASE layer-6
features with 50, 100, or 200 units (Lakhotia et al., 2021). Units are strongly correlated with
phones (HuBERT reports phone purity and PNMI), carry little speaker information, and feed unit
LMs, speech resynthesis, and speech-LLM tokenizers.

### 6.8 Layer-wise probing and SUPERB

SUPERB (Yang et al., 2021) freezes the upstream model and trains light task heads on a
learnable weighted sum of all hidden layers:

$$
h_t=\sum_{l=0}^{L}\operatorname{softmax}(w)_l\,h^{(l)}_t.
$$

Tasks: phoneme recognition (PR), ASR, keyword spotting (KS), query-by-example (QbE), speaker
identification (SID), automatic speaker verification (ASV), speaker diarization (SD), intent
classification (IC), slot filling (SF), and emotion recognition (ER). Learned weights reveal
that speaker information peaks in lower layers and phonetic content in middle-to-upper layers
(WavLM paper, Figure 2). A SUPERB snapshot from the WavLM paper: WavLM Large ASR WER 3.44,
PR PER 3.06, SID accuracy 95.49%, ASV EER 3.77%, SD DER 3.24%.

### 6.9 SSL gotchas

- Input normalization differs by checkpoint: `wav2vec2-base` expects unnormalized audio
  (`do_normalize=False` in its feature extractor), while `wav2vec2-large-lv60` and HuBERT LARGE
  expect normalized audio. A mismatch quietly adds several points of WER.
- Attention masks: some BASE checkpoints were trained without attention masks and expect
  zero-padded batches; large ones need `attention_mask`.
- Freeze the CNN feature encoder during fine-tuning (`model.freeze_feature_encoder()` in HF).
- Fine-tuning collapses to all-blank output for the first few hundred steps; that's normal.
  Use a higher LR for the head (1e-4 to 3e-4 for BASE) with warmup.
- Diversity loss with Gumbel noise included, or codebook temperature not annealed, causes
  codebook collapse; monitor perplexity.

### 6.10 SSL lab ideas

1. **Receptive field and frame count** (CPU-60s): compute receptive field, stride, and output
   length for the wav2vec 2.0 CNN; check 400, 320, and 49 frames for 1 s.
2. **Contrastive and diversity loss** (CPU-60s): implement $\mathcal{L}_m$ with distractor
   sampling from masked frames only and $\mathcal{L}_d$; compare to a reference on fixed
   tensors.
3. **Gumbel product quantizer** (CPU-60s): implement straight-through hard Gumbel with $G=2$,
   check that gradients flow to logits and that codeword indices equal the argmax.
4. **BEST-RQ targets** (CPU-60s): implement the random projection quantizer; check label
   determinism under a fixed seed and codebook utilization on a fixture.
5. **HuBERT units** (CPU-60s with fixture; GPU-L4 for full): extract layer-6 features of
   `facebook/hubert-base-ls960` for a fixture, fit k-means with 50 clusters, and compute
   phone purity against provided frame-level phone labels.
6. **Fine-tune SSL with CTC** (GPU-A100, 10-15 min, about 16-20 GB in fp16 with batch 8):
   fine-tune `facebook/wav2vec2-base` (or `facebook/hubert-base-ls960`) on a 1-hour subset
   of LibriSpeech train-clean-100 for about 2,000 steps; the check requires dev-clean CER
   under about 20%. This is a much shorter schedule than the paper's, so expect WER well above
   the paper's 24.5 for 1 h.
7. **Layer probing** (GPU-L4, about 5 min): train logistic-regression probes on each layer of
   a frozen model for phone and speaker ID; plot the curves and check that phone accuracy
   peaks in middle layers and speaker accuracy in lower layers.

**References.** van den Oord et al., 2018, CPC (https://arxiv.org/abs/1807.03748); Baevski et
al., 2020, wav2vec 2.0 (https://arxiv.org/abs/2006.11477); Hsu et al., 2021, HuBERT
(https://arxiv.org/abs/2106.07447); Chen et al., 2022, WavLM
(https://arxiv.org/abs/2110.13900); Baevski et al., 2022, data2vec
(https://arxiv.org/abs/2202.03555); Chung et al., 2021, w2v-BERT
(https://arxiv.org/abs/2108.06209); Chiu et al., 2022, BEST-RQ
(https://arxiv.org/abs/2202.01855); Yang et al., 2021, SUPERB
(https://arxiv.org/abs/2105.01051); Lakhotia et al., 2021, GSLM
(https://arxiv.org/abs/2102.01192); Jang et al., 2017, Gumbel-softmax
(https://arxiv.org/abs/1611.01144).

---

## 7. Whisper and Whisper-style multilingual models

### 7.1 Architecture

Whisper (Radford et al., 2022) is a plain Transformer encoder-decoder trained on 680k hours
of weakly supervised web audio (large-v3: 1M hours weakly labeled plus 4M hours pseudo-labeled
by large-v2). The encoder has two Conv1d layers (kernel 3, padding 1; the second has stride 2)
with GELU, then sinusoidal positions and pre-LN Transformer blocks. The decoder uses learned
positions, a context of 448 tokens, and ties input and output embeddings. Key projections have
no bias.

| Model | Params | Layers (enc = dec) | Width | Heads |
|---|---|---|---|---|
| tiny | 39M | 4 | 384 | 6 |
| base | 74M | 6 | 512 | 8 |
| small | 244M | 12 | 768 | 12 |
| medium | 769M | 24 | 1024 | 16 |
| large (v1, v2, v3) | 1550M | 32 | 1280 | 20 |
| large-v3-turbo | 809M | 32 enc, 4 dec | 1280 | 20 |

`.en` variants exist for tiny through medium. The official repository and weights are MIT
licensed; some Hugging Face model cards list Apache-2.0 in metadata.

### 7.2 Exact input preprocessing

From `whisper/audio.py`:

```python
SAMPLE_RATE = 16000
N_FFT = 400          # 25 ms window
HOP_LENGTH = 160     # 10 ms hop
CHUNK_LENGTH = 30    # seconds
N_SAMPLES = 480000   # 30 s
N_FRAMES = 3000

window = torch.hann_window(N_FFT)
stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
magnitudes = stft[..., :-1].abs() ** 2          # drop the last frame
mel_spec = filters @ magnitudes                  # librosa.filters.mel(sr=16000, n_fft=400, n_mels=80 or 128)
log_spec = torch.clamp(mel_spec, min=1e-10).log10()
log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)   # 80 dB dynamic range
log_spec = (log_spec + 4.0) / 4.0
```

Walk-through for one window: pad or trim audio to 480,000 samples; `torch.stft` with its
default `center=True` produces $1+480000/160=3001$ frames; dropping the last gives 3000;
the mel matrix gives $80\times3000$ (large-v3 and turbo use 128 mels); conv2 with stride 2
gives 1500 encoder positions, one per 20 ms. The filter bank uses librosa's default Slaney
normalization and Slaney mel scale. Common student bugs: using `center=False`, natural log
instead of log10, applying the max-8 clamp per frame instead of per utterance, or forgetting
to pad to 30 s (the encoder's positional embedding has exactly 1500 positions).

### 7.3 Multitask token format

Special tokens from `whisper/tokenizer.py`: `<|endoftext|>`, `<|startoftranscript|>`, one
language token per language (99; large-v3 adds Cantonese, `<|yue|>`), `<|translate|>`,
`<|transcribe|>`, `<|startoflm|>`, `<|startofprev|>`, `<|nospeech|>` (named
`<|nocaptions|>` in early versions), `<|notimestamps|>`, and 1,501 timestamp tokens
`<|0.00|>` to `<|30.00|>` in 0.02 s steps. Vocabulary sizes are 51,864 (`.en`), 51,865
(multilingual), and 51,866 (large-v3).

A training or decoding sequence looks like the following:

```text
<|startofprev|> previous window's text
<|startoftranscript|><|en|><|transcribe|><|0.00|> Hello world.<|2.40|><|2.40|> How are you?<|5.12|><|endoftext|>
```

Without timestamps, `<|notimestamps|>` follows the task token. Language ID is a single decoder
step after `<|startoftranscript|>`, with logits restricted to language tokens.

**Timestamp rules** (`ApplyTimestampRules` in `decoding.py`): timestamps come in pairs except
at the start; they must be non-decreasing; the first timestamp is capped by
`max_initial_timestamp` (1.0 s); and if the total probability mass on timestamp tokens exceeds
the maximum text-token probability, the decoder must emit a timestamp.

### 7.4 Long-form transcription and hallucination

The reference `transcribe()` decodes sequential 30 s windows and advances the `seek` pointer
to the last predicted timestamp, conditioning on the previous window's text
(`condition_on_previous_text=True`). Temperature fallback retries with temperatures
`(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` when the gzip compression ratio exceeds 2.4 (repetition) or
the average log-probability is below -1.0. A window counts as silence if `no_speech_prob > 0.6`
and the average log-probability is below -1.0. An optional `hallucination_silence_threshold`
skips long silences around suspected hallucinations.

Alternatives: the Hugging Face pipeline's chunked algorithm (overlapping 30 s chunks with
strides, batched, faster but slightly less accurate), and WhisperX (Bain et al., 2023):
VAD-based cut and merge, batched inference, and wav2vec 2.0 forced alignment for word
timestamps.

Hallucinations: Koenecke et al. (ACM FAccT 2024) found that about 1% of transcriptions
contained fabricated phrases absent from the audio, 38% of which included explicit harms, with
higher rates for speakers with aphasia and long non-vocal durations. Mitigations are VAD
before decoding, disabling conditioning on previous text, suppressing blank or non-speech
tokens, and checking compression ratios. Cohere Transcribe's model card also warns that it
hallucinates on non-speech audio and recommends VAD.

### 7.5 Variants

- **large-v3-turbo** (October 2024): large-v3 with the decoder pruned from 32 to 4 layers and
  then fine-tuned; 809M parameters. Much faster, with minor quality loss and weaker
  translation.
- **Distil-Whisper** (Gandhi et al., 2023): keeps the full encoder and distills into a 2-layer
  decoder with pseudo-labels and a KL term, discarding pseudo-labels with WER above 10%.
  `distil-large-v3.5` (756M, English only, MIT) trained on 98k hours with a longer "patient
  teacher" schedule, SpecAugment, and BPE dropout; it's about 1.5x faster than turbo. It can
  serve as the draft model for speculative decoding with large-v3.
- **OWSM** (ESPnet, Peng et al.): open Whisper-style models trained only on public data with
  open code. v3.1 uses E-Branchformer; OWSM-CTC is a non-autoregressive CTC variant; OWSM v4
  (Interspeech 2025 best student paper) adds a cleaned 166k-hour YODAS subset in 75 languages.
  `espnet/owsm_ctc_v4_1B` scores 7.28 average WER on the English leaderboard.
- **MMS** (Pratap et al., 2023, Meta): wav2vec 2.0 pretrained on about 1,400 languages, CTC ASR
  with per-language adapters for 1,107 languages (`facebook/mms-1b-all` loads about 1,160
  adapters), LID for 4,017 languages. License CC-BY-NC 4.0.
- **Omnilingual ASR** (Meta, November 2025, arXiv 2511.09690): wav2vec 2.0 encoders from 300M
  to 7B parameters pretrained on 4.3M hours in 1,600+ languages, with CTC heads or an
  LLM-style autoregressive decoder that supports zero-shot in-context examples. Apache-2.0
  models; a CC-BY corpus covers 350 underserved languages. `facebook/omniASR-LLM-7B-v2` and
  `omniASR-CTC-*-v2` checkpoints are on the leaderboard.
- **SeamlessM4T** (Meta, 2023) and SeamlessM4T v2: multitask speech and text translation and
  ASR for about 100 languages using a w2v-BERT 2.0 encoder; CC-BY-NC 4.0.

**Lab ideas.**

1. **Whisper log-mel from scratch** (CPU-60s): implement `log_mel_spectrogram` in numpy with a
   provided mel matrix; the check requires max abs difference under 1e-4 against
   `whisper.audio.log_mel_spectrogram` output stored in a fixture.
2. **Own greedy decoding loop** (CPU-60s): load `openai/whisper-tiny` weights (151 MB
   safetensors, fp32), build the SOT sequence, run the decoder step by step with or without a
   KV cache, apply `suppress_tokens`, and stop at `<|endoftext|>`. The check requires token
   IDs identical to `model.generate(num_beams=1)` on 3 fixture clips. Tiny runs a 30 s window
   in about 1-2 s on CPU.
3. **Beam search with length normalization and timestamp rules** (CPU-60s for tiny): extend
   lab 2; check the timestamp rules on crafted logits.
4. **Language ID** (CPU-60s): restrict first-step logits to language tokens and report top-3
   languages for FLEURS fixture clips.
5. **Long-form with seek and fallback** (GPU-L4, about 5 min): implement sequential
   transcription with temperature fallback on a 10-minute clip using `whisper-small`; compare
   WER with and without `condition_on_previous_text`.

**References.** Radford et al., 2022, Whisper (https://arxiv.org/abs/2212.04356);
Whisper code (https://github.com/openai/whisper); Gandhi et al., 2023, Distil-Whisper
(https://arxiv.org/abs/2311.00430); Bain et al., 2023, WhisperX
(https://arxiv.org/abs/2303.00747); Koenecke et al., 2024, "Careless Whisper"
(https://arxiv.org/abs/2402.08021); Peng et al., 2023, OWSM (https://arxiv.org/abs/2309.13876);
Peng et al., 2024, OWSM v3.1 (https://arxiv.org/abs/2401.16658); Peng et al., 2024, OWSM-CTC
(https://arxiv.org/abs/2402.12654); Peng et al., 2025, OWSM v4
(https://arxiv.org/abs/2506.00338); Pratap et al., 2023, MMS
(https://arxiv.org/abs/2305.13516); Omnilingual ASR team, 2025
(https://arxiv.org/abs/2511.09690); Seamless Communication, 2023, SeamlessM4T
(https://arxiv.org/abs/2308.11596) and Seamless (https://arxiv.org/abs/2312.05187).

---

## 8. The open ASR landscape in 2025-2026

### 8.1 How the Open ASR Leaderboard works (state on 2026-09-28)

The Hugging Face Open ASR Leaderboard (Srivastav et al., arXiv 2510.06961, revised March
2026) ranks models by macro-averaged WER after text normalization, and reports RTFx (total
audio seconds divided by total processing seconds, higher is faster). The paper version
evaluates 86 systems on 12 datasets with English short-form, English long-form, and
multilingual tracks. Evaluation moved to H200 GPUs and an updated English normalizer
(better mappings, compound merging with kaldialign) on 2026-06-24.

The default English short-form average changed substantially during 2026, so **numbers on
model cards from before mid-2026 aren't comparable with the live board**:

- 2026-05-05: a private-data tab (Appen and DataoceanAI, scripted and conversational, US, UK,
  AU, CA, and IN accents) was added and later included in the default average (2026-07-24).
- 2026-05-20: TED-LIUM v3 was removed from the main and long-form tabs because of a license
  change in the original data.
- 2026-07-07 and 2026-08-21: VoxPopuli, AMI, GigaSpeech, and Earnings-22 were replaced with
  cleaned versions (Artificial Analysis cleaned VoxPopuli and chunked Earnings-22; AMI and
  GigaSpeech drop clips under 1 s).
- 2026-08-28: Voice Arena Monsoon (conversational Indian-accent English) was added.
- 2026-08-21: a "benchmark fitting" tab, built with Hume AI, measures how often models
  reproduce known-wrong VoxPopuli references ("reference-error agreement") and copy each
  benchmark's spelling conventions. The accompanying paper, "Towards quantifying benchmark
  optimization in ASR models" (arXiv 2608.19936), shows top open models sometimes output
  verbatim reference spans even when the audio is masked or contradictory.

The default English average is over AMI-Cleaned, Earnings22-Cleaned-AA-chunked,
GigaSpeech-Cleaned, LibriSpeech test-clean, LibriSpeech test-other, SPGISpeech, Voice Arena
Monsoon, VoxPopuli-AA-Cleaned, Private (scripted), and Private (conversational).

### 8.2 Snapshot: English short-form leaderboard

The top 8 entries are proprietary APIs (Zoom Scribe v2 Pro 4.38, Microsoft Azure Speech
07-2026 4.53, Modulate, Reson8, ElevenLabs Scribe v2 4.84, and others). The following table
lists selected open-weight models, with WER and RTFx from the live configuration fetched on
2026-09-28 and parameters and licenses from model cards.

| Rank | Model | Avg WER | RTFx | Params | License | Architecture notes |
|---|---|---|---|---|---|---|
| 9 | `Qwen/Qwen3-ASR-1.7B-hf` | 4.95 | 820 | about 2B | Apache-2.0 | Qwen3-Omni audio encoder + Qwen3 LLM; 52 languages and dialects; streaming and offline; January 2026 |
| 13 | `AutoArk-AI/ARK-ASR-3B` | 5.18 | 482 | about 3B | Apache-2.0 | Whisper-style encoder, MLP adapter, Qwen decoder; on-policy distillation; May 2026; needs `trust_remote_code` |
| 16 | `nvidia/canary-qwen-2.5b` | 5.23 | 867 | 2.5B | CC-BY-4.0 | SALM: FastConformer from canary-1b-flash + linear projection + Qwen3-1.7B with LoRA; English; 234k h; audio up to 40 s; July 2025 |
| 20 | `CohereLabs/cohere-transcribe-03-2026` | 5.40 | 907 | 2B | Apache-2.0 | Large FastConformer encoder + light Transformer decoder, cross-entropy; 14 languages; no timestamps, no diarization, no LID; March 2026 |
| 21 | `nvidia/parakeet-tdt-0.6b-v2` | 5.42-5.48 | 6,025 (14,199 with a fast GPU decoder) | 0.6B | CC-BY-4.0 | FastConformer-TDT; English; 120k h Granary; word timestamps; up to 24 min per pass; May 2025 |
| 22 | `ibm-granite/granite-speech-4.1-2b` | 5.43 | 546 | 2B | Apache-2.0 | 16-block Conformer CTC encoder, window Q-Former projector (5x downsampling), Granite 4.0-1B LLM; 6 languages; keyword biasing; April 2026 |
| 25 | `nvidia/parakeet-tdt-0.6b-v3` | 5.54-5.66 | 6,076 (13,155 fast decoder) | 0.6B | CC-BY-4.0 | FastConformer-TDT; 25 European languages with automatic LID; 8,192-token SentencePiece; August 2025 |
| 26 | `ibm-granite/granite-speech-5.0-470m-turboctc-nc` | 5.55 | 12,762 | 470M | Apache-2.0 | Pure CTC, 16 Conformer blocks, 16,384 BPE, 8x subsampling to 12.5 Hz, self-conditioned CTC; August 2026 |
| 27 | `ibm-granite/granite-speech-4.1-2b-nar` | 5.60 | 2,074 | 2B | Apache-2.0 | Non-autoregressive: bidirectional LLM edits the CTC hypothesis in one pass (NLE, arXiv 2603.08397) |
| 31 | `nvidia/canary-1b-flash` | 5.71 | 2,129 | 883M | CC-BY-4.0 | 32-layer FastConformer encoder, 4-layer Transformer decoder; en, de, fr, es ASR and translation |
| 32 | `microsoft/Phi-4-multimodal-instruct` | 5.72 | 163 | 5.6B | MIT | 3 conv + 24 Conformer blocks (460M) at 80 ms per token, 2-layer MLP projector, speech LoRA rank 320 on Phi-4-mini; February 2025 |
| 34 | `mistralai/Voxtral-Small-24B-2507` | 5.79 | 101 | 24.3B | Apache-2.0 | Whisper-large-v3-based encoder on 30 s chunks, 4x MLP downsampling to 12.5 Hz, Mistral Small 3.1 decoder |
| 44 | `distil-whisper/distil-large-v3.5` | 6.20 | 888 | 756M | MIT | 2-layer decoder distillation |
| 47 | `mistralai/Voxtral-Mini-3B-2507` | 6.40 | 181 | 4.7B total | Apache-2.0 | Same encoder, Ministral 3B decoder; up to 30 min transcription |
| 49 | `openai/whisper-large-v3` | 6.50 | 470 | 1.55B | MIT | Baseline |
| 50 | `kyutai/stt-2.6b-en` | 6.52 | 133 | 2.6B | CC-BY-4.0 | Decoder-only delayed streams modeling on Mimi codec tokens; 2.5 s delay; streaming |
| 57 | `openai/whisper-large-v3-turbo` | 7.03 | 795 | 809M | MIT | 4-layer decoder |
| 61 | `mistralai/Voxtral-Mini-4B-Realtime-2602` | 7.49 | 103 | about 4.4B | Apache-2.0 | Causal encoder (970M) + 3.4B LM, sliding-window attention, 80-2400 ms delay (default 480 ms); 13 languages |

Other notable entries: `nvidia/nemotron-speech-streaming-en-0.6b` (6.00, cache-aware
FastConformer-RNNT, 24 layers, chunk latency 80, 160, 560, or 1,120 ms, NVIDIA Open Model
License, March 2026), `OpenMOSS-Team/MOSS-Transcribe-Diarize` (5.52, 0.9B, Apache-2.0, emits
`[start][S01]text[end]` with speaker labels in one pass for up to 90 min), Meta
`omniASR-LLM-7B-v2` (7.46), `abr-ai/niagara-9m` to `-84m` batch models (edge scale, 14.87 to
8.65), and the Zipformer CR-CTC transducer XL from soundsgoodai (290M, 6.14, RTFx over 19,000).

Moonshine (Moonshine AI, formerly Useful Sensors) was removed from the leaderboard in the
2026-07-31 cleanup of older models but remains the main edge reference: v1 `moonshine-tiny`
(27M, MIT, English) and v2 streaming models (arXiv 2602.12241) with a sliding-window
"ergodic" encoder without positional embeddings: tiny 34M, small 123M, medium 245M (MIT;
medium reports 2.08/5.00 on LibriSpeech clean/other and 6.65 average WER).

### 8.3 What makes the top models work

- **Conformer or FastConformer encoders plus LLM decoders lead on accuracy**; CTC and TDT
  decoders are 10-100x faster at slightly higher WER (leaderboard blog, November 2025).
- **SALM pattern**: a pretrained ASR encoder, a projector that downsamples to 80 ms or slower
  (linear, MLP, or window Q-Former), and an LLM fine-tuned with LoRA on a prompt such as
  "Transcribe the following:". The LLM supplies punctuation, casing, and domain vocabulary.
- **Data scale with pseudo-labels**: Granary (NVIDIA, about 643k hours in 25 languages) and
  Whisper-style pseudo-labeling; the Canary and Parakeet v3 paper reports 1.7M hours of total
  training samples including non-speech audio to reduce hallucinations.
- **Non-autoregressive speed**: pure CTC (Granite TurboCTC, Parakeet CTC), TDT frame skipping,
  and NLE-style parallel editing give RTFx in the thousands.
- **Caveat for teaching**: gaps among the top 20 models are under one WER point, the default
  average has changed composition several times in 2026, and benchmark fitting is measurable.
  Teach students to evaluate on held-out, in-domain audio.

**Lab ideas.**

1. **Reproduce a leaderboard number** (GPU-L4, about 5-10 min, under 8 GB):
   run `nvidia/parakeet-tdt-0.6b-v2` or `openai/whisper-large-v3-turbo` on LibriSpeech
   test-clean, normalize with Whisper's `EnglishTextNormalizer`, and report WER and RTFx;
   the check accepts WER within 0.3 of the published value.
2. **Build a mini SALM** (GPU-A100, about 30-45 min, about 30 GB): freeze a small encoder
   (`whisper-base` encoder or `hubert-base`), add a 2-layer MLP projector with 4x frame
   stacking, attach `Qwen/Qwen3-0.6B` with LoRA, and train on a 50-100 hour subset. The check
   requires dev-clean WER under a threshold such as 15%. This is a stretch project.

**References.** Srivastav et al., Open ASR Leaderboard
(https://arxiv.org/abs/2510.06961; https://huggingface.co/spaces/hf-audio/open_asr_leaderboard);
leaderboard blog (https://huggingface.co/blog/open-asr-leaderboard); benchmark optimization
(https://arxiv.org/abs/2608.19936); Qwen3-ASR (https://arxiv.org/abs/2601.21337); Canary-1B-v2
and Parakeet-TDT-0.6B-v3 (https://arxiv.org/abs/2509.14128); Canary-1B-flash, "Training and
inference efficiency of encoder-decoder speech models" (https://arxiv.org/abs/2503.05931);
Granary (https://arxiv.org/abs/2505.13404); SALM (https://arxiv.org/abs/2310.09424);
Granite-speech (https://arxiv.org/abs/2505.08699); NLE (https://arxiv.org/abs/2603.08397);
Phi-4-multimodal (https://arxiv.org/abs/2503.01743); Voxtral
(https://arxiv.org/abs/2507.13264); Voxtral Realtime (https://arxiv.org/abs/2602.11298);
Kyutai DSM (https://arxiv.org/abs/2509.08753); Moonshine (https://arxiv.org/abs/2410.15608);
Moonshine v2 (https://arxiv.org/abs/2602.12241); model cards on https://huggingface.co.

---

## 9. Streaming ASR

**Intuition.** A streaming recognizer must emit text while audio is still arriving, so the
encoder can't see the whole future. You trade accuracy for latency by limiting right context
(look-ahead) and by processing audio in chunks.

- **Chunked attention mask**: split frames into chunks of size $C$. Frame $i$ attends to frame
  $j$ if $\lfloor j/C\rfloor\le\lfloor i/C\rfloor$ (optionally plus $R$ look-ahead frames and a
  bounded left context of $N_L$ chunks). Train with random chunk sizes ("dynamic chunk
  training," WeNet) so one model serves several latencies.
- **Cache-aware streaming** (Noroozi et al., 2024, NVIDIA): constrain left and right context
  in every FastConformer layer, including causal depthwise convolutions, and cache the
  activations of previous chunks so that the encoder runs incrementally with no train-test
  mismatch. `att_context_size=[left, right]` is in 80 ms frames; latency is about
  $(\text{right}+1)\times80$ ms, for example `[70, 13]` gives 1,120 ms (Nemotron streaming).
- **Emformer** (Shi et al., 2021) and **Moonshine v2** use memory banks or sliding windows
  instead of full attention.
- **Delayed streams modeling** (Kyutai): a decoder-only model reads time-aligned audio tokens
  and writes text with a fixed delay (0.5 s for `stt-1b-en_fr`, 2.5 s for `stt-2.6b-en`).
- **Latency definitions**: algorithmic latency (look-ahead plus chunk), computational latency
  (processing time per chunk), and *emission latency* (time between a word's end in the audio
  and its emission). RNN-T models learn to delay emissions; FastEmit regularization counters
  this.
- **Endpointing**: decide when the user has finished speaking. Options are VAD with a
  trailing-silence threshold (typically 500-800 ms), an end-of-query token predicted by the
  ASR model (Chang et al., 2019), or a semantic VAD that also uses text context (Kyutai
  `stt-1b-en_fr`, whose semantic VAD is in the Rust server).
- **On-device**: RNN-T with a streaming encoder became the standard for on-device ASR (He et
  al., 2019), often with a second-pass rescorer (Sainath et al., 2019).

**Gotchas.** Streaming with a bidirectional LSTM or full attention isn't possible, but
"buffered" pseudo-streaming (re-running a full-context model on overlapping windows) is common
and easy to confuse with true streaming. Causal convolutions need left padding of
`kernel_size - 1`. BatchNorm statistics differ between chunked and full-utterance inference;
many streaming models use LayerNorm in the conv module.

**Lab ideas.**

1. **Chunked attention mask** (CPU-60s): build the boolean mask for chunk size $C$, look-ahead
   $R$, and left chunks $N_L$; check against a reference and verify that outputs for early
   frames don't change when future frames are perturbed.
2. **Cache equivalence** (CPU-60s): run a small causal Conformer layer on a full sequence and
   chunk by chunk with cached keys, values, and conv state; check that the outputs are equal
   within 1e-5.
3. **Latency measurement** (GPU-L4, about 5 min): run Nemotron streaming at 160 ms and
   1,120 ms chunks and plot WER against measured emission latency using forced-aligned word
   end times.

**References.** Noroozi et al., 2024, cache-aware streaming
(https://arxiv.org/abs/2312.17279); Shi et al., 2021, Emformer
(https://arxiv.org/abs/2010.10759); Yao et al., 2021, WeNet (dynamic chunk training)
(https://arxiv.org/abs/2102.01547); He et al., 2019, "Streaming end-to-end speech recognition
for mobile devices" (https://arxiv.org/abs/1811.06621); Sainath et al., 2019, two-pass
(https://arxiv.org/abs/1908.10992); Chang et al., 2019, joint endpointing and decoding
(https://research.google/pubs/pub47988/);
Zeghidour et al., 2025, DSM (https://arxiv.org/abs/2509.08753).

---

## 10. Decoding with language models and contextual biasing

**Shallow fusion.** Interpolate log-probabilities during beam search:

$$
y^*=\arg\max_y\ \log P_{\text{AM}}(y\mid x)+\lambda\log P_{\text{LM}}(y)+\beta\,|y|.
$$

$\lambda$ is typically 0.3-0.6 for subword AED or RNN-T with a neural LM, and larger (about
0.5-2) for character CTC with a word n-gram. $\beta$ is an insertion bonus. Tune both on a dev
set by grid search.

**Internal LM estimation (ILME).** AED and RNN-T models learn an implicit LM from training
transcripts; subtracting an estimate of it, $-\mu\log P_{\text{ILM}}(y)$, improves cross-domain
fusion (Meng et al., 2021).

**n-gram LMs with KenLM.** Build ARPA models with `lmplz -o 4 < corpus.txt > lm.arpa`, convert
to binary with `build_binary`, and query with the `kenlm` Python module (returns log10
scores). Kneser-Ney smoothing is the default. pyctcdecode integrates KenLM with CTC prefix beam
search (word-level LM, `alpha`, `beta`, and hotwords). Note that `kenlm` needs a C++ build,
which may not be available in a browser-based Python runtime; ship a pure-Python bigram LM for
CPU labs.

**Contextual biasing and hotwords.**

- **Shallow-fusion boosting with a prefix trie**: add a bonus $\lambda_b$ for each token that
  extends a partial match of a bias phrase, and subtract the accumulated bonus if the match
  fails (Zhao et al., 2019). pyctcdecode's hotwords and NeMo's word boosting follow this
  pattern.
- **WFST biasing** in hybrid and CTC-WFST decoders: compose a small biasing FST with a
  class-based LM.
- **Neural biasing**: CLAS (Pundak et al., 2018) encodes bias phrases and attends to them;
  deep biasing for RNN-T; LLM-based models such as Granite Speech accept keyword lists in the
  prompt.
- **CTC word spotting** (NeMo CTC-WS; Andrusenko et al., 2024): find context words in CTC
  logits and merge them into the greedy hypothesis, which is fast and needs no retraining.

**Gotchas.** KenLM returns log10, not natural log; convert before mixing with log-softmax
outputs. Word LMs need a word-boundary token for BPE (`▁`) or character (`|` or space)
vocabularies. Scores of different-length hypotheses need the insertion bonus or length
normalization.

**Lab ideas.**

1. **Prefix beam search with a character bigram LM** (CPU-60s): on shipped wav2vec 2.0
   log-probs, implement fusion and grid-search $\alpha$ and $\beta$; the check requires WER
   lower than greedy on the fixture.
2. **Hotword boosting** (CPU-60s): add trie-based boosting for a list of rare names; the check
   requires the names to appear in the output while WER on other utterances doesn't increase
   by more than 1 point.

**References.** Gulcehre et al., 2015, shallow and deep fusion
(https://arxiv.org/abs/1503.03535); Kannan et al., 2018, "An analysis of incorporating an
external LM into a sequence-to-sequence model" (https://arxiv.org/abs/1712.01996); Meng et
al., 2021, ILME (https://arxiv.org/abs/2011.01991); Heafield, 2011, KenLM
(https://kheafield.com/code/kenlm/); pyctcdecode (https://github.com/kensho-technologies/pyctcdecode);
Zhao et al., 2019, "Shallow-fusion end-to-end contextual biasing," Interspeech
(https://www.isca-archive.org/interspeech_2019/zhao19d_interspeech.html); Pundak et al.,
2018, CLAS (https://arxiv.org/abs/1808.02480); Andrusenko et al., 2024, CTC-WS
(https://arxiv.org/abs/2406.07096).

---

## 11. Evaluation: WER, normalization, speed, and datasets

### 11.1 Word error rate

Align the hypothesis to the reference with Levenshtein distance over words:

$$
\operatorname{WER}=\frac{S+D+I}{N},
$$

where $S$, $D$, and $I$ are substitutions, deletions, and insertions, and $N$ is the number of
reference words. Corpus WER sums $S+D+I$ and $N$ over all utterances; it's not the mean of
per-utterance WERs. WER can exceed 100% because insertions are unbounded. CER is the same
computation over characters and is standard for Chinese and Japanese.

Worked example: reference "the cat sat on the mat" (6 words), hypothesis "a cat sat on mat".
One substitution (the to a) and one deletion (the second "the") give WER $=2/6=33.3\%$.

**Pitfalls.**

- **Text normalization dominates** comparisons between models that output punctuation and
  casing and references that don't. Whisper's `EnglishTextNormalizer` lowercases, removes
  bracketed and parenthesized text, removes fillers matching `\b(hmm|mm|mhm|mmm|uh|um)\b`,
  expands contractions ("won't" to "will not", "gonna" to "going to"), standardizes spoken
  numbers to digits, maps British to American spelling from `english.json`, removes most
  punctuation (keeping some currency and percent symbols), and collapses whitespace.
  `BasicTextNormalizer` is the language-agnostic version. The Open ASR Leaderboard uses an
  updated derivative of this normalizer.
- Report the normalizer, the dataset split, and whether an LM was used; LibriSpeech numbers
  with and without an LM differ by 20-40% relative.
- Hyphenation and compounds ("e-mail" versus "email") and number formats count as errors
  unless normalized; the leaderboard's June 2026 normalizer update merges compounds for this
  reason.
- Use `jiwer` or `kaldialign`; implementing Levenshtein yourself is a good lab, but check
  against a library.

### 11.2 Speed metrics

- Real-time factor: $\text{RTF}=\dfrac{\text{processing time}}{\text{audio duration}}$.
- Inverse real-time factor: $\text{RTFx}=1/\text{RTF}$. The leaderboard computes it over the
  whole dataset with each model's chosen batch size, so it's a throughput measure, not a
  latency measure. RTFx 1,000 means 1 hour of audio in 3.6 seconds.
- Always report hardware, precision, and batch size.

### 11.3 Datasets and licenses

| Dataset | Content | Size | License | Notes |
|---|---|---|---|---|
| LibriSpeech | Read audiobooks | 960 h train; test-clean 5.4 h, test-other 5.1 h | CC-BY-4.0 | Normalized uppercase text; saturated near 1-2% WER |
| Libri-Light | Unlabeled audiobooks | 60k h; 10 h, 1 h, and 10 min labeled splits | LibriVox public domain audio | SSL pretraining and low-resource fine-tuning |
| TED-LIUM 3 | TED talks | 452 h | CC-BY-NC-ND-3.0 | Removed from the Open ASR Leaderboard in May 2026 after a license change |
| Common Voice | Crowdsourced read speech, 100+ languages | Varies by release | CC0 | Since October 2025 distributed only through Mozilla Data Collective, not the Hugging Face Hub |
| GigaSpeech | Audiobooks, podcasts, YouTube | 10k h labeled | Apache-2.0 (gated terms) | Punctuation as tags |
| Earnings-21 and Earnings-22 | Earnings calls | 39 h and 119 h | CC-BY-SA-4.0 | Accents, numbers, long-form |
| AMI | Meetings | About 100 h | CC-BY-4.0 | IHM (headset) and SDM (distant) conditions |
| SPGISpeech | Financial calls | 5,000 h | Kensho user agreement (non-commercial) | Fully formatted text |
| VoxPopuli | European Parliament | 400k h unlabeled; 1.8k h transcribed | CC0 | Known reference errors; leaderboard uses a cleaned version |
| FLEURS | Read Wikipedia sentences | About 12 h per language, 102 languages | CC-BY-4.0 | Multilingual ASR and LID |
| MLS | Audiobooks, 8 languages | 50k h | CC-BY-4.0 | Multilingual LibriSpeech |
| CORAAL | Sociolinguistic interviews, African American Language | 159 h | CC-BY-NC-4.0 | Long-form track |

For CPU labs, `hf-internal-testing/librispeech_asr_dummy` (73 short LibriSpeech validation
utterances) is a convenient fixture source.

**Lab ideas.**

1. **WER with backtrace** (CPU-60s): implement Levenshtein alignment that returns $S$, $D$, and
   $I$ counts and an alignment string; check against `jiwer` on random pairs.
2. **Normalization matters** (CPU-60s): score shipped Whisper outputs against raw and
   normalized references and explain the gap; the check verifies computed WERs.
3. **RTFx harness** (GPU-L4, about 5 min): measure RTFx for batch sizes 1, 8, and 32 and plot
   the throughput-latency trade-off.

**References.** Panayotov et al., 2015, LibriSpeech (https://www.openslr.org/12); Kahn et al.,
2020, Libri-Light (https://arxiv.org/abs/1912.07875); Hernandez et al., 2018, TED-LIUM 3
(https://arxiv.org/abs/1805.04699); Ardila et al., 2020, Common Voice
(https://arxiv.org/abs/1912.06670); Chen et al., 2021, GigaSpeech
(https://arxiv.org/abs/2106.06909); Del Rio et al., 2022, Earnings-22
(https://arxiv.org/abs/2203.15591); O'Neill et al., 2021, SPGISpeech
(https://arxiv.org/abs/2104.02014); Wang et al., 2021, VoxPopuli
(https://arxiv.org/abs/2101.00390); Conneau et al., 2022, FLEURS
(https://arxiv.org/abs/2205.12446); Pratap et al., 2020, MLS (https://arxiv.org/abs/2012.03411);
Whisper normalizer (https://github.com/openai/whisper/tree/main/whisper/normalizers).

---

## 12. Speaker tasks

### 12.1 Speaker verification

**Intuition.** Map an utterance of any length to a fixed-size embedding so that embeddings of
the same speaker are close. Verification compares a test embedding with an enrollment
embedding and thresholds a score.

**i-vectors (brief).** A GMM-UBM supervector is modeled as $M=m+Tw$ with total-variability
matrix $T$ and latent $w\sim\mathcal{N}(0,I)$ of 400-600 dimensions; the posterior mean of $w$
is the i-vector, scored with PLDA (Dehak et al., 2011).

**x-vectors.** A TDNN (1-D dilated convolutions) processes frames with growing context, a
statistics pooling layer concatenates the mean and standard deviation over time, and segment
layers produce a 512-D embedding trained with speaker-ID cross-entropy, with PLDA scoring
(Snyder et al., 2018).

**ECAPA-TDNN** (Desplanques et al., 2020): SE-Res2Net blocks (1-D Res2Net with
squeeze-and-excitation), multi-layer feature aggregation (concatenating outputs of all blocks),
and channel- and context-dependent *attentive statistics pooling*:

$$
e_{t,c}=v_c^{\top}f\!\big(Wh_t+b\big)+k_c,\quad
\alpha_{t,c}=\frac{\exp(e_{t,c})}{\sum_\tau\exp(e_{\tau,c})},\quad
\tilde\mu_c=\sum_t\alpha_{t,c}h_{t,c},\quad
\tilde\sigma_c=\sqrt{\textstyle\sum_t\alpha_{t,c}h_{t,c}^2-\tilde\mu_c^2}.
$$

Typical configuration: 80-D fbank input, channels $C=512$ or 1024, 192-D embedding, AAM-softmax
with margin 0.2 and scale 30. `speechbrain/spkrec-ecapa-voxceleb` (Apache-2.0, trained on
VoxCeleb 1+2) reports 0.80% EER on VoxCeleb1-O with cosine scoring.

**ResNet speaker models.** ResNet-34 and ResNet-293 with 2-D convolutions on fbanks, used by
WeSpeaker and by pyannote's embedding model (`pyannote/wespeaker-voxceleb-resnet34-LM`).

### 12.2 Additive angular margin (AAM) softmax

Normalize embeddings $x_i$ and class weights $W_j$ so the logit is $\cos\theta_j$. Add a
margin $m$ to the target angle and scale by $s$ (Deng et al., 2019, ArcFace):

$$
\mathcal{L}_{\text{AAM}}=-\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{s\cos(\theta_{y_i}+m)}}{e^{s\cos(\theta_{y_i}+m)}+\sum_{j\ne y_i}e^{s\cos\theta_j}},\qquad \cos\theta_j=\frac{W_j^{\top}x_i}{\lVert W_j\rVert\lVert x_i\rVert}.
$$

Additive margin (CosFace, AM-softmax) uses $s(\cos\theta_{y_i}-m)$ instead.

**Worked example** ($s=30$, $m=0.2$, two classes): target cosine 0.5
($\theta=1.0472$ rad), other cosine 0.4. Plain scaled softmax gives logits 15 and 12,
$p=0.9526$, loss 0.0486. With AAM, $\cos(1.0472+0.2)=0.3180$, the target logit is 9.54, so
$p=0.0787$ and the loss is 2.54. The margin keeps pushing until the target angle is at least
$m$ smaller than any other.

**Gotchas.** Implementations use the "easy margin" or a fallback when $\theta+m>\pi$ (the
cosine is no longer monotonic); compute $\cos(\theta+m)=\cos\theta\cos m-\sin\theta\sin m$
with $\sin\theta=\sqrt{\max(0,1-\cos^2\theta)}$ and clamp. Margin warmup over the first epochs
stabilizes training.

### 12.3 Scoring and metrics

- **Cosine scoring**: $\operatorname{score}=\frac{e_1^{\top}e_2}{\lVert e_1\rVert\lVert e_2\rVert}$,
  often with adaptive score normalization (AS-Norm) against a cohort.
- **EER**: the operating point where the false-acceptance rate equals the false-rejection
  (miss) rate.
- **Detection cost function**:
  $C_{\text{det}}(\theta)=C_{\text{miss}}P_{\text{miss}}(\theta)P_{\text{tar}}+C_{\text{fa}}P_{\text{fa}}(\theta)(1-P_{\text{tar}})$,
  normalized by $C_{\text{def}}=\min\big(C_{\text{miss}}P_{\text{tar}},\ C_{\text{fa}}(1-P_{\text{tar}})\big)$.
  minDCF is the minimum over thresholds. VoxSRC uses $P_{\text{tar}}=0.05$ and
  $C_{\text{miss}}=C_{\text{fa}}=1$; NIST SRE evaluations use 0.01 or 0.005.

**Worked example.** Target scores {0.9, 0.8, 0.7, 0.4}; non-target scores {0.6, 0.3, 0.2, 0.1}.
At threshold 0.5, one target is rejected (0.4) and one non-target accepted (0.6), so miss = FA =
25% and EER = 25%. With $P_{\text{tar}}=0.05$, normalized DCF is $P_{\text{miss}}+19P_{\text{fa}}$;
at a threshold of 0.65 it's $0.25+0=0.25$, which is the minimum, so minDCF = 0.25.

### 12.4 Speaker diarization

**Intuition.** Answer "who spoke when" without knowing the speakers in advance.

**Clustering pipeline.** VAD, then segmentation into short windows (1.5 s with 0.75 s shift
is classic), then speaker embeddings per window, then clustering, then resegmentation and
overlap handling.

- **Agglomerative hierarchical clustering** with a cosine or PLDA threshold.
- **Spectral clustering** (Wang et al., 2018; NME-SC by Park et al., 2020): build a cosine
  affinity matrix $A$, refine it (row-wise $p$-percentile thresholding, symmetrization
  $\max(A,A^{\top})$, and optionally Gaussian blur and diffusion $AA^{\top}$), compute the
  normalized Laplacian $L=I-D^{-1/2}AD^{-1/2}$, pick the number of speakers $k$ with the
  largest eigengap $\lambda_{k+1}-\lambda_k$, and run k-means on the rows of the first $k$
  eigenvectors.
- **VBx** (Landini et al., 2022): a Bayesian HMM over x-vectors with PLDA; used by pyannote's
  community-1 pipeline.

**pyannote.audio.** Version 3.1 pipeline: `pyannote/segmentation-3.0` (a small powerset model
on 10 s chunks predicting up to 3 local speakers, including overlap, as a multiclass
problem), WeSpeaker ResNet-34 embeddings, and agglomerative clustering. pyannote.audio 4.0 with
`speaker-diarization-community-1` (CC-BY-4.0, gated on Hugging Face) switches to VBx clustering
and adds an "exclusive" mode with at most one active speaker for aligning with ASR words. DER
(%), 3.1 versus community-1: AISHELL-4 12.2 versus 11.7, AliMeeting 24.5 versus 20.3,
AMI-IHM 18.8 versus 17.0, DIHARD 3 21.4 versus 20.2 (model card; the commercial precision-2
model is lower).

**End-to-end neural diarization (EEND).** A Transformer outputs per-frame, per-speaker
activity probabilities $\hat y_{t,c}\in[0,1]$ for up to $C$ speakers, trained with
permutation-invariant BCE (Fujita et al., 2019):

$$
\mathcal{L}_{\text{PIT}}=\frac{1}{TC}\min_{\phi\in\operatorname{perm}(C)}\sum_{t,c}\operatorname{BCE}\big(\hat y_{t,c},\,y_{t,\phi(c)}\big).
$$

EEND-EDA (Horiguchi et al., 2020) adds encoder-decoder attractors for a variable number of
speakers. Overlap is handled naturally, unlike clustering.

**Sortformer** (Park et al., ICML 2025, NVIDIA): an EEND-style Transformer on a NEST encoder
trained with *Sort Loss*, BCE against targets sorted by each speaker's arrival time, alone or
combined with PIT. The fixed arrival-time order removes the permutation ambiguity and lets
speaker outputs feed multi-speaker ASR through sinusoidal speaker kernels. Streaming
Sortformer (arXiv 2507.18446) keeps an arrival-order speaker cache for online use.

**Diarization error rate.**

$$
\operatorname{DER}=\frac{\text{false alarm}+\text{missed speech}+\text{speaker confusion}}{\text{total reference speech time}}.
$$

Speaker confusion uses the optimal one-to-one mapping between hypothesis and reference
speakers (Hungarian algorithm). Report the collar (0.25 s was traditional; 0 s is standard
now) and whether overlap is scored. Example: 10 s of reference speech with 0.5 s missed,
0.3 s false alarm, and 1.2 s confused gives DER = 20%.

### 12.5 Keyword spotting

Google Speech Commands v0.02 (Warden, 2018): 105,829 one-second 16 kHz utterances of 35 words
from 2,618 speakers, CC-BY-4.0. The standard 12-class task uses 10 words ("yes", "no", "up",
"down", "left", "right", "on", "off", "stop", "go") plus "silence" and "unknown"; a 35-class
variant also exists. Small CNNs on 40-D log-mel or MFCC features reach about 95% on the
12-class task, and BC-ResNet or Keyword Transformer reach about 98%. Report accuracy and, for
always-on detection, false alarms per hour at a fixed miss rate.

**Lab ideas.**

1. **AAM-softmax** (CPU-60s): implement the loss with the margin fallback; check against the
   worked example and a reference implementation.
2. **EER and minDCF** (CPU-60s): compute both from score arrays with a threshold sweep; check
   the worked example and a random case.
3. **Attentive statistics pooling** (CPU-60s): implement the ECAPA pooling; compare to a
   reference on fixed weights.
4. **Verification with pretrained embeddings** (CPU-60s with 20 fixture clips): extract ECAPA
   embeddings, score trials with cosine similarity, and report EER.
5. **Spectral clustering and DER** (CPU-60s): cluster synthetic embeddings with eigengap-based
   speaker counting and compute DER with Hungarian mapping; check the speaker count and DER
   against references.
6. **PIT versus Sort Loss** (CPU-60s): implement both for $C=3$ and verify that Sort Loss equals
   BCE against arrival-time-sorted targets.
7. **KWS CNN** (CPU-60s on a precomputed 10-word, 2,000-clip MFCC subset; GPU-L4 about 5-10 min
   for the full dataset): train and require at least 80% (subset) or 94% (full, 12-class)
   test accuracy.

**References.** Dehak et al., 2011, i-vectors (https://ieeexplore.ieee.org/document/5545402);
Snyder et al., 2018, x-vectors (https://www.danielpovey.com/files/2018_icassp_xvectors.pdf);
Desplanques et al., 2020, ECAPA-TDNN (https://arxiv.org/abs/2005.07143); Deng et al., 2019,
ArcFace (https://arxiv.org/abs/1801.07698); Wang et al., 2018, CosFace
(https://arxiv.org/abs/1801.09414); Nagrani et al., VoxCeleb (https://arxiv.org/abs/1706.08612)
and VoxCeleb2 (https://arxiv.org/abs/1806.05622); Wang et al., 2018, spectral clustering
diarization (https://arxiv.org/abs/1710.10468); Park et al., 2020, NME-SC
(https://arxiv.org/abs/2003.02405); Landini et al., 2022, VBx
(https://arxiv.org/abs/2012.14952); Plaquet and Bredin, 2023, powerset diarization
(https://arxiv.org/abs/2310.13025); pyannote community-1
(https://huggingface.co/pyannote/speaker-diarization-community-1); Fujita et al., 2019, EEND
(https://arxiv.org/abs/1909.06247); Horiguchi et al., 2020, EEND-EDA
(https://arxiv.org/abs/2005.09921); Park et al., 2024, Sortformer
(https://arxiv.org/abs/2409.06656); Medennikov et al., 2025, Streaming Sortformer
(https://arxiv.org/abs/2507.18446); Warden, 2018, Speech Commands
(https://arxiv.org/abs/1804.03209); Kim et al., 2021, BC-ResNet
(https://arxiv.org/abs/2106.04140); Berg et al., 2021, Keyword Transformer
(https://arxiv.org/abs/2104.00769); Wang et al., 2023, WeSpeaker
(https://arxiv.org/abs/2210.17016).

---

## 13. Forced alignment and phoneme recognition

**Intuition.** When you know the transcript, alignment is easy: find the single best path
through the CTC (or HMM) trellis constrained to that transcript. The result gives start and end
times per token, which you need for subtitles, dataset segmentation, TTS data preparation, and
diarization-ASR merging.

**CTC Viterbi alignment.** Replace the sum in the CTC forward recursion with a max and keep
backpointers:

$$
\delta_t(s)=y^t_{l'_s}\cdot\max\big(\delta_{t-1}(s),\ \delta_{t-1}(s-1),\ [\delta_{t-1}(s-2)]\big),
$$

with the same skip rule, then backtrack from $\arg\max\{\delta_T(S),\delta_T(S-1)\}$. Frames
assigned to label states give token spans; merge tokens into words at word-delimiter tokens.
`torchaudio.functional.forced_align` implements this (preserved in torchaudio according to the
2026 roadmap update), and the MMS forced aligner model (`MMS_FA` bundle) works for 1,000+
languages with romanized text (`uroman`).

**CTC segmentation** (Kürzinger et al., 2020): align long audio to long text by computing the
trellis with an extra "garbage" state for unaligned audio, then compute per-segment confidence
scores to filter bad utterances. It's available as the `ctc-segmentation` package and in
NeMo and ESPnet.

**Montreal Forced Aligner** (McAuliffe et al., 2017): Kaldi-based GMM-HMM triphone alignment
with speaker adaptation, driven by a pronunciation dictionary and pretrained acoustic models
for many languages; outputs Praat TextGrids at word and phone level. MIT license. It's often
more accurate at phone boundaries than CTC because HMM states model durations and CTC spikes.
Qwen3-ForcedAligner-0.6B (2026) is an LLM-based non-autoregressive timestamp predictor for 11
languages.

**Phoneme recognition.** TIMIT (LDC, not free) is the classic benchmark with 61 phones folded
to 39 for scoring; PER uses the same edit distance as WER. For open labs, use
`facebook/wav2vec2-xlsr-53-espeak-cv-ft` or `facebook/wav2vec2-lv-60-espeak-cv-ft`, wav2vec 2.0
models fine-tuned with CTC to eSpeak IPA phones for zero-shot cross-lingual phone recognition
(Xu et al., 2021), or POWSM (arXiv 2510.24992), a phonetic Whisper-style model.

**Gotchas.** The blank frames between spikes must be assigned somewhere; decide whether word
end times extend into following blanks. CTC alignments from peaky models are offset by up to
one or two frames (20-40 ms); report frame rate. Text must be normalized to the model's
vocabulary (uppercase letters and `|` for `wav2vec2-base-960h`).

**Lab ideas.**

1. **Viterbi forced alignment on CTC emissions** (CPU-60s): on shipped emissions from
   `wav2vec2-base-960h`, return per-word start and end frames; check against
   `torchaudio.functional.forced_align` or a reference within one frame.
2. **Segment long audio** (GPU-L4, about 5 min): align a 30-minute LibriVox chapter to its text
   with CTC segmentation and export utterances with confidence scores.
3. **Phone posteriors and PER** (CPU-60s with fixture): decode a phoneme CTC model's outputs
   and compute PER after mapping to a 39-phone set.

**References.** Kürzinger et al., 2020, CTC segmentation (https://arxiv.org/abs/2007.09127);
McAuliffe et al., 2017, Montreal Forced Aligner
(https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.html); Pratap et al.,
2023, MMS aligner (https://arxiv.org/abs/2305.13516); torchaudio forced alignment tutorial
(https://docs.pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html);
Xu et al., 2021, zero-shot phoneme recognition (https://arxiv.org/abs/2109.11680); Qwen3-ASR
technical report (https://arxiv.org/abs/2601.21337).

---

## 14. Checkpoints and tooling for labs

### 14.1 Small checkpoints

| Checkpoint | Size | License | Use |
|---|---|---|---|
| `openai/whisper-tiny` and `tiny.en` | 39M (151 MB fp32) | MIT | Decoding loops, log-mel, LID |
| `openai/whisper-base` | 74M | MIT | Better long-form demos |
| `facebook/wav2vec2-base-960h` | 95M | Apache-2.0 | CTC logits, greedy and beam decoding, alignment |
| `facebook/wav2vec2-base` | 95M (pretrained only) | Apache-2.0 | CTC fine-tuning |
| `facebook/hubert-base-ls960` | 95M | Apache-2.0 | Units, probing, fine-tuning |
| `microsoft/wavlm-base-plus` | 95M | See card (WavLM code is MIT) | Speaker and probing labs |
| `UsefulSensors/moonshine-tiny` | 27M | MIT | Edge ASR comparison |
| `moonshine-ai/moonshine-streaming-tiny` | 34M | MIT | Streaming demo |
| `speechbrain/spkrec-ecapa-voxceleb` | About 15-20M params (unverified) | Apache-2.0 | Speaker embeddings |
| `pyannote/segmentation-3.0` | About 1.5M params | MIT (gated) | Diarization segmentation |
| `facebook/wav2vec2-xlsr-53-espeak-cv-ft` | 317M | Apache-2.0 | Phoneme recognition |
| `nvidia/parakeet-tdt-0.6b-v2` | 600M | CC-BY-4.0 | GPU labs, TDT decoding |
| `nvidia/canary-180m-flash` | 180M | CC-BY-4.0 | Small multilingual AED on GPU |

### 14.2 Loading without the transformers library

- Hugging Face checkpoints ship `model.safetensors`; read them with
  `safetensors.numpy.load_file` or `safetensors.torch.load_file` and map tensor names into a
  hand-written module. For Whisper, tensor names follow `model.encoder.layers.{i}.self_attn.q_proj.weight`
  in the HF format and `encoder.blocks.{i}.attn.query.weight` in the OpenAI `.pt` format.
- The `openai-whisper` package loads `.pt` checkpoints with only torch, tiktoken, numba, and
  small helpers.
- NeMo `.nemo` files are tar archives with `model_config.yaml` and `model_weights.ckpt`; you
  can untar and load the state dict with plain torch, but rebuilding the FastConformer and TDT
  decoding without NeMo is a large task. Keep NeMo models for GPU labs with NeMo installed.
- ONNX exports (for example `onnx-community/*`, sherpa-onnx Zipformer models) run with
  `onnxruntime` on CPU and avoid PyTorch entirely.

### 14.3 Tooling notes

- torchaudio moved to maintenance in 2.9; I/O moved to TorchCodec, and many modules were
  removed. The roadmap issue (pytorch/audio#3902, edited 2026-01-22) lists `lfilter`,
  `RNNTLoss`, the CUDA CTC decoder, `forced_align`, and `overdrive` as preserved. Pin the
  torchaudio version in lab images, and don't rely on `torchaudio.load` or the
  flashlight-based `ctc_decoder` without checking.
- `F.ctc_loss` is in core PyTorch and is the safest reference for auto-checks.
- `jiwer` for WER, `pyannote.metrics` for DER, and `kaldialign` for alignment are small pure or
  near-pure Python dependencies.

---

## 15. Proposed lesson ordering

1. **The alignment problem and CTC loss.** Lab: implement CTC forward and backward in log
   space in numpy and match `F.ctc_loss` values and gradients (CPU-60s).
2. **Decoding CTC.** Lab: greedy and prefix beam search with a character bigram LM on shipped
   wav2vec 2.0 log-probs, beating greedy WER (CPU-60s).
3. **Evaluation done right.** Lab: Levenshtein WER with S, D, and I backtrace, Whisper-style
   normalization, and RTFx measurement; match `jiwer` (CPU-60s).
4. **Encoders and augmentation.** Lab: implement a Conformer block and SpecAugment and match a
   reference; optional GPU extension trains a small Conformer-CTC on train-clean-100
   (CPU-60s; GPU-A100 about 45 min optional).
5. **Transducers.** Lab: RNN-T and TDT forward recursions in log space checked against brute
   force, plus greedy transducer decoding with `max_symbols` (CPU-60s).
6. **Attention models and Whisper.** Lab: Whisper log-mel from scratch and your own greedy and
   beam decoding loop on `whisper-tiny` with timestamp rules, matching `generate()` tokens
   (CPU-60s).
7. **Self-supervised learning.** Lab: wav2vec 2.0 frame math, contrastive and diversity losses,
   and BEST-RQ targets (CPU-60s), then fine-tune `wav2vec2-base` with CTC on 1 hour of
   LibriSpeech to a CER threshold (GPU-A100, about 10-15 min).
8. **Speech units and probing.** Lab: HuBERT layer-6 k-means units with phone purity, and
   layer-wise probes for phones versus speakers (CPU-60s with fixture; GPU-L4 about 5 min for
   the full probe).
9. **The modern landscape and streaming.** Lab: reproduce a leaderboard WER and RTFx for
   Parakeet-TDT or Whisper-turbo, and verify cache-equivalent chunked inference for a causal
   layer (GPU-L4 about 10 min; CPU-60s for the cache check).
10. **Speakers: verification, diarization, and alignment.** Lab: AAM-softmax, EER and minDCF,
    spectral clustering with DER, and CTC Viterbi forced alignment on shipped emissions
    (CPU-60s).

---

## Unverified or uncertain facts

- The Open ASR Leaderboard numbers come from the Gradio configuration fetched on 2026-09-28
  (latest changelog entry 2026-09-25). Ranks change weekly; re-fetch before publishing.
- Parameter counts for Qwen3-ASR-1.7B (card says "2B" including the encoder) and ARK-ASR-3B
  are approximate; exact encoder and decoder splits weren't confirmed.
- pyannote.audio 4.0 and community-1 release date: fetched pages gave inconsistent dates;
  it's most likely September 2025.
- Kyutai STT release date (the model cards don't state one; the DSM paper is September 2025,
  and the models were public by mid-2025).
- ECAPA-TDNN AAM settings (margin 0.2, scale 30) and 192-D embedding are from memory of the
  paper and SpeechBrain recipes, not re-fetched.
- Whisper per-size layer, width, and head counts, the 448-token decoder context, and vocabulary
  sizes are from memory of `whisper/__init__.py` and model configs, not re-fetched. The
  Slaney normalization of the mel filters is inferred from librosa defaults.
- `microsoft/wavlm-base-plus` license on Hugging Face wasn't checked.
- Whether torchaudio's flashlight `ctc_decoder` survives in 2.9 and later wasn't confirmed.
- Runtime and VRAM estimates for GPU labs are engineering estimates, not measurements.
- The ESPnet default CTC weight of 0.3 and pyctcdecode defaults are from memory of the
  codebases.
- Voxtral Mini 4B Realtime total parameter count (970M encoder plus 3.4B LM from the card;
  the name says 4B).

---

## Bibliography

The following list collects the primary references cited in these notes.

**CTC, AED, and transducers**

- Graves et al., 2006, CTC. https://www.cs.toronto.edu/~graves/icml_2006.pdf
- Hannun et al., 2014, prefix beam search. https://arxiv.org/abs/1408.2873
- Hannun, 2017, "Sequence modeling with CTC." https://distill.pub/2017/ctc/
- Zeyer et al., 2021, peaky CTC. https://arxiv.org/abs/2105.14849
- Chan et al., 2016, LAS. https://arxiv.org/abs/1508.01211
- Chorowski et al., 2015, location-aware attention. https://arxiv.org/abs/1506.07503
- Kim et al., 2017, joint CTC-attention. https://arxiv.org/abs/1609.06773
- Watanabe et al., 2017, hybrid CTC/attention. https://doi.org/10.1109/JSTSP.2017.2763455
- Watanabe et al., 2018, ESPnet. https://arxiv.org/abs/1804.00015
- Chorowski and Jaitly, 2017. https://arxiv.org/abs/1612.02695
- Bengio et al., 2015, scheduled sampling. https://arxiv.org/abs/1506.03099
- Szegedy et al., 2016, label smoothing. https://arxiv.org/abs/1512.00567
- Wu et al., 2016, GNMT length penalty. https://arxiv.org/abs/1609.08144
- Graves, 2012, RNN-T. https://arxiv.org/abs/1211.3711
- Li et al., 2019, RNN-T training efficiency. https://arxiv.org/abs/1909.12415
- Kuang et al., 2022, pruned RNN-T. https://arxiv.org/abs/2206.13236
- Ghodsi et al., 2020, stateless predictor. https://research.google/pubs/rnn-transducer-with-stateless-prediction-network/
- Xu et al., 2023, TDT. https://arxiv.org/abs/2304.06795
- Yu et al., 2021, FastEmit. https://arxiv.org/abs/2010.11148

**Encoders and augmentation**

- Gulati et al., 2020, Conformer. https://arxiv.org/abs/2005.08100
- Kim et al., 2022, Squeezeformer. https://arxiv.org/abs/2206.00888
- Kim et al., 2022, E-Branchformer. https://arxiv.org/abs/2210.00077
- Yao et al., 2023, Zipformer. https://arxiv.org/abs/2310.11230
- Rekesh et al., 2023, FastConformer. https://arxiv.org/abs/2305.05084
- Park et al., 2019, SpecAugment. https://arxiv.org/abs/1904.08779
- Kudo and Richardson, 2018, SentencePiece. https://arxiv.org/abs/1808.06226

**Self-supervised learning**

- van den Oord et al., 2018, CPC. https://arxiv.org/abs/1807.03748
- Baevski et al., 2020, wav2vec 2.0. https://arxiv.org/abs/2006.11477
- Hsu et al., 2021, HuBERT. https://arxiv.org/abs/2106.07447
- Chen et al., 2022, WavLM. https://arxiv.org/abs/2110.13900
- Baevski et al., 2022, data2vec. https://arxiv.org/abs/2202.03555
- Chung et al., 2021, w2v-BERT. https://arxiv.org/abs/2108.06209
- Chiu et al., 2022, BEST-RQ. https://arxiv.org/abs/2202.01855
- Yang et al., 2021, SUPERB. https://arxiv.org/abs/2105.01051
- Lakhotia et al., 2021, GSLM. https://arxiv.org/abs/2102.01192

**Whisper, multilingual, and 2025-2026 models**

- Radford et al., 2022, Whisper. https://arxiv.org/abs/2212.04356
- Gandhi et al., 2023, Distil-Whisper. https://arxiv.org/abs/2311.00430
- Koenecke et al., 2024, Whisper hallucinations. https://arxiv.org/abs/2402.08021
- Peng et al., 2025, OWSM v4. https://arxiv.org/abs/2506.00338
- Pratap et al., 2023, MMS. https://arxiv.org/abs/2305.13516
- Omnilingual ASR, 2025. https://arxiv.org/abs/2511.09690
- Seamless Communication, 2023, SeamlessM4T. https://arxiv.org/abs/2308.11596
- Srivastav et al., Open ASR Leaderboard. https://arxiv.org/abs/2510.06961
- Benchmark optimization in ASR, 2026. https://arxiv.org/abs/2608.19936
- Sekoyan et al., 2025, Canary-1B-v2 and Parakeet-TDT-0.6B-v3. https://arxiv.org/abs/2509.14128
- Żelasko et al., 2025, Canary-1B-flash efficiency. https://arxiv.org/abs/2503.05931
- Granary, 2025. https://arxiv.org/abs/2505.13404
- Chen et al., 2023, SALM. https://arxiv.org/abs/2310.09424
- Saon et al., 2025, Granite-speech. https://arxiv.org/abs/2505.08699
- Dekel et al., 2026, NLE. https://arxiv.org/abs/2603.08397
- Qwen team, 2026, Qwen3-ASR. https://arxiv.org/abs/2601.21337
- Microsoft, 2025, Phi-4-mini and Phi-4-multimodal. https://arxiv.org/abs/2503.01743
- Mistral AI, 2025, Voxtral. https://arxiv.org/abs/2507.13264
- Mistral AI, 2026, Voxtral Realtime. https://arxiv.org/abs/2602.11298
- Zeghidour et al., 2025, delayed streams modeling (Kyutai). https://arxiv.org/abs/2509.08753
- Jeffries et al., 2024, Moonshine. https://arxiv.org/abs/2410.15608
- Kudlur et al., 2026, Moonshine v2. https://arxiv.org/abs/2602.12241
- Cohere Transcribe model card. https://huggingface.co/CohereLabs/cohere-transcribe-03-2026

**Streaming, LM fusion, and biasing**

- Noroozi et al., 2024, cache-aware streaming. https://arxiv.org/abs/2312.17279
- Shi et al., 2021, Emformer. https://arxiv.org/abs/2010.10759
- He et al., 2019, on-device streaming RNN-T. https://arxiv.org/abs/1811.06621
- Meng et al., 2021, ILME. https://arxiv.org/abs/2011.01991
- Zhao et al., 2019, shallow-fusion contextual biasing. https://www.isca-archive.org/interspeech_2019/zhao19d_interspeech.html
- Pundak et al., 2018, CLAS. https://arxiv.org/abs/1808.02480
- Andrusenko et al., 2024, CTC-WS. https://arxiv.org/abs/2406.07096
- Heafield, 2011, KenLM. https://kheafield.com/code/kenlm/

**Speaker tasks and alignment**

- Snyder et al., 2018, x-vectors. https://www.danielpovey.com/files/2018_icassp_xvectors.pdf
- Desplanques et al., 2020, ECAPA-TDNN. https://arxiv.org/abs/2005.07143
- Deng et al., 2019, ArcFace. https://arxiv.org/abs/1801.07698
- Fujita et al., 2019, EEND. https://arxiv.org/abs/1909.06247
- Park et al., 2024, Sortformer. https://arxiv.org/abs/2409.06656
- Plaquet and Bredin, 2023, powerset segmentation. https://arxiv.org/abs/2310.13025
- Landini et al., 2022, VBx. https://arxiv.org/abs/2012.14952
- Warden, 2018, Speech Commands. https://arxiv.org/abs/1804.03209
- Kürzinger et al., 2020, CTC segmentation. https://arxiv.org/abs/2007.09127
- McAuliffe et al., 2017, Montreal Forced Aligner. https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.html
- Xu et al., 2021, zero-shot phoneme recognition. https://arxiv.org/abs/2109.11680
