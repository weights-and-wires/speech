# Curriculum

The course runs from a sampled waveform to a speech language model in 66
chapters and eight parts. Each chapter teaches one idea. It is one Markdown
file in `content/lessons/` and one exercise in `content/exercises/`, with the
same slug, and a sitting of about 30 to 45 minutes. The research notes behind
each part are in `docs/research/`, named in each part's heading.

Readers of the earlier 42-lesson course keep their progress. Each old lesson
became one or more of these chapters, and `content/renames.json` maps every old
slug to the new slugs that replace it.

Every exercise's checks run on the CPU in under a minute. A chapter marked
**GPU** also has a **Run** experiment that trains or runs a real model on the
RunPod worker, and shrinks itself to a small CPU version when no GPU is there.
For more information, see [GPU lessons](CONTENT_CONTRACT.md#gpu-lessons).

Each entry gives the slug, the title, what the exercise builds, and the demos the
chapter embeds. The demo ids are registered in `frontend/src/demos/index.ts`.

## Part 1 — Sound and signals

Research: `01-signals-and-classical-speech.md`, sections 1 to 8.

| Slug | Title | The exercise | Demos |
|---|---|---|---|
| `001-what-sound-is` | What sound is | RMS, peak and RMS levels in dBFS, a sampled cosine, and the running 440 Hz example measured end to end. | none |
| `002-sampling-and-aliasing` | Sampling and aliasing | Predicting the alias frequency of any tone at any rate, a Kaiser-windowed sinc low-pass filter, and a resampler checked against `scipy.signal.resample_poly` and by SNR. | `sampling-and-aliasing` |
| `003-quantization-and-dither` | Quantization and dither | Uniform quantization to N bits with the measured SNR matching 6.02N + 1.76 dB, and triangular dither that decouples the error from the signal. | `quantization-and-dither` |
| `004-mu-law-and-wav-files` | Companding and audio files | 8-bit mu-law encoding and decoding, float to 16-bit PCM conversion, and a WAV writer and reader in `struct` whose bytes match Python's `wave` module. | none |
| `005-frequency-and-the-dft` | Frequency and the DFT | The DFT as a matrix of complex sinusoids, Parseval's relation, the real half-spectrum, and bin numbers converted to hertz to find the tones in a frame. | none |
| `006-the-fft-and-windows` | The FFT and window functions | A recursive radix-2 FFT, cosine-sum windows in periodic and symmetric form, and main-lobe and side-lobe leakage measured for each window. | `window-leakage` |
| `007-the-short-time-fourier-transform` | The short-time Fourier transform | Exact frame counts and center padding, an STFT matching `torch.stft`, the COLA check, and a least-squares ISTFT that reconstructs a signal to 1e-10. | `stft-explorer` |
| `008-phase-and-griffin-lim` | Phase and Griffin-Lim | Spectral convergence, Griffin-Lim with monotone convergence, the fast momentum update, and a phase-vocoder time stretch that keeps a tone's pitch. | `griffin-lim-iterations` |
| `009-convolution-and-filters` | Convolution and filters | Direct and FFT convolution with no circular wrap-around, a filter's frequency response, an IIR filter matching `scipy.signal.lfilter`, and pre-emphasis. | none |
| `010-rooms-and-reverb` | Rooms and reverb | A synthetic room impulse response, its Schroeder decay curve, and the RT60 read from the curve's slope. | `room-reverb` |
| `011-microphone-arrays` | Microphone arrays | GCC-PHAT on a fractional delay, delay-and-sum with the uncorrelated-noise SNR gain, MVDR weights under the distortionless constraint, and an NLMS echo canceler scored by ERLE. | none |

## Part 2 — Hearing and speech production

Research: `01-signals-and-classical-speech.md`, sections 9 to 17.

| Slug | Title | The exercise | Demos |
|---|---|---|---|
| `012-hearing-and-the-mel-scale` | Hearing and the mel scale | The HTK and Slaney mel formulas and their inverses, and a triangular mel filterbank matching librosa's. | `mel-filterbank` |
| `013-log-mel-spectrograms` | Log-mel spectrograms | Whisper's exact log-mel front end: padding to 30 seconds, the log compression with its max-minus-8 floor, and the chain from waveform to features. | none |
| `014-mfccs` | MFCCs | Power to decibels, an orthonormal DCT-II down the mel axis, truncation to 13 coefficients, and HTK liftering. | `mfcc-pipeline` |
| `015-deltas-and-normalization` | Deltas and normalization | Regression deltas, per-utterance cepstral mean and variance normalization that cancels a fixed channel, and the classic 39-dimensional features. | none |
| `016-the-source-filter-model` | The source-filter model | A source-filter vowel synthesizer: an impulse train for the source and a cascade of two-pole formant resonators for the filter. | `vowel-formants` |
| `017-linear-prediction` | Linear prediction | The prediction error, autocorrelation LPC by Levinson-Durbin, the LPC spectral envelope, formants from pole angles, and the real cepstrum. | none |
| `018-pitch-tracking` | Pitch tracking | YIN's difference function, cumulative mean normalized difference, absolute threshold, and parabolic interpolation, scored by gross pitch error on synthetic voices with known F0. | `pitch-tracker` |
| `019-phonemes-and-text-normalization` | Phonemes and text normalization | ARPAbet to IPA, a text normalizer for numbers, ordinals, years, and money, and dictionary lookup with a longest-match letter-to-sound fallback. | none |
| `020-voice-activity-detection` | Voice activity detection | Frame log energy, a two-threshold VAD with hangover, and frame F1 on synthetic noisy speech. | none |

## Part 3 — Classical recognition

Research: `01-signals-and-classical-speech.md`, sections 18 to 22.

| Slug | Title | The exercise | Demos |
|---|---|---|---|
| `021-dynamic-time-warping` | Dynamic time warping | The DTW table with a Sakoe-Chiba band checked against brute force, the backtraced path, and a template recognizer for noisy synthetic words. | `dtw-path` |
| `022-hidden-markov-models` | Hidden Markov models | Diagonal Gaussian log-likelihoods, the forward algorithm in log space, and Viterbi alignment. | `viterbi-trellis` |
| `023-training-hmms` | Training HMMs with Baum-Welch | The backward pass, Baum-Welch re-estimation with monotone likelihood, and an isolated-word recognizer with one left-to-right HMM per word. | none |
| `024-hybrid-acoustic-models` | Hybrid acoustic models | A diagonal Gaussian mixture fit with EM, triphone labels and a likelihood-gain state tie, and Viterbi on posteriors divided by the state priors. | none |
| `025-n-gram-language-models` | n-gram language models | An interpolated Kneser-Ney bigram model, its perplexity, and n-best rescoring for a recognizer. | none |
| `026-word-error-rate` | Word error rate | Levenshtein alignment with substitutions, deletions, and insertions, corpus WER (which can exceed 100%), and CER. | none |
| `027-wfsts-and-decoding-graphs` | Weighted finite-state transducers and decoding graphs | Composition of a lexicon with a bigram, tropical shortest paths and a log-semiring total checked by enumerating paths, and a CTC topology checked against `collapse`. | none |

## Part 4 — Neural speech recognition

Research: `02-asr-and-ssl.md`.

| Slug | Title | The exercise | Demos |
|---|---|---|---|
| `028-neural-acoustic-models` | Neural acoustic models | Frame stacking, exact convolution output lengths, a 4x Conv2d subsampling front end, and a frame classifier trained in torch on noisy synthetic phones. | none |
| `029-ctc-many-paths-one-label` | CTC: many paths, one label | The collapse function, every path that spells a target, and the CTC loss by brute force on tiny inputs, matched against `torch.nn.functional.ctc_loss`. | `ctc-lattice` |
| `030-ctc-forward-backward` | Computing CTC efficiently | The blank-extended target, the forward and backward recursions in log space, the gradient with respect to the logits, and a padded batch matching torch. | none |
| `031-decoding-ctc` | Decoding CTC | Greedy decoding and CTC prefix beam search checked against brute-force enumeration of every path. | `beam-search` |
| `032-language-model-fusion` | Language model fusion and hotwords | A smoothed character bigram, a hotword bonus, and both fused into CTC prefix beam search. | none |
| `033-attention-encoder-decoders` | Attention encoder-decoders | Dot-product and location-aware attention, label smoothing, the joint CTC-attention loss, beam search with a length penalty, and a small listen-attend-spell model trained on tone sequences. | `attention-alignment` |
| `034-transducers` | Transducers | The RNN-T lattice and its forward recursion checked against brute force, greedy decoding with a symbol cap, the TDT duration extension, and a tiny streaming transducer. | none |
| `035-streaming-encoders` | Streaming encoders | The chunked self-attention mask, its context and look-ahead delay for each frame, and masked attention that never reads the future chunk. | none |
| `036-conformers-and-specaugment` | Conformers and SpecAugment | **GPU.** SpecAugment masks, Transformer-XL relative positions with the `rel_shift` trick, and a Conformer block module by module. The run trains a small Conformer-CTC on spoken digits from `fsdd` and tests it on an unseen speaker. | none |
| `037-self-supervised-speech` | Self-supervised speech | **GPU.** wav2vec 2.0 frame arithmetic, span masking, the contrastive and diversity losses, HuBERT-style masked prediction, BEST-RQ targets, k-means units, and a linear probe. The run probes each layer of `wav2vec2-base` and `hubert-base`. | none |
| `038-whisper` | Whisper and large-scale recognition | **GPU.** Whisper's forced prompt, greedy and beam-search decoding over any decoder's logits, timestamp rules, a Whisper-style text normalizer, and WER. The run transcribes `librispeech-dev-clean` with `whisper-tiny` and the reader's own decoder. | none |
| `039-speaker-verification` | Speaker embeddings and verification | Additive angular margin softmax, cosine trial scores, EER, minDCF, and embeddings trained for unseen synthetic voices. | none |
| `040-diarization-and-forced-alignment` | Diarization and forced alignment | Spectral clustering with an eigengap speaker count, diarization error rate under the best speaker mapping, and CTC forced alignment by Viterbi. | none |
| `041-asr-recipe` | An ASR training recipe | Byte-pair merges checked against a worked table, speed perturbation by resampling, noise mixed at a chosen SNR, length-bucketed batches under a padding-waste bound, a warmup learning rate, and the mean of checkpoints. | none |
| `042-fine-tuning-speech-models` | Fine-tuning speech models | A LoRA forward and merge checked against a dense product, a freeze mask for the feature encoder and the CTC head, a seeded few-shot CTC loss that beats training from scratch, and a forgetting score on a planted rise. | none |

## Part 5 — Speech synthesis

Research: `03-tts-vocoders-codecs.md`, and `04-flows-diffusion-generative.md` for flows and GANs.

| Slug | Title | The exercise | Demos |
|---|---|---|---|
| `043-text-to-speech-and-tacotron` | Text to speech and Tacotron | One location-sensitive attention step, the stop-token loss, the guided attention loss, a diagonality score, and a toy Tacotron that reads made-up phones aloud. | `attention-alignment` |
| `044-durations-and-alignment-search` | Durations and alignment search | FastSpeech's batched length regulator, per-phone pitch and energy targets, and monotonic alignment search checked against brute force. | none |
| `045-wavenet` | WaveNet | Mu-law coding, a dilated causal convolution stack with its receptive field checked by an impulse test, the gated residual stack, and cached sample-by-sample generation that matches the parallel pass. | `wavenet-receptive-field` |
| `046-gan-vocoders` | GAN vocoders | **GPU.** The Snake activation, the multi-period discriminator's reshape, the multi-resolution STFT loss, and the LSGAN and feature-matching losses. The run trains a small HiFi-GAN-style generator on one `librispeech-dev-clean` speaker and plays what it makes. | none |
| `047-normalizing-flows-and-vits` | Normalizing flows and VITS | Affine coupling, ActNorm, and an invertible 1x1 convolution with log-determinants checked against autograd, a RealNVP trained on 2-D data, and the VITS KL term. | none |
| `048-evaluating-synthetic-speech` | Evaluating synthetic speech | Mel cepstral distortion with DTW alignment, F0 RMSE and voicing error, cosine speaker similarity, bootstrap confidence intervals for MOS, and the WER an ASR model gives synthetic speech. | none |

## Part 6 — Codecs and token models

Research: `03-tts-vocoders-codecs.md`.

| Slug | Title | The exercise | Demos |
|---|---|---|---|
| `049-vector-quantization` | Vector quantization | Nearest-codeword lookup, the straight-through estimator, the codebook and commitment losses, EMA codebook updates with dead-code restart, and the perplexity that detects collapse. | none |
| `050-residual-and-scalar-quantization` | Residual and scalar quantization | Residual vector quantization with falling residual norms and prefix decoding, and finite scalar quantization with its mixed-radix index. | `rvq-2d` |
| `051-neural-audio-codecs` | Neural audio codecs | **GPU.** Bitrate from frame rate, codebook count, and codebook size, quantizer dropout, a toy convolutional codec with RVQ, and SI-SDR and mel distance. The run measures bitrate against quality for `encodec-24khz` and `mimi` on real speech. | `codec-bitrate` |
| `052-codec-language-models` | Codec language models | The MusicGen delay pattern and its inverse, VALL-E's autoregressive and non-autoregressive inputs, MaskGIT-style parallel decoding with a cosine schedule, and a small autoregressive model over synthetic RVQ codes whose loss reaches the known entropy. | `delay-pattern` |

## Part 7 — Diffusion and flow matching

Research: `04-flows-diffusion-generative.md`.

| Slug | Title | The exercise | Demos |
|---|---|---|---|
| `053-diffusion-from-scratch` | Diffusion from scratch | Linear and cosine schedules, the closed-form forward process, conversions between noise, data, velocity, and score, the DDPM posterior, the loss, ancestral sampling, and a DDPM trained on 2-D data. | `diffusion-1d` |
| `054-fast-samplers` | Sampling faster: DDIM and the probability-flow ODE | DDIM's deterministic and stochastic updates, the VP probability-flow ODE, EDM's noise levels, and the Heun sampler, all checked against closed-form Gaussian answers. | none |
| `055-classifier-free-guidance` | Classifier-free guidance | Condition dropout in training, guidance in both conventions, and guidance moving and narrowing a distribution of notes. | none |
| `056-diffusion-for-audio` | Diffusion for audio | **GPU.** DiffWave's step embedding and gated residual block, alignment of a 6-step fast schedule to the training schedule, Grad-TTS's forward statistics, the loss, and a fast sampler. The run trains a small DiffWave-style vocoder on spoken digits and plays its samples. | none |
| `057-flow-matching` | Flow matching | The optimal-transport path and its loss, the closed-form Gaussian velocity field, an Euler sampler that shows why one step gives the data mean, and the bridge to the diffusion convention. | `flow-trajectories` |
| `058-straighter-paths-and-better-solvers` | Straighter paths and better solvers | Euler, midpoint, and RK4 solvers with their convergence orders, minibatch OT pairing, and a straightness measure that shows what one round of reflow does. | `flow-trajectories` |
| `059-flow-matching-for-speech` | Flow matching for speech | **GPU.** Sway sampling, the masked infilling loss of Voicebox and F5-TTS, classifier-free guidance for a flow, CosyVoice's chunk-causal mask, and a mel pseudo-inverse. The run trains a small conditional flow-matching model on `fsdd` and plays the digits it generates. | none |
| `060-few-step-generation-and-audio-metrics` | Few-step generation and audio metrics | The shortcut-model loss and the MeanFlow target for one-step generation, the Fréchet audio distance with its FAD-infinity correction, and the CLAP score. | none |

## Part 8 — Beyond recognition and synthesis

Research: `05-speech-llms-tasks-and-pedagogy.md`, half A.

| Slug | Title | The exercise | Demos |
|---|---|---|---|
| `061-speech-enhancement` | Speech enhancement | SI-SDR, mixing noise at an exact SNR, spectral subtraction, a decision-directed Wiener filter, and ideal masks. | none |
| `062-source-separation` | Separating voices | The permutation-invariant loss and a small Conv-TasNet trained to separate a chirp and a vowel train from a mixture. | `separation-pit` |
| `063-classification-and-voice-conversion` | Audio classification and voice conversion | AST's spectrogram patch count, CLAP's symmetric contrastive loss, EER for spoof detection, and kNN voice conversion by averaging the nearest target frames. | none |
| `064-spoken-language-understanding` | Spoken language understanding | Intent accuracy, micro slot F1, an n-best intent, sentence BLEU with smoothing and the brevity penalty, character F1, and a two-turn dialogue-state merge. | none |
| `065-spoken-language-models` | Spoken language models | **GPU.** Deduplicated k-means units with a bigram unit LM, then Whisper frame stacking, a projector into a frozen language model, and loss masking. The run trains the projector to transcribe digits from `fsdd` with frozen `whisper-tiny` and `smollm2-135m`. | none |
| `066-full-duplex-and-voice-agents` | Full-duplex models and voice agents | Moshi-style parallel streams with delays, text-speech interleaving, an RQ-Transformer's depth step, a chunked causal convolution that matches the full one, and a turn-taking state machine against a latency budget. | `latency-timeline`, `delay-pattern` |
