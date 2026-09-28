# Curriculum

The course runs from a sampled waveform to a speech language model in 42
lessons and eight parts. Each lesson is one Markdown file in `content/lessons/`
and one exercise in `content/exercises/`, with the same slug. The research
notes behind each part are in `docs/research/`, named in each part's heading.

Every exercise's checks run on the CPU in under a minute. A lesson marked
**GPU** also has a **Run** experiment that trains or runs a real model on the
RunPod worker, and shrinks itself to a small CPU version when no GPU is there.
For more information, see [GPU lessons](CONTENT_CONTRACT.md#gpu-lessons).

Each entry gives the slug, the title, what the exercise builds, and the demos the
lesson embeds. The demo ids are registered in `frontend/src/demos/index.ts`.

## Part 1 — Sound and signals

Research: `01-signals-and-classical-speech.md`, sections 1 to 8.

| Slug | Title | The exercise | Demos |
|---|---|---|---|
| `01-sound-and-sampling` | Sound and sampling | Amplitude and decibel helpers (RMS, dBFS), sampling a continuous tone, predicting the alias frequency of any tone at any rate, and a windowed-sinc resampler checked by SNR against a directly synthesized tone. | `sampling-and-aliasing` |
| `02-quantization-and-audio-files` | Quantization and audio files | Uniform quantization to N bits with the measured SNR matching 6.02N + 1.76 dB, dither, mu-law companding and 8-bit mu-law codes, and a WAV writer and reader in numpy and `struct` that round-trips a clip byte for byte. | `quantization-and-dither` |
| `03-the-dft-and-the-fft` | The DFT and the FFT | The DFT as a matrix, a recursive radix-2 FFT matching `np.fft.fft`, the real FFT and bin frequencies, window functions matching `scipy.signal.get_window`, and leakage measured as side-lobe level for each window. | `window-leakage` |
| `04-the-short-time-fourier-transform` | The short-time Fourier transform | Framing with exact frame counts and center padding, the STFT, the COLA check for a window and hop, and a least-squares ISTFT that reconstructs a signal to 1e-10. | `stft-explorer` |
| `05-phase-and-griffin-lim` | Phase and Griffin-Lim | Why magnitude alone is not a signal, Griffin-Lim with monotone spectral convergence, the fast Griffin-Lim momentum update, and a phase-vocoder time stretch that keeps a tone's pitch. | `griffin-lim-iterations` |
| `06-filters-convolution-and-reverb` | Filters, convolution, and reverb | FFT convolution with no circular wrap-around, an IIR filter from its difference equation matching `scipy.signal.lfilter`, pre-emphasis, and a synthetic room impulse response with its RT60 measured by Schroeder integration. | `room-reverb` |

## Part 2 — Hearing and speech production

Research: `01-signals-and-classical-speech.md`, sections 9 to 17.

| Slug | Title | The exercise | Demos |
|---|---|---|---|
| `07-hearing-and-the-mel-scale` | Hearing and the mel scale | The HTK and Slaney mel formulas, a triangular mel filterbank matching librosa's to 1e-6, and Whisper's exact log-mel front end (400-point periodic Hann, hop 160, clamp, max minus 8 dB floor, `(x + 4) / 4`). | `mel-filterbank` |
| `08-mfccs-and-normalization` | MFCCs and normalization | The DCT-II with orthonormal scaling, MFCCs from log-mel, liftering, delta features by regression, and per-utterance cepstral mean and variance normalization. | `mfcc-pipeline` |
| `09-speech-production-and-lpc` | Speech production and linear prediction | A source-filter vowel synthesizer from an impulse train and formant resonators, autocorrelation LPC by Levinson-Durbin, the LPC spectral envelope, formants from pole angles, and the real cepstrum. | `vowel-formants` |
| `10-pitch-tracking` | Pitch tracking | Autocorrelation pitch, YIN (difference function, cumulative mean normalized difference, absolute threshold, parabolic interpolation), voicing decisions, and gross pitch error on synthetic voices with known F0. | `pitch-tracker` |
| `11-phonetics-text-and-voice-activity` | Phonetics, text, and voice activity | ARPAbet and IPA, a text normalizer for numbers, ordinals, and currency, dictionary G2P with a rule fallback, and an energy VAD with hysteresis scored by frame F1 on synthetic noisy speech. | none |

## Part 3 — Classical recognition

Research: `01-signals-and-classical-speech.md`, sections 18 to 22.

| Slug | Title | The exercise | Demos |
|---|---|---|---|
| `12-dynamic-time-warping` | Dynamic time warping | The DTW recurrence with a Sakoe-Chiba band, the backtraced path, and a template recognizer that classifies synthetic spoken "words" by nearest template. | `dtw-path` |
| `13-hidden-markov-models` | Hidden Markov models | Diagonal Gaussian log-likelihoods, the forward algorithm, backward, and Viterbi in log space, Baum-Welch re-estimation with monotone likelihood, and an isolated-word recognizer with one left-to-right HMM per word. | `viterbi-trellis` |
| `14-language-models-and-word-error-rate` | Language models and word error rate | The noisy-channel view of recognition, an interpolated Kneser-Ney bigram model with perplexity, Levenshtein alignment with substitutions, deletions, and insertions, WER and CER, and how HMM-GMM systems composed H, C, L, and G. | none |

## Part 4 — Neural speech recognition

Research: `02-asr-and-ssl.md`.

| Slug | Title | The exercise | Demos |
|---|---|---|---|
| `15-neural-acoustic-models` | Neural acoustic models | Frame stacking and context windows, a convolutional subsampling front end with its exact output length, a frame classifier trained in torch on synthetic phone-like sounds, and frame accuracy. | none |
| `16-connectionist-temporal-classification` | Connectionist temporal classification | The blank-extended label sequence, the collapse function, the CTC forward and backward recursions in log space, the loss and its gradient with respect to the logits, all matched against `torch.nn.functional.ctc_loss`. | `ctc-lattice` |
| `17-decoding-ctc` | Decoding CTC | Greedy decoding, CTC prefix beam search checked against brute-force enumeration on small inputs, shallow fusion with a character n-gram model, and a hotword bonus. | `beam-search` |
| `18-attention-encoder-decoders` | Attention encoder-decoders | Dot-product and location-aware attention, a small listen-attend-spell model in torch on synthetic sequences, label smoothing, beam search with length normalization, and the joint CTC-attention loss. | `attention-alignment` |
| `19-transducers-and-streaming` | Transducers and streaming | The RNN-T lattice, its forward recursion checked against brute force, greedy transducer decoding with a per-frame symbol cap, the TDT duration idea, and chunked attention masks that make an encoder streamable. | none |
| `20-conformers-and-specaugment` | Conformers and SpecAugment | **GPU.** SpecAugment masks, relative positional encoding, and a Conformer block (half-step feed-forward, attention, convolution module) checked piece by piece. The run trains a small Conformer-CTC on sequences of spoken digits from `fsdd`. | none |
| `21-self-supervised-speech` | Self-supervised speech | **GPU.** The wav2vec 2.0 frame arithmetic, the contrastive loss with sampled distractors, the diversity loss, HuBERT-style masked prediction of k-means targets, and BEST-RQ targets. The run probes each layer of `wav2vec2-base` and `hubert-base` on real speech. | none |
| `22-whisper-and-large-scale-recognition` | Whisper and large-scale recognition | **GPU.** Whisper's special-token prompt, a greedy and a beam-search decoding loop over the decoder's logits, timestamp rules, Whisper-style text normalization, and WER. The run transcribes `librispeech-dev-clean` with `whisper-tiny` and the reader's own decoder. | none |
| `23-speaker-recognition-and-diarization` | Speaker recognition and diarization | Additive angular margin softmax, cosine scoring, EER and minDCF from trial scores, spectral clustering of segment embeddings with diarization error rate, and CTC forced alignment by Viterbi on given emissions. | none |

## Part 5 — Speech synthesis

Research: `03-tts-vocoders-codecs.md`, and `04-flows-diffusion-generative.md` for flows and GANs.

| Slug | Title | The exercise | Demos |
|---|---|---|---|
| `24-text-to-speech-and-tacotron` | Text to speech and Tacotron | The TTS pipeline from text to waveform, one location-sensitive attention step, the stop-token loss, the guided attention loss, and an alignment diagonality score, in torch. | `attention-alignment` |
| `25-durations-and-alignment-search` | Durations and alignment search | The FastSpeech length regulator, batched with padding, pitch and energy targets per phone, and monotonic alignment search as a dynamic program checked against brute force. | none |
| `26-wavenet-and-autoregressive-vocoders` | WaveNet and autoregressive vocoders | A dilated causal convolution stack with its receptive field checked by an impulse test, the gated activation, mu-law targets, and sample-by-sample generation with a cache that matches the full forward pass. | `wavenet-receptive-field` |
| `27-gan-vocoders` | GAN vocoders | **GPU.** The multi-resolution STFT loss, the multi-period discriminator's reshape, least-squares GAN and feature-matching losses, and the Snake activation. The run trains a small HiFi-GAN-style generator on `librispeech-dev-clean` and plays what it makes. | none |
| `28-normalizing-flows-and-vits` | Normalizing flows and VITS | Affine coupling with its exact log-determinant, ActNorm, an invertible 1x1 convolution, a flow trained on 2-D data, and the VITS KL term between the posterior and the flow-transformed prior. | none |
| `29-evaluating-synthetic-speech` | Evaluating synthetic speech | Mel cepstral distortion with DTW alignment, F0 RMSE over voiced frames, speaker similarity by cosine, bootstrap confidence intervals for MOS, and how intelligibility is scored with an ASR model. | none |

## Part 6 — Codecs and token models

Research: `03-tts-vocoders-codecs.md`.

| Slug | Title | The exercise | Demos |
|---|---|---|---|
| `30-vector-quantization` | Vector quantization | Nearest-codeword quantization, the straight-through estimator, the codebook and commitment losses, EMA codebook updates with dead-code restart, residual vector quantization with falling residual norms, and finite scalar quantization. | `rvq-2d` |
| `31-neural-audio-codecs` | Neural audio codecs | **GPU.** Bitrate from frame rate, codebook count, and codebook size, quantizer dropout, a toy convolutional codec with RVQ, and SI-SDR and mel distance. The run measures bitrate against quality for `encodec-24khz` and `mimi` on real speech. | `codec-bitrate` |
| `32-codec-language-models` | Codec language models | Semantic and acoustic tokens, the MusicGen delay pattern and its inverse, VALL-E's autoregressive and non-autoregressive inputs, MaskGIT-style parallel decoding with a cosine schedule, and a small autoregressive model over synthetic RVQ codes whose loss reaches the known entropy. | `delay-pattern` |

## Part 7 — Diffusion and flow matching

Research: `04-flows-diffusion-generative.md`.

| Slug | Title | The exercise | Demos |
|---|---|---|---|
| `33-diffusion-from-scratch` | Diffusion from scratch | Noise schedules and cumulative products, the closed-form forward process, conversions between noise, data, velocity, and score, the posterior mean and variance, and a small DDPM trained on 2-D data with sample moments checked. | `diffusion-1d` |
| `34-samplers-scores-and-guidance` | Samplers, scores, and guidance | DDIM with its deterministic and stochastic updates, the probability-flow ODE, the EDM noise levels and Heun step, and classifier-free guidance, checked against closed-form Gaussian answers. | none |
| `35-diffusion-for-audio` | Diffusion for audio | **GPU.** DiffWave's dilated residual denoiser and its fast-sampling schedule alignment, Grad-TTS's prior, and latent diffusion as used by AudioLDM and Stable Audio. The run trains a small DiffWave-style vocoder and plays its samples. | none |
| `36-flow-matching` | Flow matching | The optimal-transport conditional path and its target velocity, the flow-matching loss, Euler, midpoint, and RK4 solvers with their convergence orders, minibatch OT pairing, and one round of reflow. | `flow-trajectories` |
| `37-flow-matching-for-speech` | Flow matching for speech | **GPU.** The masked infilling loss of Voicebox and F5-TTS, sway sampling, classifier-free guidance for a flow, and the chunk-causal mask of CosyVoice. The run trains a small conditional flow-matching mel generator on `fsdd` and turns its mels into sound with the reader's Griffin-Lim. | none |
| `38-few-step-generation-and-audio-metrics` | Few-step generation and audio metrics | Consistency and shortcut models, MeanFlow, distillation, the Fréchet audio distance with its Gaussian closed form, and the CLAP score, on 2-D toys and synthetic embeddings. | none |

## Part 8 — Beyond recognition and synthesis

Research: `05-speech-llms-tasks-and-pedagogy.md`, half A.

| Slug | Title | The exercise | Demos |
|---|---|---|---|
| `39-enhancement-and-separation` | Enhancement and separation | Spectral subtraction, the Wiener gain with decision-directed SNR, ideal binary and ratio masks, SI-SDR, and permutation-invariant training, with a small Conv-TasNet that separates two synthetic sources. | `separation-pit` |
| `40-classification-and-voice-conversion` | Audio classification and voice conversion | AST patch embedding with its patch count, the symmetric CLAP contrastive loss, EER for spoof detection, and kNN voice conversion by matching self-supervised features and averaging the k nearest target frames. | none |
| `41-spoken-language-models` | Spoken language models | **GPU.** Discrete units from k-means with deduplication, a unit language model, and the adapter pattern: frame stacking, a projector, and loss masking. The run plugs frozen `whisper-tiny` encoder features into a frozen `smollm2-135m` through a projector the reader trains to transcribe digits from `fsdd`. | none |
| `42-full-duplex-and-voice-agents` | Full-duplex models and voice agents | Moshi-style parallel streams with an inner monologue, text-speech interleaving, an RQ-Transformer's depth step checked against the joint likelihood, a cache-equivalent chunked causal convolution, and a turn-taking state machine with a latency budget. | `delay-pattern`, `latency-timeline` |
