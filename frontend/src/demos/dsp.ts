/**
 * A small signal-processing library for the demos: the FFT, windows, the
 * short-time Fourier transform, mel filterbanks, and the DCT.
 *
 * Every function here matches its usual Python counterpart, so a demo shows
 * the same numbers the reader's numpy code computes:
 *
 * - `fft` matches `numpy.fft.fft`, and `rfftMagnitude` is `abs(numpy.fft.rfft)`.
 * - `hann` and `hamming` match `scipy.signal.get_window`, periodic by default.
 * - `stft` matches `librosa.stft` with `center=True` and `pad_mode="reflect"`,
 *   which is also what `torch.stft(center=True)` does.
 * - `hzToMel`, `melToHz`, and `melFilterbank` match `librosa`, with Slaney's
 *   formula by default and HTK's on request.
 * - `dct2` matches `scipy.fft.dct(type=2)`, with `norm="ortho"` by default,
 *   and `idct2` inverts it.
 * - `istft` matches `librosa.istft`, `fftConvolve` matches
 *   `scipy.signal.fftconvolve`, and `levinsonDurbin` matches
 *   `scipy.linalg.solve_toeplitz`; `polynomialRoots` matches `numpy.roots`.
 * - `yinDifference`, `cumulativeMeanNormalized`, `yinPick`, `dtw`,
 *   `schroederDecay`, and `decayTime` follow the papers the course cites.
 *
 * `frontend/tests/dsp.test.ts` checks each one against reference values from
 * those libraries. The file uses no browser API and no TypeScript-only syntax,
 * so Node runs the tests on it directly.
 */

// Basics ----------------------------------------------------------------

export function isPowerOfTwo(n: number): boolean {
  return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
}

export function nextPowerOfTwo(n: number): number {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

/** `count` evenly spaced values from `start` to `stop`, both included. */
export function linspace(start: number, stop: number, count: number): Float64Array {
  const out = new Float64Array(count);
  if (count === 1) {
    out[0] = start;
    return out;
  }
  const step = (stop - start) / (count - 1);
  for (let i = 0; i < count; i += 1) out[i] = start + i * step;
  return out;
}

// The FFT ---------------------------------------------------------------

/**
 * An in-place radix-2 FFT over separate real and imaginary arrays.
 *
 * The length must be a power of two. With `inverse`, it computes the inverse
 * transform, divided by the length as `numpy.fft.ifft` does.
 */
export function fft(re: Float64Array, im: Float64Array, inverse = false): void {
  const n = re.length;
  if (im.length !== n) throw new Error("fft: re and im must have the same length");
  if (!isPowerOfTwo(n)) throw new Error(`fft: the length must be a power of two, got ${n}`);

  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }

  const sign = inverse ? 1 : -1;
  for (let size = 2; size <= n; size *= 2) {
    const half = size / 2;
    const angle = (sign * 2 * Math.PI) / size;
    const wr = Math.cos(angle);
    const wi = Math.sin(angle);
    for (let start = 0; start < n; start += size) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < half; k += 1) {
        const a = start + k;
        const b = a + half;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
        const next = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = next;
      }
    }
  }

  if (inverse) {
    for (let i = 0; i < n; i += 1) {
      re[i] /= n;
      im[i] /= n;
    }
  }
}

/**
 * The magnitudes of the non-negative frequency bins, `n / 2 + 1` of them.
 *
 * `frame` is zero-padded, or cut, to `n`, which defaults to the next power of
 * two at least as long as the frame. Bin `k` is at `k * sampleRate / n` Hz.
 */
export function rfftMagnitude(frame: ArrayLike<number>, n = nextPowerOfTwo(frame.length)): Float64Array {
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  const m = Math.min(n, frame.length);
  for (let i = 0; i < m; i += 1) re[i] = frame[i];
  fft(re, im);
  const out = new Float64Array(n / 2 + 1);
  for (let k = 0; k <= n / 2; k += 1) out[k] = Math.hypot(re[k], im[k]);
  return out;
}

// Windows ---------------------------------------------------------------

export type WindowName = "rectangular" | "hann" | "hamming" | "blackman";

function cosineWindow(n: number, a0: number, periodic: boolean): Float64Array {
  const out = new Float64Array(n);
  if (n === 1) {
    out[0] = 1;
    return out;
  }
  const denominator = periodic ? n : n - 1;
  for (let i = 0; i < n; i += 1) out[i] = a0 - (1 - a0) * Math.cos((2 * Math.PI * i) / denominator);
  return out;
}

/**
 * The Hann window. Periodic, the default, suits the STFT: it is the
 * symmetric window one sample longer with its last sample dropped.
 */
export function hann(n: number, periodic = true): Float64Array {
  return cosineWindow(n, 0.5, periodic);
}

/** The Hamming window, `0.54 - 0.46 cos(2 pi i / N)`. */
export function hamming(n: number, periodic = true): Float64Array {
  return cosineWindow(n, 0.54, periodic);
}

/**
 * The Blackman window, `0.42 - 0.5 cos(2 pi i / N) + 0.08 cos(4 pi i / N)`,
 * which buys lower side lobes than Hann with a wider main lobe.
 */
export function blackman(n: number, periodic = true): Float64Array {
  const out = new Float64Array(n);
  if (n === 1) {
    out[0] = 1;
    return out;
  }
  const denominator = periodic ? n : n - 1;
  for (let i = 0; i < n; i += 1) {
    const phase = (2 * Math.PI * i) / denominator;
    out[i] = 0.42 - 0.5 * Math.cos(phase) + 0.08 * Math.cos(2 * phase);
  }
  return out;
}

export function makeWindow(name: WindowName, n: number, periodic = true): Float64Array {
  if (name === "hann") return hann(n, periodic);
  if (name === "hamming") return hamming(n, periodic);
  if (name === "blackman") return blackman(n, periodic);
  return new Float64Array(n).fill(1);
}

// The STFT --------------------------------------------------------------

export type StftOptions = {
  /** The FFT length and the frame length. A power of two. */
  nFft: number;
  /** Samples between the starts of successive frames. */
  hop: number;
  /** A window name, or the window itself, `nFft` long. Hann by default. */
  window?: WindowName | Float64Array;
  /**
   * Pad `nFft / 2` samples at each end, so frame `t` is centred on sample
   * `t * hop`. On by default, as in librosa and torch.
   */
  center?: boolean;
  /** How to pad when centring. Reflect by default. */
  padMode?: "reflect" | "constant";
};

/**
 * A magnitude spectrogram, `bins` rows by `frames` columns, row-major:
 * `data[bin * frames + frame]`. Row 0 is 0 Hz, row `bins - 1` is Nyquist.
 */
export type Spectrogram = {
  bins: number;
  frames: number;
  nFft: number;
  hop: number;
  data: Float32Array;
};

function reflectIndex(i: number, n: number): number {
  if (n === 1) return 0;
  const period = 2 * (n - 1);
  let k = ((i % period) + period) % period;
  if (k >= n) k = period - k;
  return k;
}

/** The short-time Fourier transform's magnitudes. */
export function stft(signal: ArrayLike<number>, options: StftOptions): Spectrogram {
  const { nFft, hop } = options;
  if (!isPowerOfTwo(nFft)) throw new Error(`stft: nFft must be a power of two, got ${nFft}`);
  if (!(hop >= 1)) throw new Error(`stft: hop must be at least 1, got ${hop}`);
  const center = options.center ?? true;
  const padMode = options.padMode ?? "reflect";
  const window =
    options.window instanceof Float64Array ? options.window : makeWindow(options.window ?? "hann", nFft);
  if (window.length !== nFft) throw new Error("stft: the window must be nFft long");

  const n = signal.length;
  const offset = center ? nFft / 2 : 0;
  const padded = n + 2 * offset;
  const frames = padded < nFft ? 0 : 1 + Math.floor((padded - nFft) / hop);
  const bins = nFft / 2 + 1;
  const data = new Float32Array(bins * frames);
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);

  for (let t = 0; t < frames; t += 1) {
    const start = t * hop - offset;
    for (let i = 0; i < nFft; i += 1) {
      const at = start + i;
      let sample: number;
      if (at >= 0 && at < n) sample = signal[at];
      else if (padMode === "reflect" && n > 0) sample = signal[reflectIndex(at, n)];
      else sample = 0;
      re[i] = sample * window[i];
      im[i] = 0;
    }
    fft(re, im);
    for (let k = 0; k < bins; k += 1) data[k * frames + t] = Math.hypot(re[k], im[k]);
  }
  return { bins, frames, nFft, hop, data };
}

// Decibels --------------------------------------------------------------

/**
 * Magnitudes to decibels relative to the largest, `20 log10(x / max)`, or
 * `10 log10` for powers. Anything below `floorDb` is set to `floorDb`, so
 * the result runs from `floorDb` to 0.
 */
export function toDb(
  values: ArrayLike<number>,
  options: { floorDb?: number; power?: boolean; ref?: number } = {},
): Float32Array {
  const floorDb = options.floorDb ?? -80;
  const scale = options.power ? 10 : 20;
  let ref = options.ref ?? 0;
  if (!options.ref) for (let i = 0; i < values.length; i += 1) ref = Math.max(ref, values[i]);
  const out = new Float32Array(values.length);
  if (!(ref > 0)) return out.fill(floorDb);
  for (let i = 0; i < values.length; i += 1) {
    const db = scale * Math.log10(Math.max(values[i], 1e-12) / ref);
    out[i] = db < floorDb ? floorDb : db;
  }
  return out;
}

// Mel -------------------------------------------------------------------

export type MelFormula = "slaney" | "htk";

const SLANEY_STEP = 200 / 3;
const SLANEY_BREAK_HZ = 1000;
const SLANEY_BREAK_MEL = SLANEY_BREAK_HZ / SLANEY_STEP;
const SLANEY_LOG_STEP = Math.log(6.4) / 27;

/**
 * Hertz to mels. Slaney's formula, librosa's default, is linear below 1 kHz
 * and logarithmic above; HTK's is `2595 log10(1 + f / 700)`.
 */
export function hzToMel(hz: number, formula: MelFormula = "slaney"): number {
  if (formula === "htk") return 2595 * Math.log10(1 + hz / 700);
  if (hz < SLANEY_BREAK_HZ) return hz / SLANEY_STEP;
  return SLANEY_BREAK_MEL + Math.log(hz / SLANEY_BREAK_HZ) / SLANEY_LOG_STEP;
}

/** Mels to hertz, the inverse of `hzToMel`. */
export function melToHz(mel: number, formula: MelFormula = "slaney"): number {
  if (formula === "htk") return 700 * (10 ** (mel / 2595) - 1);
  if (mel < SLANEY_BREAK_MEL) return mel * SLANEY_STEP;
  return SLANEY_BREAK_HZ * Math.exp(SLANEY_LOG_STEP * (mel - SLANEY_BREAK_MEL));
}

export type MelOptions = {
  sampleRate: number;
  nFft: number;
  nMels: number;
  fmin?: number;
  /** Defaults to half the sample rate. */
  fmax?: number;
  formula?: MelFormula;
  /**
   * `"slaney"` divides each triangle by its width in Hz, so every band holds
   * the same energy for white noise; `null` leaves each triangle's peak at 1.
   */
  norm?: "slaney" | null;
};

/**
 * Triangular mel filters, `nMels` rows of `nFft / 2 + 1` weights each, as
 * `librosa.filters.mel` builds them. Multiply a power or magnitude spectrum
 * by row `m` and sum to get mel band `m`.
 */
export function melFilterbank(options: MelOptions): Float64Array[] {
  const { sampleRate, nFft, nMels } = options;
  const fmin = options.fmin ?? 0;
  const fmax = options.fmax ?? sampleRate / 2;
  const formula = options.formula ?? "slaney";
  const norm = options.norm === undefined ? "slaney" : options.norm;
  const bins = Math.floor(nFft / 2) + 1;

  const fftFreqs = linspace(0, sampleRate / 2, bins);
  const lowMel = hzToMel(fmin, formula);
  const highMel = hzToMel(fmax, formula);
  const edges = Array.from(linspace(lowMel, highMel, nMels + 2), (mel) => melToHz(mel, formula));

  const bank: Float64Array[] = [];
  for (let m = 0; m < nMels; m += 1) {
    const left = edges[m];
    const centre = edges[m + 1];
    const right = edges[m + 2];
    const row = new Float64Array(bins);
    const scale = norm === "slaney" ? 2 / (right - left) : 1;
    for (let k = 0; k < bins; k += 1) {
      const f = fftFreqs[k];
      const rising = (f - left) / (centre - left);
      const falling = (right - f) / (right - centre);
      row[k] = Math.max(0, Math.min(rising, falling)) * scale;
    }
    bank.push(row);
  }
  return bank;
}

/** Apply a filterbank to a spectrogram, giving `bank.length` rows by the same frames. */
export function applyFilterbank(bank: Float64Array[], spectrogram: Spectrogram): Spectrogram {
  const { frames, bins } = spectrogram;
  const data = new Float32Array(bank.length * frames);
  bank.forEach((row, m) => {
    for (let k = 0; k < bins; k += 1) {
      const w = row[k];
      if (w === 0) continue;
      const source = k * frames;
      const target = m * frames;
      for (let t = 0; t < frames; t += 1) data[target + t] += w * spectrogram.data[source + t];
    }
  });
  return { ...spectrogram, bins: bank.length, data };
}

// The DCT ---------------------------------------------------------------

/**
 * The type-II discrete cosine transform, `2 sum x[n] cos(pi k (2n + 1) / 2N)`.
 * With `norm: "ortho"`, the default, the result is scaled so the transform is
 * orthonormal, as `scipy.fft.dct(x, norm="ortho")` and MFCCs use it; with
 * `"backward"` it is left unscaled, as scipy's default.
 */
export function dct2(x: ArrayLike<number>, norm: "ortho" | "backward" = "ortho"): Float64Array {
  const n = x.length;
  const out = new Float64Array(n);
  for (let k = 0; k < n; k += 1) {
    let sum = 0;
    for (let i = 0; i < n; i += 1) sum += x[i] * Math.cos((Math.PI * k * (2 * i + 1)) / (2 * n));
    let value = 2 * sum;
    if (norm === "ortho") value *= k === 0 ? Math.sqrt(1 / (4 * n)) : Math.sqrt(1 / (2 * n));
    out[k] = value;
  }
  return out;
}

// Test signals ----------------------------------------------------------

/** A sine at `hz`, `seconds` long, as float32 for Web Audio. */
export function sine(hz: number, sampleRate: number, seconds: number, amplitude = 0.5, phase = 0): Float32Array {
  const n = Math.round(seconds * sampleRate);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 1) out[i] = amplitude * Math.sin((2 * Math.PI * hz * i) / sampleRate + phase);
  return out;
}

/** A linear sweep from `f0` to `f1` Hz. */
export function linearChirp(f0: number, f1: number, sampleRate: number, seconds: number, amplitude = 0.5): Float32Array {
  const n = Math.round(seconds * sampleRate);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    const t = i / sampleRate;
    out[i] = amplitude * Math.sin(2 * Math.PI * (f0 * t + ((f1 - f0) * t * t) / (2 * seconds)));
  }
  return out;
}

/** Fades the first and last `seconds` in and out, so playback does not click. */
export function fadeEdges(samples: Float32Array, sampleRate: number, seconds = 0.02): Float32Array {
  const n = Math.min(Math.round(seconds * sampleRate), Math.floor(samples.length / 2));
  for (let i = 0; i < n; i += 1) {
    const g = 0.5 - 0.5 * Math.cos((Math.PI * i) / n);
    samples[i] *= g;
    samples[samples.length - 1 - i] *= g;
  }
  return samples;
}

// Separation metrics ----------------------------------------------------

/**
 * Scale-invariant signal-to-distortion ratio in dB (Le Roux et al., 2019).
 * Both signals lose their mean, the estimate is projected onto the
 * reference, and the ratio is that projection's energy over what is left.
 * Call it as `siSdr(estimate, reference)`. The value depends only on the
 * correlation r of the two zero-mean signals, `10 log10(r^2 / (1 - r^2))`,
 * so it is the same with the arguments swapped; the projection is not.
 */
export function siSdr(estimate: ArrayLike<number>, reference: ArrayLike<number>, eps = 1e-8): number {
  const n = Math.min(estimate.length, reference.length);
  let meanEstimate = 0;
  let meanReference = 0;
  for (let i = 0; i < n; i += 1) {
    meanEstimate += estimate[i];
    meanReference += reference[i];
  }
  meanEstimate /= n || 1;
  meanReference /= n || 1;
  let dot = 0;
  let energy = 0;
  for (let i = 0; i < n; i += 1) {
    const s = reference[i] - meanReference;
    dot += (estimate[i] - meanEstimate) * s;
    energy += s * s;
  }
  const scale = dot / (energy + eps);
  let target = 0;
  let error = 0;
  for (let i = 0; i < n; i += 1) {
    const projected = scale * (reference[i] - meanReference);
    const residual = estimate[i] - meanEstimate - projected;
    target += projected * projected;
    error += residual * residual;
  }
  return 10 * Math.log10((target + eps) / (error + eps));
}

// Seeded randomness ------------------------------------------------------

/**
 * A seeded generator of uniform numbers in `[0, 1)` (mulberry32), so a demo
 * that adds noise or dither draws the same figure in every build.
 */
export function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A standard normal number from a uniform generator, by Box-Muller. */
export function gaussian(random: () => number): number {
  const u = 1 - random();
  const v = random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// The complex STFT and its inverse ---------------------------------------

/**
 * A complex spectrogram, laid out as `Spectrogram` is: `bins` rows by
 * `frames` columns, `re[bin * frames + frame]`.
 */
export type ComplexSpectrogram = {
  bins: number;
  frames: number;
  nFft: number;
  hop: number;
  re: Float64Array;
  im: Float64Array;
};

/** `librosa.stft` itself, complex, with the framing `stft` uses. */
export function stftComplex(signal: ArrayLike<number>, options: StftOptions): ComplexSpectrogram {
  const { nFft, hop } = options;
  if (!isPowerOfTwo(nFft)) throw new Error(`stftComplex: nFft must be a power of two, got ${nFft}`);
  if (!(hop >= 1)) throw new Error(`stftComplex: hop must be at least 1, got ${hop}`);
  const center = options.center ?? true;
  const padMode = options.padMode ?? "reflect";
  const window =
    options.window instanceof Float64Array ? options.window : makeWindow(options.window ?? "hann", nFft);
  if (window.length !== nFft) throw new Error("stftComplex: the window must be nFft long");

  const n = signal.length;
  const offset = center ? nFft / 2 : 0;
  const padded = n + 2 * offset;
  const frames = padded < nFft ? 0 : 1 + Math.floor((padded - nFft) / hop);
  const bins = nFft / 2 + 1;
  const outRe = new Float64Array(bins * frames);
  const outIm = new Float64Array(bins * frames);
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);

  for (let t = 0; t < frames; t += 1) {
    const start = t * hop - offset;
    for (let i = 0; i < nFft; i += 1) {
      const at = start + i;
      let sample: number;
      if (at >= 0 && at < n) sample = signal[at];
      else if (padMode === "reflect" && n > 0) sample = signal[reflectIndex(at, n)];
      else sample = 0;
      re[i] = sample * window[i];
      im[i] = 0;
    }
    fft(re, im);
    for (let k = 0; k < bins; k += 1) {
      outRe[k * frames + t] = re[k];
      outIm[k * frames + t] = im[k];
    }
  }
  return { bins, frames, nFft, hop, re: outRe, im: outIm };
}

/**
 * The inverse STFT by weighted overlap-add, divided by the summed squared
 * window, as `librosa.istft` computes it. It is the least-squares signal
 * whose STFT is closest to `spec`, so `istft(stftComplex(x))` gives `x` back.
 *
 * `length` trims or pads the result; by default a centred transform gives
 * `hop * (frames - 1)` samples, as librosa does.
 */
export function istft(
  spec: ComplexSpectrogram,
  options: { window?: WindowName | Float64Array; center?: boolean; length?: number } = {},
): Float64Array {
  const { nFft, hop, frames, bins } = spec;
  const center = options.center ?? true;
  const window =
    options.window instanceof Float64Array ? options.window : makeWindow(options.window ?? "hann", nFft);
  if (window.length !== nFft) throw new Error("istft: the window must be nFft long");
  const total = nFft + hop * Math.max(0, frames - 1);
  const y = new Float64Array(total);
  const norm = new Float64Array(total);
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);

  for (let t = 0; t < frames; t += 1) {
    // The full spectrum from its non-negative half, as `irfft` builds it.
    for (let k = 0; k < bins; k += 1) {
      re[k] = spec.re[k * frames + t];
      im[k] = k === 0 || k === nFft / 2 ? 0 : spec.im[k * frames + t];
    }
    for (let k = 1; k < nFft / 2; k += 1) {
      re[nFft - k] = re[k];
      im[nFft - k] = -im[k];
    }
    fft(re, im, true);
    const start = t * hop;
    for (let i = 0; i < nFft; i += 1) {
      y[start + i] += re[i] * window[i];
      norm[start + i] += window[i] * window[i];
    }
  }
  for (let i = 0; i < total; i += 1) if (norm[i] > 1e-10) y[i] /= norm[i];

  const offset = center ? nFft / 2 : 0;
  const length = options.length ?? (center ? total - nFft : total);
  const out = new Float64Array(length);
  for (let i = 0; i < length && offset + i < total; i += 1) out[i] = y[offset + i];
  return out;
}

// Convolution and filters ------------------------------------------------

/**
 * The full linear convolution of `a` and `b`, `a.length + b.length - 1`
 * samples, by FFT with enough zero-padding that nothing wraps around.
 * Matches `scipy.signal.fftconvolve(a, b)`.
 */
export function fftConvolve(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const length = a.length + b.length - 1;
  if (a.length === 0 || b.length === 0) return new Float64Array(0);
  const n = nextPowerOfTwo(length);
  const aRe = new Float64Array(n);
  const aIm = new Float64Array(n);
  const bRe = new Float64Array(n);
  const bIm = new Float64Array(n);
  for (let i = 0; i < a.length; i += 1) aRe[i] = a[i];
  for (let i = 0; i < b.length; i += 1) bRe[i] = b[i];
  fft(aRe, aIm);
  fft(bRe, bIm);
  for (let k = 0; k < n; k += 1) {
    const r = aRe[k] * bRe[k] - aIm[k] * bIm[k];
    aIm[k] = aRe[k] * bIm[k] + aIm[k] * bRe[k];
    aRe[k] = r;
  }
  fft(aRe, aIm, true);
  return aRe.slice(0, length);
}

/**
 * Pre-emphasis, `y[n] = x[n] - alpha x[n - 1]`, from rest, as
 * `scipy.signal.lfilter([1, -alpha], [1], x)` computes it.
 */
export function preEmphasis(x: ArrayLike<number>, alpha = 0.97): Float64Array {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i += 1) out[i] = x[i] - (i > 0 ? alpha * x[i - 1] : 0);
  return out;
}

/**
 * A two-pole resonator at `hz` with bandwidth `bandwidth` Hz, Klatt's
 * formant filter: `y[n] = A x[n] + B y[n - 1] + C y[n - 2]` with poles at
 * radius `exp(-pi bandwidth / fs)` and angle `2 pi hz / fs`, and `A` chosen
 * for a gain of 1 at 0 Hz. Cascading a few of them makes a vowel.
 */
export function resonate(x: ArrayLike<number>, hz: number, bandwidth: number, sampleRate: number): Float64Array {
  const c = -Math.exp((-2 * Math.PI * bandwidth) / sampleRate);
  const b = 2 * Math.exp((-Math.PI * bandwidth) / sampleRate) * Math.cos((2 * Math.PI * hz) / sampleRate);
  const a = 1 - b - c;
  const out = new Float64Array(x.length);
  let y1 = 0;
  let y2 = 0;
  for (let i = 0; i < x.length; i += 1) {
    const y = a * x[i] + b * y1 + c * y2;
    out[i] = y;
    y2 = y1;
    y1 = y;
  }
  return out;
}

// Room acoustics ---------------------------------------------------------

/**
 * Schroeder's energy decay curve in dB: the energy left in the impulse
 * response after each sample, `10 log10(sum_{k >= n} h[k]^2 / sum h^2)`.
 * It starts at 0 dB and falls; its slope is the reverberation time.
 */
export function schroederDecay(ir: ArrayLike<number>, floorDb = -200): Float64Array {
  const out = new Float64Array(ir.length);
  let energy = 0;
  for (let i = ir.length - 1; i >= 0; i -= 1) {
    energy += ir[i] * ir[i];
    out[i] = energy;
  }
  const total = out[0] || 1;
  for (let i = 0; i < out.length; i += 1) {
    const db = out[i] > 0 ? 10 * Math.log10(out[i] / total) : floorDb;
    out[i] = Math.max(floorDb, db);
  }
  return out;
}

/**
 * The reverberation time from a decay curve: a least-squares line through
 * the samples between `fromDb` and `toDb`, extrapolated to 60 dB. The
 * defaults, -5 and -35 dB, give T30; -5 and -25 give T20. `null` when the
 * curve never falls that far.
 */
export function decayTime(
  edcDb: ArrayLike<number>,
  sampleRate: number,
  fromDb = -5,
  toDb = -35,
): { rt60: number; slope: number; intercept: number } | null {
  let count = 0;
  let sumT = 0;
  let sumY = 0;
  let sumTT = 0;
  let sumTY = 0;
  let reached = false;
  for (let i = 0; i < edcDb.length; i += 1) {
    const y = edcDb[i];
    if (y < toDb) {
      reached = true;
      break;
    }
    if (y > fromDb) continue;
    const t = i / sampleRate;
    count += 1;
    sumT += t;
    sumY += y;
    sumTT += t * t;
    sumTY += t * y;
  }
  if (!reached || count < 2) return null;
  const slope = (count * sumTY - sumT * sumY) / (count * sumTT - sumT * sumT);
  if (!(slope < 0)) return null;
  const intercept = (sumY - slope * sumT) / count;
  return { rt60: -60 / slope, slope, intercept };
}

// Linear prediction ------------------------------------------------------

/** `r[k] = sum_n x[n] x[n - k]` for `k = 0..maxLag`, the frame zero outside itself. */
export function autocorrelation(x: ArrayLike<number>, maxLag: number): Float64Array {
  const out = new Float64Array(maxLag + 1);
  for (let k = 0; k <= maxLag; k += 1) {
    let sum = 0;
    for (let n = k; n < x.length; n += 1) sum += x[n] * x[n - k];
    out[k] = sum;
  }
  return out;
}

/**
 * The Levinson-Durbin recursion: the predictor `alpha` of `x[n]` from the
 * `order` samples before it, for `A(z) = 1 - sum alpha_k z^-k`, from the
 * autocorrelation `r`. It returns the prediction-error energy and the
 * reflection coefficients, each below 1 in magnitude for a real frame, so
 * `1 / A(z)` is stable. `alpha` matches `scipy.linalg.solve_toeplitz`.
 */
export function levinsonDurbin(
  r: ArrayLike<number>,
  order: number,
): { alpha: Float64Array; error: number; reflection: Float64Array } {
  let alpha = new Float64Array(order);
  const reflection = new Float64Array(order);
  let error = r[0];
  for (let i = 1; i <= order; i += 1) {
    if (!(error > 0)) break;
    let acc = r[i];
    for (let j = 1; j < i; j += 1) acc -= alpha[j - 1] * r[i - j];
    const k = acc / error;
    reflection[i - 1] = k;
    const next = alpha.slice();
    next[i - 1] = k;
    for (let j = 1; j < i; j += 1) next[j - 1] = alpha[j - 1] - k * alpha[i - j - 1];
    alpha = next;
    error *= 1 - k * k;
  }
  return { alpha, error, reflection };
}

/**
 * The all-pole magnitude `gain / |A(e^jw)|` with `A(z) = 1 - sum alpha_k z^-k`,
 * at `points` frequencies from 0 to half the sample rate, both included.
 */
export function allPoleMagnitude(alpha: ArrayLike<number>, gain: number, points: number): Float64Array {
  const out = new Float64Array(points);
  for (let p = 0; p < points; p += 1) {
    const w = points === 1 ? 0 : (Math.PI * p) / (points - 1);
    let re = 1;
    let im = 0;
    for (let k = 1; k <= alpha.length; k += 1) {
      re -= alpha[k - 1] * Math.cos(w * k);
      im += alpha[k - 1] * Math.sin(w * k);
    }
    out[p] = gain / Math.max(1e-12, Math.hypot(re, im));
  }
  return out;
}

/**
 * The roots of a polynomial with real coefficients, highest power first, as
 * `numpy.roots` takes them, by the Durand-Kerner iteration. For LPC, pass
 * `[1, -alpha_1, ..., -alpha_p]`; a root's angle is a formant's frequency.
 */
export function polynomialRoots(coefficients: ArrayLike<number>): { re: Float64Array; im: Float64Array } {
  let first = 0;
  while (first < coefficients.length && coefficients[first] === 0) first += 1;
  let last = coefficients.length - 1;
  let zeros = 0;
  while (last > first && coefficients[last] === 0) {
    last -= 1;
    zeros += 1;
  }
  const degree = last - first;
  const lead = coefficients[first];
  const c = Array.from({ length: degree + 1 }, (_, i) => coefficients[first + i] / lead);
  const re = new Float64Array(degree + zeros);
  const im = new Float64Array(degree + zeros);
  if (degree <= 0) return { re, im };

  // Start on a circle that holds every root, at angles that avoid symmetry.
  let radius = 0;
  for (let i = 1; i <= degree; i += 1) radius = Math.max(radius, Math.abs(c[i]) ** (1 / i));
  radius = 2 * radius || 1;
  for (let i = 0; i < degree; i += 1) {
    const angle = (2 * Math.PI * i) / degree + 0.4;
    re[i] = radius * Math.cos(angle);
    im[i] = radius * Math.sin(angle);
  }

  for (let iteration = 0; iteration < 1000; iteration += 1) {
    let change = 0;
    for (let i = 0; i < degree; i += 1) {
      // p(z) by Horner's rule.
      let pr = 1;
      let pi = 0;
      for (let k = 1; k <= degree; k += 1) {
        const nr = pr * re[i] - pi * im[i] + c[k];
        pi = pr * im[i] + pi * re[i];
        pr = nr;
      }
      // The product of (z_i - z_j) over the other roots.
      let qr = 1;
      let qi = 0;
      for (let j = 0; j < degree; j += 1) {
        if (j === i) continue;
        const dr = re[i] - re[j];
        const di = im[i] - im[j];
        const nr = qr * dr - qi * di;
        qi = qr * di + qi * dr;
        qr = nr;
      }
      const denominator = qr * qr + qi * qi || 1e-300;
      const stepRe = (pr * qr + pi * qi) / denominator;
      const stepIm = (pi * qr - pr * qi) / denominator;
      re[i] -= stepRe;
      im[i] -= stepIm;
      change = Math.max(change, Math.hypot(stepRe, stepIm));
    }
    if (change < 1e-14 * radius) break;
  }
  return { re, im };
}

// Pitch ------------------------------------------------------------------

/**
 * YIN's difference function, `d(tau) = sum_{j < W} (x[s + j] - x[s + j + tau])^2`
 * for `tau = 0..maxLag`, over the `window` samples from `start`.
 */
export function yinDifference(x: ArrayLike<number>, start: number, window: number, maxLag: number): Float64Array {
  // Expanded, d(tau) = e(0) + e(tau) - 2 c(tau): the energy of the window,
  // the energy of the window shifted by tau, and their cross-correlation.
  // The energies come from a running sum and the correlation from one FFT
  // product, which is what makes a whole track fast enough to redraw.
  const length = window + maxLag;
  const n = nextPowerOfTwo(length);
  const aRe = new Float64Array(n);
  const aIm = new Float64Array(n);
  const bRe = new Float64Array(n);
  const bIm = new Float64Array(n);
  for (let i = 0; i < length; i += 1) {
    const at = start + i;
    const v = at >= 0 && at < x.length ? x[at] : 0;
    bRe[i] = v;
    if (i < window) aRe[i] = v;
  }
  const squares = new Float64Array(length + 1);
  for (let i = 0; i < length; i += 1) squares[i + 1] = squares[i] + bRe[i] * bRe[i];

  fft(aRe, aIm);
  fft(bRe, bIm);
  // conj(A) B, whose inverse is c(tau) = sum_j a[j] b[j + tau].
  for (let k = 0; k < n; k += 1) {
    const re = aRe[k] * bRe[k] + aIm[k] * bIm[k];
    const im = aRe[k] * bIm[k] - aIm[k] * bRe[k];
    aRe[k] = re;
    aIm[k] = im;
  }
  fft(aRe, aIm, true);

  const out = new Float64Array(maxLag + 1);
  const e0 = squares[window];
  for (let tau = 1; tau <= maxLag; tau += 1) {
    const shifted = squares[tau + window] - squares[tau];
    out[tau] = Math.max(0, e0 + shifted - 2 * aRe[tau]);
  }
  return out;
}

/**
 * YIN's cumulative mean normalized difference, `d'(0) = 1` and
 * `d'(tau) = d(tau) / ((1 / tau) sum_{j = 1..tau} d(j))`.
 */
export function cumulativeMeanNormalized(d: ArrayLike<number>): Float64Array {
  const out = new Float64Array(d.length);
  out[0] = 1;
  let running = 0;
  for (let tau = 1; tau < d.length; tau += 1) {
    running += d[tau];
    out[tau] = running > 0 ? (d[tau] * tau) / running : 1;
  }
  return out;
}

/**
 * YIN's pick: the first dip of `dPrime` below `threshold` between `minLag`
 * and `maxLag`, walked down to the bottom of that dip, or the lowest point
 * when none dips that far; then refined by a parabola through its
 * neighbours. `lag` is in samples, fractional; `value` is `d'` there, which
 * measures how aperiodic the frame is.
 */
export function yinPick(
  dPrime: ArrayLike<number>,
  threshold: number,
  minLag = 2,
  maxLag = dPrime.length - 2,
): { index: number; lag: number; value: number; belowThreshold: boolean } {
  const lo = Math.max(1, minLag);
  const hi = Math.min(dPrime.length - 2, maxLag);
  let index = -1;
  for (let tau = lo; tau <= hi; tau += 1) {
    if (dPrime[tau] < threshold) {
      while (tau + 1 <= hi && dPrime[tau + 1] < dPrime[tau]) tau += 1;
      index = tau;
      break;
    }
  }
  const belowThreshold = index >= 0;
  if (!belowThreshold) {
    index = lo;
    for (let tau = lo; tau <= hi; tau += 1) if (dPrime[tau] < dPrime[index]) index = tau;
  }
  const a = dPrime[index - 1];
  const b = dPrime[index];
  const c = dPrime[index + 1];
  const curve = a - 2 * b + c;
  const shift = curve > 0 ? (a - c) / (2 * curve) : 0;
  return { index, lag: index + Math.max(-1, Math.min(1, shift)), value: b, belowThreshold };
}

// Dynamic time warping ---------------------------------------------------

/**
 * Step patterns, named as the dtw-python package names them:
 *
 * - `symmetric1`: `D(i, j) = c(i, j) + min(D(i-1, j), D(i, j-1), D(i-1, j-1))`.
 * - `symmetric2`: the same, with the diagonal step counting `c(i, j)` twice,
 *   so every path from corner to corner weighs `n + m` steps.
 * - `asymmetric`: `i` advances every step and `j` by 0, 1, or 2, so the path
 *   is one cell per row of the first sequence and its slope stays within 0 to 2.
 */
export type DtwStepPattern = "symmetric1" | "symmetric2" | "asymmetric";

/**
 * Dynamic time warping over an `n` by `m` local cost matrix, row-major
 * (`cost[i * m + j]`), from `(0, 0)` to `(n - 1, m - 1)`. `band` limits the
 * path to within that many columns of the straight line between the
 * corners, a Sakoe-Chiba band; cells outside it cost infinity. A tie
 * between steps goes to the diagonal, as `librosa.sequence.dtw` breaks it.
 */
export function dtw(
  cost: ArrayLike<number>,
  n: number,
  m: number,
  options: { band?: number; pattern?: DtwStepPattern } = {},
): { cumulative: Float64Array; distance: number; path: [number, number][] } {
  const band = options.band ?? Infinity;
  const pattern = options.pattern ?? "symmetric1";
  const D = new Float64Array(n * m).fill(Infinity);
  const from = new Int32Array(n * m).fill(-1);
  const slope = n > 1 ? (m - 1) / (n - 1) : 0;
  const inBand = (i: number, j: number) => Math.abs(j - i * slope) <= band + 1e-9;

  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < m; j += 1) {
      if (!inBand(i, j)) continue;
      const c = cost[i * m + j];
      if (i === 0 && j === 0) {
        D[0] = c;
        continue;
      }
      let best = Infinity;
      let at = -1;
      const consider = (pi: number, pj: number, weight: number) => {
        if (pi < 0 || pj < 0) return;
        const v = D[pi * m + pj] + weight * c;
        if (v < best) {
          best = v;
          at = pi * m + pj;
        }
      };
      if (pattern === "asymmetric") {
        consider(i - 1, j - 1, 1);
        consider(i - 1, j, 1);
        consider(i - 1, j - 2, 1);
      } else {
        consider(i - 1, j - 1, pattern === "symmetric2" ? 2 : 1);
        consider(i - 1, j, 1);
        consider(i, j - 1, 1);
      }
      D[i * m + j] = best;
      from[i * m + j] = at;
    }
  }

  const distance = D[n * m - 1];
  const path: [number, number][] = [];
  if (Number.isFinite(distance)) {
    let cell = n * m - 1;
    while (cell >= 0) {
      path.push([Math.floor(cell / m), cell % m]);
      cell = from[cell];
    }
    path.reverse();
  }
  return { cumulative: D, distance, path };
}

// The inverse DCT ---------------------------------------------------------

/**
 * The inverse of `dct2`: `idct2(dct2(x, norm), norm)` gives `x`. Matches
 * `scipy.fft.idct(c, type=2, norm=...)`. Truncating `c` to its first few
 * coefficients before inverting smooths the curve it came from, which is
 * how a few MFCCs describe a spectral envelope.
 */
export function idct2(c: ArrayLike<number>, norm: "ortho" | "backward" = "ortho", length = c.length): Float64Array {
  const n = length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    let sum = 0;
    for (let k = 0; k < c.length; k += 1) {
      const basis = Math.cos((Math.PI * k * (2 * i + 1)) / (2 * n));
      if (norm === "ortho") sum += c[k] * basis * (k === 0 ? Math.sqrt(1 / n) : Math.sqrt(2 / n));
      else sum += c[k] * basis * (k === 0 ? 1 : 2);
    }
    out[i] = norm === "ortho" ? sum : sum / (2 * n);
  }
  return out;
}
