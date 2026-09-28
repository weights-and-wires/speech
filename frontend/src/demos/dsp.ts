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
 * - `dct2` matches `scipy.fft.dct(type=2)`, with `norm="ortho"` by default.
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

export type WindowName = "rectangular" | "hann" | "hamming";

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

export function makeWindow(name: WindowName, n: number, periodic = true): Float64Array {
  if (name === "hann") return hann(n, periodic);
  if (name === "hamming") return hamming(n, periodic);
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
