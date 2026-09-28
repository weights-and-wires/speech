/**
 * A small source-filter speech synthesizer for the demos that need a voice
 * with known answers: a vowel with known formants, a voice with a known pitch
 * contour, or a word said at two speeds.
 *
 * The source is a pulse train at F0, shaped by two one-pole low-pass filters
 * (the glottal pulse, about -12 dB per octave) and a first difference (the
 * lips, about +6 dB per octave). The filter is a cascade of Klatt resonators,
 * one per formant. Everything is seeded, so a printed figure never changes.
 */

import { seededRandom, gaussian } from "./dsp";

/** A formant: a centre frequency and a bandwidth, both in Hz. */
export type Formant = [hz: number, bandwidth: number];

/**
 * Peterson and Barney's (1952) average formants for men, in Hz, for ten
 * American English vowels, with the word each was said in.
 */
export const VOWELS = [
  { ipa: "i", word: "heed", f1: 270, f2: 2290, f3: 3010 },
  { ipa: "ɪ", word: "hid", f1: 390, f2: 1990, f3: 2550 },
  { ipa: "ɛ", word: "head", f1: 530, f2: 1840, f3: 2480 },
  { ipa: "æ", word: "had", f1: 660, f2: 1720, f3: 2410 },
  { ipa: "ɑ", word: "hod", f1: 730, f2: 1090, f3: 2440 },
  { ipa: "ɔ", word: "hawed", f1: 570, f2: 840, f3: 2410 },
  { ipa: "ʊ", word: "hood", f1: 440, f2: 1020, f3: 2240 },
  { ipa: "u", word: "who'd", f1: 300, f2: 870, f3: 2240 },
  { ipa: "ʌ", word: "hud", f1: 640, f2: 1190, f3: 2390 },
  { ipa: "ɝ", word: "heard", f1: 490, f2: 1350, f3: 1690 },
] as const;

export type Vowel = (typeof VOWELS)[number];

/** Bandwidths that sound like a man's voice, formant by formant. */
export const BANDWIDTHS = [60, 90, 150, 200, 250];

/** The formants of a vowel, with a fourth at 3.5 kHz for a fuller sound. */
export function vowelFormants(f1: number, f2: number, f3 = 2500, f4 = 3500): Formant[] {
  return [
    [f1, BANDWIDTHS[0]],
    [f2, BANDWIDTHS[1]],
    [f3, BANDWIDTHS[2]],
    [f4, BANDWIDTHS[3]],
  ];
}

export type VoiceOptions = {
  sampleRate: number;
  seconds: number;
  /** F0 in Hz at time `t` seconds; 0 or less means unvoiced there. */
  f0: number | ((t: number) => number);
  /** The formants, fixed or as a function of time. */
  formants: Formant[] | ((t: number) => Formant[]);
  /** Breath noise mixed into the source, relative to the pulses. */
  noise?: number | ((t: number) => number);
  /**
   * Alternate pulses louder and softer by this fraction, as a creaky voice
   * does, so the waveform repeats every two periods rather than every one.
   */
  alternate?: (t: number) => number;
  seed?: number;
  /** The peak of the result. */
  peak?: number;
};

/**
 * A voice from a pulse train and a cascade of resonators. The pulses sit at
 * fractional sample positions, split between the two nearest samples, so a
 * pitch tracker can find F0 to a fraction of a sample.
 */
export function synthesizeVoice(options: VoiceOptions): Float32Array {
  const { sampleRate } = options;
  const n = Math.round(options.seconds * sampleRate);
  const f0 = typeof options.f0 === "number" ? () => options.f0 as number : options.f0;
  const formantsAt =
    typeof options.formants === "function" ? options.formants : () => options.formants as Formant[];
  const noiseAt = typeof options.noise === "function" ? options.noise : () => (options.noise as number) ?? 0;
  const random = seededRandom(options.seed ?? 1);

  // The source: one pulse each time the phase passes a whole cycle.
  const source = new Float64Array(n + 2);
  let phase = 0;
  let pulses = 0;
  for (let i = 0; i < n; i += 1) {
    const hz = f0(i / sampleRate);
    if (!(hz > 0)) {
      phase = 0;
      continue;
    }
    const step = hz / sampleRate;
    const next = phase + step;
    if (next >= 1) {
      // Where between i and i + 1 the cycle ended.
      const fraction = (1 - phase) / step;
      const swing = options.alternate?.(i / sampleRate) ?? 0;
      const gain = 1 + (pulses % 2 ? -swing : swing);
      source[i] += gain * (1 - fraction);
      source[i + 1] += gain * fraction;
      pulses += 1;
    }
    phase = next % 1;
  }

  // The glottal pulse shape, then the lips.
  const pole = Math.exp((-2 * Math.PI * 120) / sampleRate);
  let a = 0;
  let b = 0;
  let previous = 0;
  let energy = 0;
  let voiced = 0;
  const excitation = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    a = (1 - pole) * source[i] + pole * a;
    b = (1 - pole) * a + pole * b;
    excitation[i] = b - previous;
    previous = b;
    if (f0(i / sampleRate) > 0) {
      energy += excitation[i] * excitation[i];
      voiced += 1;
    }
  }
  // Breath noise, scaled to the voiced source, so `noise: 1` is as loud as it.
  const rms = Math.sqrt(energy / Math.max(1, voiced)) || 1e-4;
  for (let i = 0; i < n; i += 1) {
    const breath = noiseAt(i / sampleRate);
    if (breath > 0) excitation[i] += breath * rms * gaussian(random);
  }

  // The vocal tract: resonators whose coefficients follow the formants,
  // updated every 32 samples.
  const count = formantsAt(0).length;
  const state = Array.from({ length: count }, () => [0, 0]);
  const coefficients = Array.from({ length: count }, () => [0, 0, 0]);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    if (i % 32 === 0) {
      const formants = formantsAt(i / sampleRate);
      for (let f = 0; f < count; f += 1) {
        const [hz, bandwidth] = formants[f];
        const c = -Math.exp((-2 * Math.PI * bandwidth) / sampleRate);
        const bb = 2 * Math.exp((-Math.PI * bandwidth) / sampleRate) * Math.cos((2 * Math.PI * hz) / sampleRate);
        coefficients[f][0] = 1 - bb - c;
        coefficients[f][1] = bb;
        coefficients[f][2] = c;
      }
    }
    let x = excitation[i];
    for (let f = 0; f < count; f += 1) {
      const [ga, gb, gc] = coefficients[f];
      const s = state[f];
      const y = ga * x + gb * s[0] + gc * s[1];
      s[1] = s[0];
      s[0] = y;
      x = y;
    }
    out[i] = x;
  }

  let max = 0;
  for (let i = 0; i < n; i += 1) max = Math.max(max, Math.abs(out[i]));
  const scale = max > 0 ? (options.peak ?? 0.8) / max : 0;
  for (let i = 0; i < n; i += 1) out[i] *= scale;
  return out;
}

/** A smooth step from 0 to 1 as `t` goes from `a` to `b`. */
export function smoothstep(a: number, b: number, t: number): number {
  const x = Math.min(1, Math.max(0, (t - a) / (b - a)));
  return x * x * (3 - 2 * x);
}
