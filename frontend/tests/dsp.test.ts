/**
 * Checks src/demos/dsp.ts against numpy, scipy, and librosa.
 *
 *   npm test
 *
 * The reference values in dsp-reference.json come from make_dsp_reference.py.
 * Node strips the TypeScript types itself, so this needs no build step.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  applyFilterbank,
  dct2,
  fft,
  hamming,
  hann,
  hzToMel,
  linearChirp,
  melFilterbank,
  melToHz,
  rfftMagnitude,
  stft,
  toDb,
} from "../src/demos/dsp.ts";

const ref = JSON.parse(readFileSync(new URL("./dsp-reference.json", import.meta.url), "utf8"));

function close(got: ArrayLike<number>, want: ArrayLike<number>, tol: number, what: string) {
  assert.equal(got.length, want.length, `${what}: length ${got.length}, expected ${want.length}`);
  let worst = 0;
  let at = -1;
  for (let i = 0; i < got.length; i += 1) {
    const diff = Math.abs(got[i] - want[i]);
    if (diff > worst) {
      worst = diff;
      at = i;
    }
  }
  assert.ok(worst <= tol, `${what}: largest difference ${worst.toExponential(2)} at index ${at}, above ${tol}`);
}

test("fft matches numpy.fft.fft", () => {
  const re = Float64Array.from(ref.fft.input);
  const im = new Float64Array(re.length);
  fft(re, im);
  close(re, ref.fft.re, 1e-9, "real part");
  close(im, ref.fft.im, 1e-9, "imaginary part");
});

test("the inverse fft undoes the forward one", () => {
  const input = Float64Array.from({ length: 64 }, (_, i) => Math.sin(i) + 0.1 * i);
  const re = input.slice();
  const im = new Float64Array(64);
  fft(re, im);
  fft(re, im, true);
  close(re, input, 1e-12, "round trip");
  close(im, new Float64Array(64), 1e-12, "imaginary residue");
});

test("fft rejects a length that is not a power of two", () => {
  assert.throws(() => fft(new Float64Array(12), new Float64Array(12)), /power of two/);
});

test("Parseval: the energy in time equals the energy in frequency over N", () => {
  const x = Float64Array.from({ length: 128 }, (_, i) => Math.cos(0.3 * i) * Math.exp(-i / 60));
  const re = x.slice();
  const im = new Float64Array(128);
  fft(re, im);
  const time = x.reduce((sum, v) => sum + v * v, 0);
  let freq = 0;
  for (let k = 0; k < 128; k += 1) freq += re[k] ** 2 + im[k] ** 2;
  assert.ok(Math.abs(time - freq / 128) < 1e-9);
});

test("a pure tone's rfft peaks at its bin", () => {
  const n = 256;
  const bin = 19;
  const x = Float64Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * bin * i) / n));
  const magnitude = rfftMagnitude(x);
  assert.equal(magnitude.length, n / 2 + 1);
  const peak = magnitude.indexOf(Math.max(...magnitude));
  assert.equal(peak, bin);
  assert.ok(Math.abs(magnitude[bin] - n / 2) < 1e-9);
});

test("hann and hamming match scipy.signal.get_window", () => {
  close(hann(16), ref.hann16, 1e-12, "periodic Hann");
  close(hamming(16), ref.hamming16, 1e-12, "periodic Hamming");
  close(hann(16, false), ref.hann16_symmetric, 1e-12, "symmetric Hann");
});

test("hzToMel matches librosa for both formulas, and melToHz inverts it", () => {
  close(ref.mel.hz.map((hz: number) => hzToMel(hz)), ref.mel.slaney, 1e-9, "Slaney");
  close(ref.mel.hz.map((hz: number) => hzToMel(hz, "htk")), ref.mel.htk, 1e-7, "HTK");
  for (const hz of ref.mel.hz as number[]) {
    assert.ok(Math.abs(melToHz(hzToMel(hz)) - hz) < 1e-9, `Slaney round trip at ${hz} Hz`);
    assert.ok(Math.abs(melToHz(hzToMel(hz, "htk"), "htk") - hz) < 1e-9, `HTK round trip at ${hz} Hz`);
  }
});

test("melFilterbank matches librosa.filters.mel", () => {
  const { sample_rate, n_fft, n_mels, fmin, fmax } = ref.filterbank;
  const flat = (bank: Float64Array[]) => bank.flatMap((row) => Array.from(row));
  const slaney = melFilterbank({ sampleRate: sample_rate, nFft: n_fft, nMels: n_mels, fmin, fmax });
  assert.equal(slaney.length, n_mels);
  assert.equal(slaney[0].length, n_fft / 2 + 1);
  close(flat(slaney), ref.filterbank.slaney, 1e-7, "Slaney, area-normalised");
  const htk = melFilterbank({
    sampleRate: sample_rate,
    nFft: n_fft,
    nMels: n_mels,
    fmin,
    fmax,
    formula: "htk",
    norm: null,
  });
  close(flat(htk), ref.filterbank.htk_unnormalised, 1e-7, "HTK, peak 1");
});

test("stft matches librosa.stft with centred reflect padding", () => {
  const { n_fft, hop, shape } = ref.stft;
  const spec = stft(ref.stft.signal, { nFft: n_fft, hop, window: "hann" });
  assert.deepEqual([spec.bins, spec.frames], shape);
  close(spec.data, ref.stft.magnitude, 1e-5, "magnitudes");
});

test("a chirp's spectrogram peak rises over time", () => {
  const sr = 8000;
  const signal = linearChirp(200, 3000, sr, 1);
  const spec = stft(signal, { nFft: 256, hop: 128 });
  const peakBin = (t: number) => {
    let best = 0;
    for (let k = 1; k < spec.bins; k += 1) {
      if (spec.data[k * spec.frames + t] > spec.data[best * spec.frames + t]) best = k;
    }
    return best;
  };
  const early = peakBin(5);
  const late = peakBin(spec.frames - 6);
  assert.ok(late > early + 40, `peak bin went from ${early} to ${late}`);
});

test("applyFilterbank sums the weighted bins", () => {
  const spec = stft(linearChirp(100, 3000, 8000, 0.25), { nFft: 64, hop: 32 });
  const bank = melFilterbank({ sampleRate: 8000, nFft: 64, nMels: 8 });
  const mel = applyFilterbank(bank, spec);
  assert.equal(mel.bins, 8);
  assert.equal(mel.frames, spec.frames);
  let want = 0;
  for (let k = 0; k < spec.bins; k += 1) want += bank[3][k] * spec.data[k * spec.frames + 2];
  assert.ok(Math.abs(mel.data[3 * mel.frames + 2] - want) < 1e-5);
});

test("dct2 matches scipy.fft.dct, orthonormal and unscaled", () => {
  close(dct2(ref.dct.input), ref.dct.ortho, 1e-9, "norm=ortho");
  close(dct2(ref.dct.input, "backward"), ref.dct.backward, 1e-9, "norm=backward");
});

test("toDb is 0 at the peak and floored below", () => {
  const db = toDb([1, 0.1, 0.01, 0], { floorDb: -30 });
  close(db, [0, -20, -30, -30], 1e-5, "decibels");
  close(toDb([4, 1], { power: true }), [0, -6.0206], 1e-4, "power decibels");
});
