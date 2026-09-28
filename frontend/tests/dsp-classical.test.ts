/**
 * Checks the classical-speech helpers in src/demos/dsp.ts: the Blackman
 * window, the complex STFT and its inverse, convolution, resonators, room
 * decay, linear prediction, polynomial roots, YIN, DTW, and the inverse DCT.
 *
 *   npm test
 *
 * The reference values are the worked examples in
 * docs/research/01-signals-and-classical-speech.md, and numbers printed by
 * numpy 2.3 and scipy 1.16, written in here because each is short.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  allPoleMagnitude,
  autocorrelation,
  blackman,
  cumulativeMeanNormalized,
  dct2,
  decayTime,
  dtw,
  fftConvolve,
  gaussian,
  idct2,
  istft,
  levinsonDurbin,
  linearChirp,
  polynomialRoots,
  preEmphasis,
  resonate,
  rfftMagnitude,
  schroederDecay,
  seededRandom,
  stft,
  stftComplex,
  yinDifference,
  yinPick,
} from "../src/demos/dsp.ts";

function close(got: ArrayLike<number>, want: ArrayLike<number>, tol: number, what: string) {
  assert.equal(got.length, want.length, `${what}: length ${got.length}, expected ${want.length}`);
  for (let i = 0; i < got.length; i += 1) {
    assert.ok(
      Math.abs(got[i] - want[i]) <= tol,
      `${what}: index ${i} is ${got[i]}, expected ${want[i]} within ${tol}`,
    );
  }
}

test("blackman matches numpy.blackman and scipy's periodic window", () => {
  close(blackman(5, false), [0, 0.34, 1, 0.34, 0], 1e-12, "symmetric, 5");
  close(
    blackman(8),
    [0, 0.0664466094, 0.34, 0.773553391, 1, 0.773553391, 0.34, 0.0664466094],
    1e-9,
    "periodic, 8",
  );
});

test("seededRandom repeats itself and gaussian has unit variance", () => {
  const a = seededRandom(7);
  const b = seededRandom(7);
  for (let i = 0; i < 5; i += 1) assert.equal(a(), b());
  const random = seededRandom(1);
  let sum = 0;
  let squares = 0;
  const n = 20000;
  for (let i = 0; i < n; i += 1) {
    const v = gaussian(random);
    sum += v;
    squares += v * v;
  }
  assert.ok(Math.abs(sum / n) < 0.03, `mean ${sum / n}`);
  assert.ok(Math.abs(squares / n - 1) < 0.05, `variance ${squares / n}`);
});

test("stftComplex agrees with stft, and istft gives the signal back", () => {
  const signal = linearChirp(200, 3000, 8000, 0.2);
  const complex = stftComplex(signal, { nFft: 256, hop: 64 });
  const magnitude = stft(signal, { nFft: 256, hop: 64 });
  const joined = complex.re.map((re, i) => Math.hypot(re, complex.im[i]));
  close(joined, magnitude.data, 1e-4, "magnitudes");
  const back = istft(complex, { length: signal.length });
  close(back, signal, 1e-9, "round trip");
  // Without a length, a centred transform gives hop * (frames - 1) samples.
  assert.equal(istft(complex).length, 64 * (complex.frames - 1));
});

test("fftConvolve matches scipy.signal.fftconvolve", () => {
  const a = [2.040919121385, -2.555665031314, 0.418098846726, -0.567769606128, -0.45264929211, -0.21559716309, -2.019986129147];
  const b = [-0.231932377644, -0.865213076275, 3.322999516645, 0.225786613228, -0.352630794342];
  close(
    fftConvolve(a, b),
    [
      -0.473355224402356, -1.17308844426722, 8.8961973979345, -8.26172188213285, 0.688842096744571,
      -0.44944876491939, -1.1247450133338, 1.12930004066118, -6.6014738046262, -0.380059627989398,
      0.7123093132802,
    ],
    1e-9,
    "convolution",
  );
});

test("preEmphasis matches scipy.signal.lfilter([1, -0.97], [1], x)", () => {
  close(preEmphasis([1, 2, -1, 0.5]), [1, 1.03, -2.94, 1.47], 1e-12, "pre-emphasis");
});

test("resonate passes 0 Hz at unit gain and peaks at its frequency", () => {
  const rate = 16000;
  const impulse = new Float64Array(4096);
  impulse[0] = 1;
  const response = resonate(impulse, 1000, 80, rate);
  const dc = response.reduce((sum, v) => sum + v, 0);
  assert.ok(Math.abs(dc - 1) < 1e-6, `gain at 0 Hz is ${dc}`);
  const spectrum = rfftMagnitude(response);
  const peak = spectrum.indexOf(Math.max(...spectrum));
  assert.ok(Math.abs((peak * rate) / 4096 - 1000) < 8, `peak at ${(peak * rate) / 4096} Hz`);
});

test("schroederDecay and decayTime recover an exponential decay's RT60", () => {
  const rate = 8000;
  const rt60 = 0.5;
  const ir = Float64Array.from({ length: rate }, (_, n) => Math.exp((-6.908 * n) / (rate * rt60)));
  const edc = schroederDecay(ir);
  assert.equal(edc[0], 0);
  const fit = decayTime(edc, rate);
  assert.ok(fit && Math.abs(fit.rt60 - rt60) < 0.005, `RT60 ${fit?.rt60}`);
  assert.equal(decayTime(edc, rate, -5, -500), null);
});

test("levinsonDurbin solves the worked example and matches solve_toeplitz", () => {
  const worked = levinsonDurbin([1, 0.5, 0.1], 2);
  close(worked.alpha, [0.6, -0.2], 1e-12, "alpha");
  close(worked.reflection, [0.5, -0.2], 1e-12, "reflection");
  assert.ok(Math.abs(worked.error - 0.72) < 1e-12);
  const fourth = levinsonDurbin([1, 0.6, 0.2, -0.1, -0.25], 4);
  close(fourth.alpha, [0.6928571428571427, -0.1366071428571428, -0.10089285714285716, -0.09285714285714286], 1e-12, "order 4");
});

test("autocorrelation and allPoleMagnitude", () => {
  close(autocorrelation([1, 2, 3], 2), [14, 8, 3], 1e-12, "autocorrelation");
  close(allPoleMagnitude([], 2, 3), [2, 2, 2], 1e-12, "order 0 is flat");
  // One pole at 0.5: |1 / (1 - 0.5 e^{-jw})| is 2 at 0 Hz and 2/3 at Nyquist.
  close(allPoleMagnitude([0.5], 1, 2), [2, 2 / 3], 1e-12, "one pole");
});

test("polynomialRoots matches numpy.roots", () => {
  const sorted = (roots: { re: Float64Array; im: Float64Array }) =>
    Array.from(roots.re, (re, i) => [re, roots.im[i]] as const).sort(
      (a, b) => Math.round((a[0] - b[0]) * 1e6) || a[1] - b[1],
    );
  const cubic = sorted(polynomialRoots([1, -6, 11, -6]));
  close(cubic.map((r) => r[0]), [1, 2, 3], 1e-9, "real roots");
  close(cubic.map((r) => r[1]), [0, 0, 0], 1e-9, "no imaginary part");
  const quartic = sorted(polynomialRoots([1, -1.2, 0.8, -0.3, 0.1]));
  close(quartic.map((r) => r[0]), [0.06212435, 0.06212435, 0.53787565, 0.53787565], 1e-7, "real parts");
  close(quartic.map((r) => r[1]), [-0.47372958, 0.47372958, -0.38568113, 0.38568113], 1e-7, "imaginary parts");
  const withZero = polynomialRoots([2, -2, 0]);
  close(Array.from(withZero.re).sort(), [0, 1], 1e-12, "a trailing zero is a root at 0");
});

test("YIN's difference function and CMNDF match the worked example", () => {
  const x = [0, 1, 0, -1, 0, 1, 0, -1, 0, 1];
  const d = yinDifference(x, 0, 4, 5);
  close(d, [0, 4, 8, 4, 0, 4], 1e-12, "d");
  const dPrime = cumulativeMeanNormalized(d);
  close(dPrime, [1, 1, 4 / 3, 0.75, 0, 1], 1e-12, "d'");
  const pick = yinPick(dPrime, 0.1, 1, 4);
  assert.equal(pick.index, 4);
  assert.ok(pick.belowThreshold);
});

test("dtw reproduces the worked 3 by 3 example", () => {
  const a = [0, 1, 2];
  const b = [0, 2, 2];
  const cost = a.flatMap((x) => b.map((y) => Math.abs(x - y)));
  const result = dtw(cost, 3, 3);
  close(result.cumulative, [0, 2, 4, 1, 1, 2, 3, 1, 1], 1e-12, "D");
  assert.equal(result.distance, 1);
  // Two paths cost 1: the worked example's (0,0) (1,0) (2,1) (2,2), and the
  // diagonal. A tie goes to the diagonal step, as librosa breaks it.
  assert.deepEqual(result.path, [
    [0, 0],
    [1, 1],
    [2, 2],
  ]);
  const repeat = dtw([1, 2, 3].flatMap((x) => [1, 1, 2, 3].map((y) => Math.abs(x - y))), 3, 4);
  assert.equal(repeat.distance, 0);
});

test("a wide Sakoe-Chiba band changes nothing and a narrow one only raises the cost", () => {
  const a = Array.from({ length: 12 }, (_, i) => Math.sin(i / 2));
  const b = Array.from({ length: 18 }, (_, j) => Math.sin(((j / 17) * 11) ** 1.3 / 2 / 11 ** 0.3));
  const cost = a.flatMap((x) => b.map((y) => Math.abs(x - y)));
  const free = dtw(cost, 12, 18);
  const wide = dtw(cost, 12, 18, { band: 100 });
  const narrow = dtw(cost, 12, 18, { band: 1 });
  assert.equal(wide.distance, free.distance);
  assert.ok(narrow.distance >= free.distance);
  for (const pattern of ["symmetric2", "asymmetric"] as const) {
    const result = dtw(cost, 12, 18, { pattern });
    assert.deepEqual(result.path[0], [0, 0]);
    assert.deepEqual(result.path[result.path.length - 1], [11, 17]);
  }
});

test("idct2 matches scipy.fft.idct and inverts dct2", () => {
  const x = [-0.2812874181513504, -0.6680463461089501, -1.0551505512051214, -0.39080097723465473, 0.48194538850678587, -0.2385536065733667];
  close(
    idct2(x),
    [-1.0710298896885087, -0.4088822531927625, 0.4785479999339656, 0.6251836358816821, -0.3772905615639858, 0.06446042309391359],
    1e-12,
    "ortho",
  );
  close(
    idct2(x, "backward"),
    [-0.299470275600631, -0.10832471751358427, 0.14785433025128278, 0.1901843921948858, -0.09920498168056935, 0.028317543272940855],
    1e-12,
    "backward",
  );
  close(idct2(dct2(x)), x, 1e-12, "round trip");
});
