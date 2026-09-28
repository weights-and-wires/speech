/**
 * Checks `siSdr` in src/demos/dsp.ts against its closed form. It needs no
 * reference file: the estimate is the reference plus noise made orthogonal
 * to it, so the answer is the ratio of the two energies.
 *
 *   npm test
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { siSdr } from "../src/demos/dsp.ts";

const mean = (x: Float64Array) => x.reduce((s, v) => s + v, 0) / x.length;
const dot = (x: Float64Array, y: Float64Array) => x.reduce((s, v, i) => s + v * y[i], 0);

function orthogonalCase() {
  const reference = Float64Array.from({ length: 512 }, (_, i) => Math.sin(0.07 * i) + 0.3 * Math.sin(0.31 * i) + 0.2);
  const raw = Float64Array.from({ length: 512 }, (_, i) => Math.cos(1.3 * i));
  const r0 = reference.map((v) => v - mean(reference));
  const n0 = raw.map((v) => v - mean(raw));
  const noise = n0.map((v, i) => v - (dot(n0, r0) / dot(r0, r0)) * r0[i]);
  const estimate = reference.map((v, i) => v + 0.1 * noise[i]);
  const want = 10 * Math.log10(dot(r0, r0) / (0.01 * dot(noise, noise)));
  return { reference, estimate, want };
}

test("siSdr matches the closed form for orthogonal noise", () => {
  const { reference, estimate, want } = orthogonalCase();
  assert.ok(Math.abs(siSdr(estimate, reference) - want) < 1e-6);
});

test("siSdr ignores the estimate's scale, sign, and offset", () => {
  const { reference, estimate, want } = orthogonalCase();
  assert.ok(Math.abs(siSdr(estimate.map((v) => 3 * v), reference) - want) < 1e-6);
  assert.ok(Math.abs(siSdr(estimate.map((v) => -0.5 * v), reference) - want) < 1e-6);
  assert.ok(Math.abs(siSdr(estimate.map((v) => v + 4), reference) - want) < 1e-6);
});

test("siSdr is very high for a copy and depends only on the correlation", () => {
  const { reference, estimate } = orthogonalCase();
  assert.ok(siSdr(reference, reference) > 80);
  const other = estimate.map((v, i) => v + 0.5 * Math.sin(0.9 * i));
  const a = other.map((v) => v - mean(other));
  const b = reference.map((v) => v - mean(reference));
  const rho2 = dot(a, b) ** 2 / (dot(a, a) * dot(b, b));
  const want = 10 * Math.log10(rho2 / (1 - rho2));
  assert.ok(Math.abs(siSdr(other, reference) - want) < 1e-6);
  // So swapping the arguments leaves the value alone, though not the projection.
  assert.ok(Math.abs(siSdr(reference, other) - want) < 1e-6);
});
