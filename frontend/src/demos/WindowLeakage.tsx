import { useMemo, useState } from "react";

import { makeWindow, rfftMagnitude, type WindowName } from "./dsp";
import type { DemoProps } from "./index";
import {
  Choice,
  Curve,
  DemoFrame,
  Legend,
  Plot,
  Points,
  Readout,
  Readouts,
  Rule,
  Slider,
  Stems,
  Toggle,
} from "./kit";

/** The frame length. The readouts are in bins of an N-point DFT, fs / N apart. */
const N = 64;
const FLOOR_DB = -120;
/** How finely the window's own spectrum is sampled to measure its lobes. */
const FINE = 64;
const QUIET_TONE_DB = -50;
const QUIET_TONE_OFFSET = 6;

const WINDOWS: { value: WindowName; label: string }[] = [
  { value: "rectangular", label: "Rectangular" },
  { value: "hann", label: "Hann" },
  { value: "hamming", label: "Hamming" },
  { value: "blackman", label: "Blackman" },
];
const PADS = ["1", "2", "4", "8", "16"] as const;
type Pad = (typeof PADS)[number];

/** The spectrum of the windowed frame, in dB of amplitude, against N-point bins. */
function spectrumDb(frame: Float64Array, window: Float64Array, fftLength: number): [number, number][] {
  const windowed = frame.map((x, i) => x * window[i]);
  const magnitude = rfftMagnitude(windowed, fftLength);
  // A sine of amplitude 1 on a bin reads sum(w) / 2, so dividing by that
  // makes the peak read the tone's amplitude, 0 dB, when it sits on a bin.
  const coherent = window.reduce((sum, w) => sum + w, 0) / 2;
  return Array.from(magnitude, (m, k): [number, number] => [
    (k * N) / fftLength,
    Math.max(FLOOR_DB, 20 * Math.log10(Math.max(1e-12, m / coherent))),
  ]);
}

/**
 * The window's main lobe and highest side lobe, measured on a spectrum
 * sampled 64 times more finely than the DFT's bins.
 */
function lobes(window: Float64Array): { halfWidth: number; sideLobeDb: number } {
  const magnitude = rfftMagnitude(window, N * FINE);
  let null_ = 1;
  while (null_ + 1 < magnitude.length && magnitude[null_ + 1] < magnitude[null_]) null_ += 1;
  let side = 0;
  for (let k = null_; k < magnitude.length; k += 1) side = Math.max(side, magnitude[k]);
  return { halfWidth: null_ / FINE, sideLobeDb: 20 * Math.log10(side / magnitude[0]) };
}

export default function WindowLeakage({ caption }: DemoProps) {
  const [bin, setBin] = useState(10.5);
  const [windowName, setWindowName] = useState<WindowName>("hann");
  const [pad, setPad] = useState<Pad>("4");
  const [quietTone, setQuietTone] = useState(false);

  const fftLength = N * Number(pad);

  const frame = useMemo(() => {
    const quiet = 10 ** (QUIET_TONE_DB / 20);
    return Float64Array.from({ length: N }, (_, n) => {
      let x = Math.cos((2 * Math.PI * bin * n) / N);
      if (quietTone) x += quiet * Math.cos((2 * Math.PI * (bin + QUIET_TONE_OFFSET) * n) / N);
      return x;
    });
  }, [bin, quietTone]);

  const { window, spectrum, bins, rectangular, shape, peak } = useMemo(() => {
    const window = makeWindow(windowName, N);
    const spectrum = spectrumDb(frame, window, fftLength);
    const bins = spectrumDb(frame, window, N);
    const rectangular = windowName === "rectangular" ? null : spectrumDb(frame, makeWindow("rectangular", N), fftLength);
    const shape = lobes(window);
    let peak = spectrum[0];
    for (const point of spectrum) if (point[1] > peak[1]) peak = point;
    return { window, spectrum, bins, rectangular, shape, peak };
  }, [frame, windowName, fftLength]);

  const time = useMemo(() => {
    const windowed = Array.from(frame, (x, n): [number, number] => [n, x * window[n]]);
    const envelope = Array.from(window, (w, n): [number, number] => [n, w]);
    return { windowed, envelope };
  }, [frame, window]);

  const label = WINDOWS.find((w) => w.value === windowName)?.label ?? windowName;

  return (
    <DemoFrame
      title="How a window trades main-lobe width for side-lobe leakage"
      caption={caption}
      controls={
        <>
          <Slider
            label="Tone frequency"
            value={bin}
            min={4}
            max={20}
            step={0.05}
            format={(v) => `bin ${v.toFixed(2)}`}
            onChange={setBin}
          />
          <Choice label="Window" value={windowName} options={WINDOWS} onChange={setWindowName} />
          <Choice
            label="Zero-padding"
            value={pad}
            options={PADS.map((p) => ({ value: p, label: p === "1" ? "none" : `${p}x` }))}
            onChange={setPad}
          />
          <Toggle
            label={`Add a tone ${-QUIET_TONE_DB} dB quieter, ${QUIET_TONE_OFFSET} bins higher`}
            checked={quietTone}
            onChange={setQuietTone}
          />
        </>
      }
    >
      <Plot
        height={130}
        xDomain={[0, N]}
        yDomain={[-1.1, 1.1]}
        xLabel="sample"
        yTicks={2}
        xTicks={4}
        tickFormat={(value, axis) => (axis === "x" ? value.toFixed(0) : value.toFixed(1))}
      >
        {() => (
          <>
            <Curve points={time.envelope} stroke="var(--ink-4)" dashed width={1} />
            <Curve points={time.envelope.map(([n, w]) => [n, -w] as [number, number])} stroke="var(--ink-4)" dashed width={1} />
            <Stems points={time.windowed} r={1.8} />
          </>
        )}
      </Plot>

      <div className="mt-3">
        <Plot
          height={230}
          xDomain={[0, N / 2]}
          yDomain={[FLOOR_DB, 0]}
          xLabel="frequency (bins of the 64-point DFT)"
          yLabel="dB"
          xTicks={8}
          yTicks={6}
          tickFormat={(value) => value.toFixed(0)}
        >
          {() => (
            <>
              {rectangular && <Curve points={rectangular} stroke="var(--ink-4)" width={1} opacity={0.7} />}
              <Rule x={bin} label="tone" />
              <Curve points={spectrum} width={1.5} />
              <Points points={bins} r={2.6} fill="var(--series-2)" />
            </>
          )}
        </Plot>
      </div>
      <Legend
        items={[
          { label: `${label}, ${pad === "1" ? "no padding" : `zero-padded to ${fftLength}`}`, colour: "var(--demo)" },
          { label: "the 64-point DFT's own bins", colour: "var(--series-2)" },
          ...(rectangular ? [{ label: "rectangular, for comparison", colour: "var(--ink-4)" }] : []),
        ]}
      />

      <Readouts>
        <Readout label="Main lobe, null to null" value={`${(2 * shape.halfWidth).toFixed(2)} bins`} />
        <Readout label="Highest side lobe" value={`${shape.sideLobeDb.toFixed(1)} dB`} />
        <Readout
          label="Peak reads"
          value={`${peak[1].toFixed(2)} dB`}
          hint={`at bin ${peak[0].toFixed(2)}${peak[1] < -0.05 ? ", short of 0 dB: scalloping" : ""}`}
          tone={Math.abs(peak[1]) > 0.5 ? "bad" : "good"}
        />
        <Readout label="Grid spacing" value={pad === "1" ? "1 bin" : `1/${pad} bin`} hint="padding only adds points" />
      </Readouts>
    </DemoFrame>
  );
}
