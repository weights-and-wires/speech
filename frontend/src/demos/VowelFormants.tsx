import { useContext, useMemo, useRef, useState } from "react";

import {
  allPoleMagnitude,
  autocorrelation,
  fadeEdges,
  hamming,
  levinsonDurbin,
  polynomialRoots,
  preEmphasis,
  rfftMagnitude,
} from "./dsp";
import type { DemoProps } from "./index";
import {
  Action,
  Actions,
  Curve,
  DemoFrame,
  Legend,
  Plot,
  Points,
  PrintedFigure,
  Readout,
  Readouts,
  Rule,
  Slider,
  useAudioPlayer,
} from "./kit";
import { VOWELS, synthesizeVoice, vowelFormants } from "./synth";

const RATE = 16000;
const SECONDS = 0.9;
/** A 32 ms analysis frame from the middle of the vowel. */
const FRAME = 512;
const N_FFT = 1024;
const F3 = 2500;
const F4 = 3500;
const F1_RANGE: [number, number] = [1000, 200];
const F2_RANGE: [number, number] = [2600, 600];
const FLOOR_DB = -80;

type Pole = { hz: number; bandwidth: number; re: number; im: number; formant: boolean };

/**
 * The poles of the LPC filter `1 / A(z)`: each complex pair is a resonance,
 * at the pole's angle and with a bandwidth set by its distance from the unit
 * circle. Narrow ones above 90 Hz count as formants.
 */
function poles(alpha: Float64Array): Pole[] {
  const roots = polynomialRoots([1, ...Array.from(alpha, (a) => -a)]);
  const out: Pole[] = [];
  for (let i = 0; i < roots.re.length; i += 1) {
    const re = roots.re[i];
    const im = roots.im[i];
    const hz = (Math.atan2(im, re) * RATE) / (2 * Math.PI);
    const bandwidth = (-RATE / Math.PI) * Math.log(Math.hypot(re, im));
    out.push({ hz, bandwidth, re, im, formant: im > 1e-9 && hz > 90 && hz < RATE / 2 - 90 && bandwidth < 400 });
  }
  return out;
}

export default function VowelFormants({ caption }: DemoProps) {
  const [f1, setF1] = useState(730);
  const [f2, setF2] = useState(1090);
  const [f0, setF0] = useState(120);
  const [order, setOrder] = useState(18);
  const { play, stop, playing } = useAudioPlayer();
  const dragging = useRef(false);
  // A printed page is shorter than a screen, so the chart and the z-plane shrink.
  const printed = useContext(PrintedFigure) !== null;

  const vowel = useMemo(
    () => synthesizeVoice({ sampleRate: RATE, seconds: SECONDS, f0, formants: vowelFormants(f1, f2, F3, F4) }),
    [f0, f1, f2],
  );

  const analysis = useMemo(() => {
    const start = Math.floor((vowel.length - FRAME) / 2);
    const window = hamming(FRAME);
    const frame = preEmphasis(vowel.subarray(start - 1, start + FRAME)).subarray(1).map((x, i) => x * window[i]);
    const power = rfftMagnitude(frame, N_FFT).map((m) => m * m);
    const ref = Math.max(...power);
    const { alpha, error } = levinsonDurbin(autocorrelation(frame, order), order);
    // E / |A|^2 has the same area as |X|^2, so the two share one scale.
    const model = allPoleMagnitude(alpha, Math.sqrt(Math.max(error, 1e-20)), N_FFT / 2 + 1);
    const hz = (k: number) => (k * RATE) / N_FFT;
    const db = (p: number) => Math.max(FLOOR_DB, 10 * Math.log10(Math.max(1e-30, p / ref)));
    const spectrum = Array.from(power, (p, k): [number, number] => [hz(k), db(p)]);
    const envelope = Array.from(model, (m, k): [number, number] => [hz(k), db(m * m)]);
    const all = poles(alpha);
    const formants = all.filter((p) => p.formant).sort((a, b) => a.hz - b.hz);
    return { spectrum, envelope, all, formants };
  }, [vowel, order]);

  function playVowel() {
    play("vowel", fadeEdges(Float32Array.from(vowel), RATE, 0.04), RATE, 0.5);
  }

  function onPad({ x, y, type }: { x: number; y: number; type: "down" | "move" | "up" }) {
    if (type === "down") dragging.current = true;
    if (!dragging.current) return;
    setF2(Math.round(Math.min(F2_RANGE[0], Math.max(F2_RANGE[1], x)) / 10) * 10);
    setF1(Math.round(Math.min(F1_RANGE[0], Math.max(F1_RANGE[1], y)) / 10) * 10);
    if (type === "up") {
      dragging.current = false;
      if (playing) window.setTimeout(playVowel, 0);
    }
  }

  const nearest = VOWELS.reduce((best, v) =>
    Math.hypot(v.f1 - f1, (v.f2 - f2) / 2) < Math.hypot(best.f1 - f1, (best.f2 - f2) / 2) ? v : best,
  );
  const estimate = (i: number) => analysis.formants[i]?.hz;
  const off = (truth: number, got: number | undefined) => got !== undefined && Math.abs(got - truth) / truth < 0.1;
  const circle = Array.from({ length: 121 }, (_, i): [number, number] => [
    Math.cos((2 * Math.PI * i) / 120),
    Math.sin((2 * Math.PI * i) / 120),
  ]);

  return (
    <DemoFrame
      title="Vowels as a source and a filter"
      caption={caption}
      controls={
        <>
          <Slider label="Pitch of the source, F0" value={f0} min={70} max={300} step={5} format={(v) => `${v} Hz`} onChange={setF0} />
          <Slider label="LPC order" value={order} min={2} max={32} onChange={setOrder} />
          <Actions>
            <Action onClick={playing ? stop : playVowel}>{playing ? "Stop" : "Play the vowel"}</Action>
          </Actions>
          <p className="text-[12.5px] leading-snug text-ink-3">
            Drag on the chart to move F1 and F2. F3 stays at {F3} Hz and F4 at {F4} Hz.
          </p>
        </>
      }
    >
      <Plot
        width={460}
        height={printed ? 190 : 300}
        xDomain={F2_RANGE}
        yDomain={F1_RANGE}
        xLabel="F2 (Hz), falling to the right"
        yLabel="F1 (Hz), rising downward"
        xTicks={4}
        yTicks={4}
        tickFormat={(v) => v.toFixed(0)}
        onPointer={onPad}
      >
        {({ sx, sy }) => (
          <>
            {VOWELS.map((v) => (
              <g key={v.ipa}>
                <circle cx={sx(v.f2)} cy={sy(v.f1)} r={2.5} fill="var(--ink-4)" />
                <text
                  x={sx(v.f2) + 5}
                  y={sy(v.f1) - 4}
                  fontSize={13}
                  fontFamily="var(--font-serif)"
                  fill={v === nearest ? "var(--ink)" : "var(--ink-3)"}
                >
                  {v.ipa}
                </text>
              </g>
            ))}
            <circle cx={sx(f2)} cy={sy(f1)} r={9} fill="color-mix(in oklab, var(--demo) 18%, transparent)" stroke="var(--demo)" strokeWidth={1.5} />
            <circle cx={sx(f2)} cy={sy(f1)} r={3} fill="var(--demo)" />
          </>
        )}
      </Plot>
      <p className="mt-1 text-[12px] leading-snug text-ink-3">
        Peterson and Barney's average formants for men. Closest: /{nearest.ipa}/ as in “{nearest.word}”.
      </p>

      <div
        className={`mt-4 grid gap-4 sm:items-start ${printed ? "sm:grid-cols-[minmax(0,1fr)_minmax(0,10rem)]" : "sm:grid-cols-[minmax(0,1fr)_minmax(0,15rem)]"}`}
      >
        <div className="min-w-0">
          <Plot
            width={420}
            height={250}
            xDomain={[0, 4500]}
            yDomain={[FLOOR_DB, 20]}
            xLabel="frequency (Hz)"
            yLabel="dB"
            xTicks={9}
            yTicks={5}
            tickFormat={(v, axis) => (axis === "x" ? (v >= 1000 ? `${v / 1000}k` : v.toFixed(0)) : v.toFixed(0))}
          >
            {() => (
              <>
                {[f1, f2, F3].map((hz) => (
                  <Rule key={hz} x={hz} colour="var(--ink-4)" />
                ))}
                <Curve points={analysis.spectrum.filter(([hz]) => hz <= 4500)} stroke="var(--ink-3)" width={0.9} />
                <Curve points={analysis.envelope.filter(([hz]) => hz <= 4500)} width={2} />
                {analysis.formants
                  .filter((p) => p.hz <= 4500)
                  .slice(0, 4)
                  .map((p, i) => (
                    <Rule key={i} x={p.hz} label={`F${i + 1}`} colour="var(--bad)" dashed={false} />
                  ))}
              </>
            )}
          </Plot>
        </div>
        <div className={`mx-auto w-full ${printed ? "max-w-[10rem]" : "max-w-[15rem]"}`}>
          <Plot width={258} height={242} xDomain={[-1.1, 1.1]} yDomain={[-1.1, 1.1]} xTicks={2} yTicks={2} xLabel="z-plane" tickFormat={(v) => v.toFixed(0)}>
            {() => (
              <>
                <Curve points={circle} stroke="var(--rule-strong)" width={1} />
                <Points points={analysis.all.filter((p) => !p.formant).map((p) => [p.re, p.im])} r={2.6} fill="var(--ink-4)" />
                <Points points={analysis.all.filter((p) => p.formant).map((p) => [p.re, p.im])} r={3.2} fill="var(--bad)" />
              </>
            )}
          </Plot>
          <p className="mt-1 text-[12px] leading-snug text-ink-3">
            {printed
              ? `The ${order} poles.`
              : `The ${order} poles. A pole's angle is its frequency; its closeness to the circle, its sharpness.`}
          </p>
        </div>
      </div>

      <Legend
        items={[
          { label: "spectrum of a pre-emphasized 32 ms frame", colour: "var(--ink-3)" },
          { label: `LPC envelope, order ${order}`, colour: "var(--demo)" },
          { label: "formants from the poles", colour: "var(--bad)" },
          { label: "the formants the vowel was made with", colour: "var(--ink-4)", dashed: true },
        ]}
      />
      <Readouts>
        <Readout label="F1" value={`${f1} / ${estimate(0)?.toFixed(0) ?? "none"}`} hint="set / found" tone={off(f1, estimate(0)) ? "good" : "bad"} />
        <Readout label="F2" value={`${f2} / ${estimate(1)?.toFixed(0) ?? "none"}`} hint="set / found" tone={off(f2, estimate(1)) ? "good" : "bad"} />
        <Readout label="F3" value={`${F3} / ${estimate(2)?.toFixed(0) ?? "none"}`} hint="set / found" tone={off(F3, estimate(2)) ? "good" : "bad"} />
        <Readout label="Harmonics" value={`${f0} Hz`} hint="apart: the source" />
      </Readouts>
    </DemoFrame>
  );
}
