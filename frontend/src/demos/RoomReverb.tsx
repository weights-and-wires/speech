import { useEffect, useMemo, useState } from "react";

import { decayTime, fadeEdges, fftConvolve, gaussian, schroederDecay, seededRandom } from "./dsp";
import type { DemoProps } from "./index";
import {
  Action,
  Actions,
  Choice,
  Curve,
  DemoFrame,
  Legend,
  Plot,
  Readout,
  Readouts,
  Slider,
  Waveform,
  envelope,
  useAudioPlayer,
} from "./kit";
import { smoothstep, synthesizeVoice, type Formant } from "./synth";

const RATE = 16000;
const DRY_SECONDS = 1.2;
/** The reverberant tail starts this long after the direct sound. */
const TAIL_START_S = 0.004;
/** A 60 dB fall in energy is an amplitude factor of 10^-3 = e^-6.908. */
const LN_1000 = Math.log(1000);
/** The floor of the decay plot. */
const FLOOR_DB = -80;

/** Three syllables, each a vowel with its own formants and a falling pitch. */
const SYLLABLES: { start: number; end: number; formants: Formant[]; f0: [number, number] }[] = [
  { start: 0.05, end: 0.3, formants: [[700, 80], [1200, 90], [2500, 150], [3500, 200]], f0: [135, 120] },
  { start: 0.42, end: 0.64, formants: [[300, 60], [2200, 90], [3000, 150], [3600, 200]], f0: [150, 128] },
  { start: 0.78, end: 1.08, formants: [[450, 70], [900, 90], [2400, 150], [3400, 200]], f0: [128, 100] },
];

function syllableAt(t: number) {
  return SYLLABLES.find((s) => t >= s.start && t < s.end);
}

function speechLike(): Float32Array {
  const voice = synthesizeVoice({
    sampleRate: RATE,
    seconds: DRY_SECONDS,
    f0: (t) => {
      const s = syllableAt(t);
      if (!s) return 0;
      const x = (t - s.start) / (s.end - s.start);
      return s.f0[0] + (s.f0[1] - s.f0[0]) * x;
    },
    formants: (t) => (syllableAt(t) ?? SYLLABLES[0]).formants,
    peak: 0.7,
  });
  // Each syllable swells in and dies away, as a spoken one does.
  for (let i = 0; i < voice.length; i += 1) {
    const t = i / RATE;
    const s = syllableAt(t);
    voice[i] *= s ? smoothstep(s.start, s.start + 0.03, t) * (1 - smoothstep(s.end - 0.06, s.end, t)) : 0;
  }
  return voice;
}

function clicks(): Float32Array {
  const out = new Float32Array(Math.round(DRY_SECONDS * RATE));
  for (const at of [0.05, 0.35, 0.65, 0.95]) {
    const start = Math.round(at * RATE);
    [0.5, 1, 1, 0.5].forEach((g, k) => {
      out[start + k] = 0.8 * g * (k % 2 ? -1 : 1);
    });
  }
  return out;
}

const DRY = { speech: speechLike(), clicks: clicks() };
type DryName = keyof typeof DRY;

/**
 * A synthetic room: a direct-path impulse, then Gaussian noise under an
 * exponential envelope that falls 60 dB in `rt60` seconds, scaled so the
 * direct sound holds `drrDb` more energy than the tail.
 */
function impulseResponse(rt60: number, drrDb: number): Float64Array {
  const length = Math.round(Math.min(3, 1.25 * rt60 + 0.05) * RATE);
  const ir = new Float64Array(length);
  const random = seededRandom(11);
  const start = Math.round(TAIL_START_S * RATE);
  let energy = 0;
  for (let n = start; n < length; n += 1) {
    ir[n] = gaussian(random) * Math.exp((-LN_1000 * n) / (RATE * rt60));
    energy += ir[n] * ir[n];
  }
  const scale = Math.sqrt(10 ** (-drrDb / 10) / energy);
  for (let n = start; n < length; n += 1) ir[n] *= scale;
  ir[0] = 1;
  return ir;
}

/** `value`, but only once it has stopped changing for `ms`. */
function useSettled<T>(value: T, ms: number): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setSettled(value), ms);
    return () => window.clearTimeout(timer);
  }, [value, ms]);
  return settled;
}

function normalised(samples: ArrayLike<number>, peak: number): Float32Array {
  let max = 0;
  for (let i = 0; i < samples.length; i += 1) max = Math.max(max, Math.abs(samples[i]));
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) out[i] = max > 0 ? (samples[i] * peak) / max : 0;
  return out;
}

export default function RoomReverb({ caption }: DemoProps) {
  const [rt60, setRt60] = useState(0.6);
  const [drrDb, setDrrDb] = useState(0);
  const [dryName, setDryName] = useState<DryName>("speech");
  const { play, stop, playing } = useAudioPlayer();

  const analysis = useMemo(() => {
    const ir = impulseResponse(rt60, drrDb);
    const edc = schroederDecay(ir);
    const t30 = decayTime(edc, RATE, -5, -35);
    const t20 = decayTime(edc, RATE, -5, -25);
    const level = envelope(ir, RATE, 500).map(([t, v]): [number, number] => [
      t,
      Math.max(FLOOR_DB, 20 * Math.log10(Math.abs(v) + 1e-12)),
    ]);
    const decay = envelope(edc, RATE, 500).map(([t, v]): [number, number] => [t, Math.max(FLOOR_DB, v)]);
    let tail = 0;
    for (let n = 1; n < ir.length; n += 1) tail += ir[n] * ir[n];
    const measuredDrr = 10 * Math.log10(1 / tail);
    return { ir, t30, t20, level, decay, measuredDrr, seconds: ir.length / RATE };
  }, [rt60, drrDb]);

  // Convolving a second of sound with a long room takes a few tens of
  // milliseconds, so it waits for the sliders to settle.
  const settled = useSettled(`${rt60}|${drrDb}|${dryName}`, 150);
  const wet = useMemo(() => {
    const [t, d, name] = settled.split("|");
    const ir = impulseResponse(Number(t), Number(d));
    return normalised(fftConvolve(DRY[name as DryName], ir), 0.8);
  }, [settled]);

  const dry = DRY[dryName];
  const fit = analysis.t30;
  const fitLine: [number, number][] | null = fit
    ? [
        [0, fit.intercept],
        [(-60 - fit.intercept) / fit.slope, -60],
      ]
    : null;
  const ideal: [number, number][] = [
    [0, 0],
    [rt60, -60],
  ];
  const close = fit ? Math.abs(fit.rt60 - rt60) / rt60 < 0.1 : false;
  const span = Math.max(analysis.seconds, wet.length / RATE);

  return (
    <DemoFrame
      title="A room as an impulse response"
      caption={caption}
      controls={
        <>
          <Slider
            label="Reverberation time, RT60"
            value={rt60}
            min={0.1}
            max={2}
            step={0.05}
            format={(v) => `${v.toFixed(2)} s`}
            onChange={setRt60}
          />
          <Slider
            label="Direct-to-reverberant ratio"
            value={drrDb}
            min={-10}
            max={20}
            step={1}
            format={(v) => `${v > 0 ? "+" : ""}${v} dB`}
            onChange={setDrrDb}
          />
          <Choice
            label="Dry sound"
            value={dryName}
            options={[
              { value: "speech", label: "Three syllables" },
              { value: "clicks", label: "Four clicks" },
            ]}
            onChange={setDryName}
          />
          <Actions>
            <Action onClick={playing === "dry" ? stop : () => play("dry", fadeEdges(Float32Array.from(dry), RATE, 0.005), RATE, 0.5)}>
              {playing === "dry" ? "Stop" : "Play dry"}
            </Action>
            <Action tone="quiet" onClick={playing === "wet" ? stop : () => play("wet", wet, RATE, 0.5)}>
              {playing === "wet" ? "Stop" : "Play in the room"}
            </Action>
          </Actions>
        </>
      }
    >
      <Plot
        height={220}
        xDomain={[0, analysis.seconds]}
        yDomain={[FLOOR_DB, 0]}
        xLabel="time (s)"
        yLabel="dB"
        yTicks={4}
        tickFormat={(value, axis) => (axis === "x" ? value.toFixed(2) : value.toFixed(0))}
      >
        {({ sx, sy }) => (
          <>
            <Curve points={analysis.level} stroke="var(--ink-4)" width={0.8} opacity={0.8} />
            <Curve points={ideal} stroke="var(--ink-3)" width={1.2} dashed />
            {fitLine && <Curve points={fitLine} stroke="var(--bad)" width={1.6} dashed />}
            <Curve points={analysis.decay} width={2} />
            {[-5, -35].map((db) => (
              <text
                key={db}
                x={sx(analysis.seconds) - 4}
                y={sy(db) - 3}
                textAnchor="end"
                fontSize={10}
                fontFamily="var(--font-mono)"
                fill="var(--ink-3)"
              >
                {db} dB
              </text>
            ))}
            {[-5, -35].map((db) => (
              <line
                key={`l${db}`}
                x1={sx(0)}
                x2={sx(analysis.seconds)}
                y1={sy(db)}
                y2={sy(db)}
                stroke="var(--rule-strong)"
                strokeDasharray="2 3"
              />
            ))}
          </>
        )}
      </Plot>
      <Legend
        items={[
          { label: "the impulse response, |h| in dB", colour: "var(--ink-4)" },
          { label: "Schroeder decay curve", colour: "var(--demo)" },
          { label: "a line fitted from -5 to -35 dB", colour: "var(--bad)", dashed: true },
          { label: `the set decay, 60 dB in ${rt60.toFixed(2)} s`, colour: "var(--ink-3)", dashed: true },
        ]}
      />

      <div className="mt-4">
        <Plot
          height={130}
          xDomain={[0, span]}
          yDomain={[-1, 1]}
          xLabel="time (s)"
          yTicks={2}
          tickFormat={(value, axis) => (axis === "x" ? value.toFixed(1) : value.toFixed(0))}
        >
          {() => (
            <>
              <Waveform samples={wet} sampleRate={RATE} colour="var(--demo)" width={0.9} />
              <Waveform samples={dry} sampleRate={RATE} colour="var(--ink-2)" width={0.9} />
            </>
          )}
        </Plot>
      </div>
      <Legend
        items={[
          { label: "dry", colour: "var(--ink-2)" },
          { label: "in the room: dry convolved with the response", colour: "var(--demo)" },
        ]}
      />

      <Readouts>
        <Readout label="RT60 set" value={`${rt60.toFixed(2)} s`} />
        <Readout
          label="Measured, T30"
          value={fit ? `${fit.rt60.toFixed(2)} s` : "n/a"}
          hint="the fit, extended to 60 dB"
          tone={close ? "good" : "bad"}
        />
        <Readout label="Measured, T20" value={analysis.t20 ? `${analysis.t20.rt60.toFixed(2)} s` : "n/a"} hint="fit from -5 to -25 dB" />
        <Readout label="Direct to reverberant" value={`${analysis.measuredDrr.toFixed(1)} dB`} hint="direct energy over the tail's" />
      </Readouts>
    </DemoFrame>
  );
}
