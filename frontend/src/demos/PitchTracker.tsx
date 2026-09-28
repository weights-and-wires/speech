import { useEffect, useMemo, useRef, useState } from "react";

import { cumulativeMeanNormalized, yinDifference, yinPick } from "./dsp";
import type { DemoProps } from "./index";
import {
  Action,
  Actions,
  Curve,
  DemoFrame,
  Legend,
  Plot,
  Points,
  Readout,
  Readouts,
  Rule,
  Slider,
  useAudioPlayer,
  useMicrophone,
} from "./kit";
import { smoothstep, synthesizeVoice, type Formant } from "./synth";

const RATE = 16000;
const SECONDS = 1.6;
/** YIN's integration window, 25 ms, and a hop of 10 ms. */
const WINDOW = 400;
const HOP = 160;
const FMIN = 60;
const FMAX = 500;
const MIN_LAG = Math.floor(RATE / FMAX);
const MAX_LAG = Math.ceil(RATE / FMIN);
/** Lags computed and drawn: a little past the longest period searched. */
const SHOWN_LAG = 320;
/** An estimate more than 20% from the truth is a gross pitch error. */
const GROSS = 0.2;
const MIC_SECONDS = 3;

/**
 * The synthetic voice's pitch: a rise, a breath, then a fall with vibrato
 * that settles into a creaky ending.
 */
function trueF0(t: number): number {
  if (t >= 0.12 && t < 0.72) return 110 + 90 * smoothstep(0.12, 0.72, t);
  if (t >= 0.86 && t < 1.5) return 230 - 105 * smoothstep(0.86, 1.25, t) + 4 * Math.sin(2 * Math.PI * 5.5 * t);
  return 0;
}

/** Creak at the end of the phrase: every other pulse softer. */
const CREAK: [number, number] = [1.28, 1.48];
const CREAK_SWING = 0.1;

function formantsAt(t: number): Formant[] {
  // /ɑ/ gliding to /i/ over the second phrase.
  const x = smoothstep(0.86, 1.4, t);
  return [
    [730 + (300 - 730) * x, 70],
    [1090 + (2250 - 1090) * x, 90],
    [2440 + (2900 - 2440) * x, 150],
    [3500, 200],
  ];
}

function makeVoice(breath: number): Float32Array {
  return synthesizeVoice({
    sampleRate: RATE,
    seconds: SECONDS,
    f0: trueF0,
    formants: formantsAt,
    // A breathy source, and a stretch of pure breath noise between phrases.
    noise: (t) => (t >= 0.74 && t < 0.84 ? 1.5 : trueF0(t) > 0 ? breath : 0),
    alternate: (t) => (t >= CREAK[0] && t < CREAK[1] ? CREAK_SWING : 0),
    seed: 4,
  });
}

type Frame = { time: number; dPrime: Float64Array; silent: boolean };

/** YIN's cumulative mean normalized difference for every frame. */
function analyse(signal: Float32Array): Frame[] {
  const frames: Frame[] = [];
  for (let start = 0; start + WINDOW + SHOWN_LAG <= signal.length; start += HOP) {
    let energy = 0;
    for (let j = 0; j < WINDOW; j += 1) energy += signal[start + j] * signal[start + j];
    const d = yinDifference(signal, start, WINDOW, SHOWN_LAG);
    frames.push({
      time: (start + WINDOW / 2) / RATE,
      dPrime: cumulativeMeanNormalized(d),
      silent: Math.sqrt(energy / WINDOW) < 1e-3,
    });
  }
  return frames;
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

export default function PitchTracker({ caption }: DemoProps) {
  const [threshold, setThreshold] = useState(0.1);
  const [breath, setBreath] = useState(0.15);
  const [selected, setSelected] = useState(1.38);
  const [recorded, setRecorded] = useState<Float32Array | null>(null);
  const { play, stop, playing } = useAudioPlayer();
  const microphone = useMicrophone();
  const dragging = useRef(false);

  const settledBreath = useSettled(breath, 120);
  const synthetic = useMemo(() => makeVoice(settledBreath), [settledBreath]);
  const signal = recorded ?? synthetic;
  const known = !recorded;
  const seconds = signal.length / RATE;

  const frames = useMemo(() => analyse(signal), [signal]);

  const track = useMemo(() => {
    let voicedTruth = 0;
    let gross = 0;
    let unvoicedTruth = 0;
    let falseAlarms = 0;
    const points = frames.map((frame) => {
      // YIN's step 4: the first dip below the threshold, or else the lowest
      // point. A frame whose dip crossed the threshold is called voiced.
      const pick = yinPick(frame.dPrime, threshold, MIN_LAG, MAX_LAG);
      const voiced = pick.belowThreshold && !frame.silent;
      const hz = RATE / pick.lag;
      const truth = known ? trueF0(frame.time) : 0;
      const error = !frame.silent && truth > 0 && Math.abs(hz - truth) / truth > GROSS;
      if (known && !frame.silent && truth > 0) {
        voicedTruth += 1;
        if (error) gross += 1;
      }
      if (known && !frame.silent && truth === 0) {
        unvoicedTruth += 1;
        if (voiced) falseAlarms += 1;
      }
      const falseAlarm = known && voiced && truth === 0;
      return { time: frame.time, hz, voiced, error, falseAlarm, silent: frame.silent, pick };
    });
    return { points, voicedTruth, gross, unvoicedTruth, falseAlarms };
  }, [frames, threshold, known]);

  const index = Math.max(
    0,
    Math.min(frames.length - 1, Math.round((selected * RATE - WINDOW / 2) / HOP)),
  );
  const frame = frames[index];
  const current = track.points[index];

  const detail = useMemo(() => {
    if (!frame) return null;
    const start = Math.round(frame.time * RATE - WINDOW / 2);
    const d = yinDifference(signal, start, WINDOW, SHOWN_LAG);
    const dMax = Math.max(...d) || 1;
    return {
      d: Array.from(d, (v, tau): [number, number] => [tau, v / dMax]),
      dPrime: Array.from(frame.dPrime, (v, tau): [number, number] => [tau, Math.min(v, 1.95)]),
    };
  }, [frame, signal]);

  const truthSegments = useMemo(() => {
    if (!known) return [];
    const segments: [number, number][][] = [];
    let run: [number, number][] = [];
    for (let i = 0; i <= 800; i += 1) {
      const t = (i / 800) * SECONDS;
      const hz = trueF0(t);
      if (hz > 0) run.push([t, hz]);
      else if (run.length) {
        segments.push(run);
        run = [];
      }
    }
    if (run.length) segments.push(run);
    return segments;
  }, [known]);

  const truthHere = known && frame ? trueF0(frame.time) : 0;
  const truthLag = truthHere > 0 ? RATE / truthHere : null;

  function onTrack({ x, type }: { x: number; type: "down" | "move" | "up" }) {
    if (type === "down") dragging.current = true;
    if (!dragging.current) return;
    setSelected(Math.min(seconds, Math.max(0, x)));
    if (type === "up") dragging.current = false;
  }

  async function listenToMicrophone() {
    stop();
    const samples = await microphone.record(MIC_SECONDS, RATE);
    if (samples) {
      setRecorded(samples);
      setSelected(MIC_SECONDS / 2);
    }
  }

  const micNote: Record<string, string> = {
    asking: "Waiting for permission to use the microphone.",
    recording: `Recording ${MIC_SECONDS} seconds. Hum, sing, or say a long vowel.`,
    denied: "The browser was not allowed to use the microphone, so the demo keeps its synthetic voice.",
    unsupported: "This browser has no microphone to offer here, so the demo keeps its synthetic voice.",
    failed: "The recording could not be read, so the demo keeps its synthetic voice.",
  };

  const at = (p: { time: number; hz: number }): [number, number] => [p.time, Math.min(p.hz, 490)];
  const shown = track.points.filter((p) => !p.silent);
  const estimates = shown.filter((p) => p.voiced && !p.error && !p.falseAlarm).map(at);
  const fallbacks = shown.filter((p) => !p.voiced && !p.error).map(at);
  const errors = shown.filter((p) => p.error).map(at);
  const falseAlarms = shown.filter((p) => p.falseAlarm).map(at);
  const grossRate = track.voicedTruth ? track.gross / track.voicedTruth : 0;

  return (
    <DemoFrame
      title="Finding the pitch with YIN"
      caption={caption}
      controls={
        <>
          <Slider
            label="Absolute threshold"
            value={threshold}
            min={0.02}
            max={0.6}
            step={0.01}
            format={(v) => v.toFixed(2)}
            onChange={setThreshold}
          />
          <Slider
            label="Frame"
            value={Math.min(selected, seconds)}
            min={0}
            max={seconds}
            step={HOP / RATE}
            format={(v) => `${v.toFixed(2)} s`}
            onChange={setSelected}
          />
          {!recorded && (
            <Slider
              label="Breathiness"
              value={breath}
              min={0}
              max={1.5}
              step={0.05}
              format={(v) => v.toFixed(2)}
              onChange={setBreath}
            />
          )}
          <Actions>
            <Action onClick={playing ? stop : () => play("voice", signal, RATE, 0.5)}>
              {playing ? "Stop" : "Listen"}
            </Action>
            {recorded ? (
              <Action tone="quiet" onClick={() => setRecorded(null)}>
                Back to the synthetic voice
              </Action>
            ) : (
              <Action tone="quiet" onClick={listenToMicrophone}>
                Record {MIC_SECONDS} seconds from the microphone
              </Action>
            )}
          </Actions>
          {microphone.state !== "idle" && (
            <p className="text-[12.5px] leading-snug text-ink-3 sm:col-span-2" role="status">
              {micNote[microphone.state]}
            </p>
          )}
        </>
      }
    >
      <Plot
        height={200}
        xDomain={[0, seconds]}
        yDomain={[0, 500]}
        xLabel="time (s)"
        yLabel="F0 (Hz)"
        yTicks={5}
        tickFormat={(v, axis) => (axis === "x" ? v.toFixed(1) : v.toFixed(0))}
        onPointer={onTrack}
      >
        {() => (
          <>
            {frame && <Rule x={frame.time} />}
            {truthSegments.map((segment, i) => (
              <Curve key={i} points={segment} stroke="var(--ink-4)" width={5} opacity={0.45} />
            ))}
            <Points points={fallbacks} r={2.3} fill="var(--raise)" stroke="var(--ink-3)" />
            <Points points={estimates} r={2.3} />
            <Points points={falseAlarms} r={3} fill="var(--warn)" />
            <Points points={errors} r={3} fill="var(--bad)" />
          </>
        )}
      </Plot>
      <Legend
        items={[
          ...(known ? [{ label: "the true F0", colour: "var(--ink-4)" }] : []),
          { label: "a dip fell below the threshold: voiced", colour: "var(--demo)" },
          { label: "none did: YIN's lowest point, called unvoiced", colour: "var(--ink-3)" },
          ...(known
            ? [
                { label: "more than 20% off the truth", colour: "var(--bad)" },
                { label: "a pitch where there is none", colour: "var(--warn)" },
              ]
            : []),
        ]}
      />

      {detail && current && (
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <div className="min-w-0">
            <p className="text-[12.5px] font-medium text-ink-2">Difference function d(τ)</p>
            <Plot width={360} height={190} xDomain={[0, SHOWN_LAG]} yDomain={[0, 1.05]} xLabel="lag τ (samples)" yTicks={2} xTicks={4} tickFormat={(v) => v.toFixed(v < 1 && v > 0 ? 1 : 0)}>
              {() => (
                <>
                  {truthLag && <Rule x={truthLag} label="true" colour="var(--ink-4)" />}
                  <Curve points={detail.d} width={1.4} />
                </>
              )}
            </Plot>
          </div>
          <div className="min-w-0">
            <p className="text-[12.5px] font-medium text-ink-2">Cumulative mean normalized d′(τ)</p>
            <Plot width={360} height={190} xDomain={[0, SHOWN_LAG]} yDomain={[0, 2]} xLabel="lag τ (samples)" yTicks={4} xTicks={4} tickFormat={(v) => v.toFixed(v < 2 && v > 0 ? 1 : 0)}>
              {({ sx, sy, area }) => (
                <>
                  <rect
                    x={area.left}
                    y={area.top}
                    width={Math.max(0, sx(MIN_LAG) - area.left)}
                    height={area.bottom - area.top}
                    fill="color-mix(in oklab, var(--ink) 6%, transparent)"
                  />
                  <rect
                    x={sx(MAX_LAG)}
                    y={area.top}
                    width={Math.max(0, area.right - sx(MAX_LAG))}
                    height={area.bottom - area.top}
                    fill="color-mix(in oklab, var(--ink) 6%, transparent)"
                  />
                  <line x1={area.left} x2={area.right} y1={sy(threshold)} y2={sy(threshold)} stroke="var(--bad)" strokeDasharray="4 3" />
                  {truthLag && <Rule x={truthLag} colour="var(--ink-4)" />}
                  <Curve points={detail.dPrime} width={1.4} />
                  <Points points={[[current.pick.lag, Math.min(1.95, current.pick.value)]]} r={4} fill={current.voiced ? "var(--demo)" : "var(--ink-4)"} />
                </>
              )}
            </Plot>
          </div>
        </div>
      )}
      <p className="mt-2 text-[12.5px] leading-snug text-ink-3">
        The dashed red line is the threshold; the shaded lags are outside {FMIN} to {FMAX} Hz. YIN takes the first dip
        below the line; when none reaches it, it falls back to the lowest point and calls the frame unvoiced.
      </p>

      <Readouts>
        <Readout
          label={frame ? `True F0 at ${frame.time.toFixed(2)} s` : "True F0"}
          value={known ? (truthHere > 0 ? `${truthHere.toFixed(1)} Hz` : "unvoiced") : "unknown"}
          hint={`${WINDOW / 16} ms window`}
        />
        <Readout
          label="YIN says"
          value={current ? (current.silent ? "silence" : `${current.hz.toFixed(1)} Hz`) : "n/a"}
          hint={current ? `${current.voiced ? "voiced" : "unvoiced"}, d′ = ${current.pick.value.toFixed(2)} at τ = ${current.pick.lag.toFixed(1)}` : undefined}
          tone={current?.error ? "bad" : current?.falseAlarm ? "plain" : "demo"}
        />
        <Readout
          label="Gross pitch errors"
          value={known ? `${(grossRate * 100).toFixed(1)}%` : "no truth"}
          hint={known ? `${track.gross} of ${track.voicedTruth} voiced frames` : "a recording has no answer key"}
          tone={known ? (grossRate > 0.02 ? "bad" : "good") : "plain"}
        />
        <Readout
          label="Unvoiced, given a pitch"
          value={known ? `${track.falseAlarms} of ${track.unvoicedTruth}` : "no truth"}
          hint={known ? "breath frames called voiced" : "a recording has no answer key"}
          tone={known ? (track.falseAlarms ? "bad" : "good") : "plain"}
        />
      </Readouts>
    </DemoFrame>
  );
}
