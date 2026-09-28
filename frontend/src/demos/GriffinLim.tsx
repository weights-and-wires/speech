import { useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";

import {
  fadeEdges,
  istft,
  linearChirp,
  seededRandom,
  stft,
  stftComplex,
  toDb,
  type ComplexSpectrogram,
} from "./dsp";
import type { DemoProps } from "./index";
import {
  Action,
  Actions,
  Choice,
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
  Spectrogram,
  useAudioPlayer,
  useWhenPrinted,
} from "./kit";
import { synthesizeVoice, vowelFormants } from "./synth";

const RATE = 16000;
const SECONDS = 0.8;
const N_FFT = 512;
const HOP = 128;
const FLOOR_DB = -70;
/** Iterations a printed figure shows. */
const PRINTED_ITERATIONS = 50;
/** A cap on the iterations kept, since each one keeps its reconstruction. */
const MAX_ITERATIONS = 200;
/** The time budget for each animation frame while running. */
const FRAME_BUDGET_MS = 12;

/**
 * The test signal: a vowel with vibrato for the first half, then a chirp.
 * The vowel's harmonics need their phases to line up across frames, and the
 * chirp needs them to follow a moving frequency, so both show how far the
 * phase Griffin-Lim invents is from the real one.
 */
function testSignal(): Float32Array {
  const n = Math.round(SECONDS * RATE);
  const out = new Float32Array(n);
  const vowel = synthesizeVoice({
    sampleRate: RATE,
    seconds: 0.36,
    f0: (t) => 150 + 8 * Math.sin(2 * Math.PI * 5 * t),
    formants: vowelFormants(730, 1090, 2440),
    peak: 0.6,
  });
  fadeEdges(vowel, RATE, 0.03);
  out.set(vowel, Math.round(0.03 * RATE));
  const chirp = fadeEdges(linearChirp(500, 3500, RATE, 0.34, 0.3), RATE, 0.03);
  out.set(chirp, Math.round(0.43 * RATE));
  return out;
}

const SIGNAL = testSignal();

type Chain = {
  momentum: number;
  angleRe: Float64Array;
  angleIm: Float64Array;
  previous: ComplexSpectrogram | null;
  sc: number[];
  /** The reconstruction at each iteration, kept for the chain the reader hears. */
  snapshots: Float32Array[] | null;
};

/** The target magnitudes and their norm, which spectral convergence divides by. */
function target() {
  const spec = stftComplex(SIGNAL, { nFft: N_FFT, hop: HOP });
  const magnitude = new Float64Array(spec.re.length);
  let norm = 0;
  for (let i = 0; i < magnitude.length; i += 1) {
    magnitude[i] = Math.hypot(spec.re[i], spec.im[i]);
    norm += magnitude[i] * magnitude[i];
  }
  return { spec, magnitude, norm: Math.sqrt(norm) };
}

const TARGET = target();

function newChain(momentum: number, init: "random" | "zero", keep: boolean): Chain {
  const size = TARGET.magnitude.length;
  const angleRe = new Float64Array(size);
  const angleIm = new Float64Array(size);
  const random = seededRandom(5);
  for (let i = 0; i < size; i += 1) {
    const phase = init === "random" ? 2 * Math.PI * random() : 0;
    angleRe[i] = Math.cos(phase);
    angleIm[i] = Math.sin(phase);
  }
  return { momentum, angleRe, angleIm, previous: null, sc: [], snapshots: keep ? [] : null };
}

/**
 * One Griffin-Lim iteration, as `librosa.griffinlim` runs it: invert the
 * target magnitudes with the current phases, take the STFT of the result,
 * and keep its phases, pushed along by momentum for the fast variant.
 */
function step(chain: Chain) {
  const { bins, frames, nFft, hop } = TARGET.spec;
  const size = TARGET.magnitude.length;
  const re = new Float64Array(size);
  const im = new Float64Array(size);
  for (let i = 0; i < size; i += 1) {
    re[i] = TARGET.magnitude[i] * chain.angleRe[i];
    im[i] = TARGET.magnitude[i] * chain.angleIm[i];
  }
  const signal = istft({ bins, frames, nFft, hop, re, im }, { length: SIGNAL.length });
  const rebuilt = stftComplex(signal, { nFft: N_FFT, hop: HOP });

  let distance = 0;
  for (let i = 0; i < size; i += 1) {
    const d = Math.hypot(rebuilt.re[i], rebuilt.im[i]) - TARGET.magnitude[i];
    distance += d * d;
  }
  chain.sc.push(Math.sqrt(distance) / TARGET.norm);
  chain.snapshots?.push(Float32Array.from(signal));

  const push = chain.previous ? chain.momentum / (1 + chain.momentum) : 0;
  for (let i = 0; i < size; i += 1) {
    let ar = rebuilt.re[i];
    let ai = rebuilt.im[i];
    if (push && chain.previous) {
      ar -= push * chain.previous.re[i];
      ai -= push * chain.previous.im[i];
    }
    const length = Math.hypot(ar, ai) + 1e-16;
    chain.angleRe[i] = ar / length;
    chain.angleIm[i] = ai / length;
  }
  chain.previous = rebuilt;
}

const dB = (sc: number) => 20 * Math.log10(Math.max(1e-6, sc));

export default function GriffinLim({ caption }: DemoProps) {
  const [momentum, setMomentum] = useState(0.99);
  const [init, setInit] = useState<"random" | "zero">("random");
  const [goal, setGoal] = useState<"25" | "50" | "100">("50");
  const [running, setRunning] = useState(false);
  /** Bumped whenever the chains move on, so the plots redraw. */
  const [version, setVersion] = useState(0);
  const [shown, setShown] = useState<number | null>(null);
  const { play, stop, playing } = useAudioPlayer();

  const printed = useContext(PrintedFigure) !== null;
  const chains = useRef<{ main: Chain; plain: Chain | null } | null>(null);
  const frame = useRef<number | null>(null);

  const reset = useCallback(() => {
    if (frame.current !== null) cancelAnimationFrame(frame.current);
    frame.current = null;
    const main = newChain(momentum, init, true);
    const plain = momentum > 0 ? newChain(0, init, false) : null;
    step(main);
    if (plain) step(plain);
    chains.current = { main, plain };
    setRunning(false);
    setShown(null);
    setVersion((v) => v + 1);
  }, [momentum, init]);

  useEffect(() => {
    reset();
  }, [reset]);

  useEffect(() => () => {
    if (frame.current !== null) cancelAnimationFrame(frame.current);
  }, []);

  const iterations = Number(goal);

  function runTo(count: number, animate: boolean) {
    const current = chains.current;
    if (!current || current.main.sc.length > count) return;
    const advance = () => {
      step(current.main);
      if (current.plain) step(current.plain);
    };
    if (!animate) {
      while (current.main.sc.length <= count) advance();
      setVersion((v) => v + 1);
      return;
    }
    setShown(null);
    setRunning(true);
    const tick = () => {
      const started = performance.now();
      do advance();
      while (current.main.sc.length <= count && performance.now() - started < FRAME_BUDGET_MS);
      setVersion((v) => v + 1);
      if (current.main.sc.length <= count) frame.current = requestAnimationFrame(tick);
      else {
        frame.current = null;
        setRunning(false);
      }
    };
    frame.current = requestAnimationFrame(tick);
  }

  function pause() {
    if (frame.current !== null) cancelAnimationFrame(frame.current);
    frame.current = null;
    setRunning(false);
  }

  useWhenPrinted(() => {
    // The effect above has built iteration 0 by now; run on from there.
    if (!chains.current) reset();
    runTo(PRINTED_ITERATIONS, false);
  });

  const main = chains.current?.main;
  const plain = chains.current?.plain ?? null;
  const done = main ? main.sc.length - 1 : 0;
  const at = Math.min(done, shown ?? done);

  const curves = useMemo(() => {
    const toPoints = (sc: number[]) => sc.map((v, i): [number, number] => [i, dB(v)]);
    return { main: toPoints(main?.sc ?? []), plain: plain ? toPoints(plain.sc) : null };
    // `version` says the chains have moved on.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, main, plain]);

  const reconstruction = main?.snapshots?.[at] ?? null;
  const spectrogram = useMemo(() => {
    const original = stft(SIGNAL, { nFft: N_FFT, hop: HOP });
    return { ...original, data: toDb(original.data, { floorDb: FLOOR_DB }) };
  }, []);
  const rebuiltSpectrogram = useMemo(() => {
    if (!reconstruction) return null;
    const spec = stft(reconstruction, { nFft: N_FFT, hop: HOP });
    return { ...spec, data: toDb(spec.data, { floorDb: FLOOR_DB }) };
  }, [reconstruction]);

  const allValues = [...curves.main, ...(curves.plain ?? [])].map((p) => p[1]);
  const deepest = Math.min(...allValues, 0);
  const tickStep = deepest < -38 ? 10 : 5;
  const lowest = Math.min(-10, Math.floor((deepest - 2) / tickStep) * tickStep);
  const xMax = Math.max(iterations, done);
  const fast = momentum > 0;
  const scAt = main?.sc[at] ?? 1;
  const plainAt = plain?.sc[at];

  return (
    <DemoFrame
      title="Recovering phase from magnitude, one iteration at a time"
      caption={caption}
      controls={
        <>
          <Slider
            label="Momentum (fast Griffin-Lim)"
            value={momentum}
            min={0}
            max={0.99}
            step={0.01}
            format={(v) => (v === 0 ? "0, plain" : v.toFixed(2))}
            onChange={setMomentum}
          />
          <Choice
            label="Starting phase"
            value={init}
            options={[
              { value: "random", label: "Random" },
              { value: "zero", label: "Zero" },
            ]}
            onChange={setInit}
          />
          <Choice
            label="Iterations"
            value={goal}
            options={[
              { value: "25", label: "25" },
              { value: "50", label: "50" },
              { value: "100", label: "100" },
            ]}
            onChange={setGoal}
          />
          <Slider
            label="Show iteration"
            value={at}
            min={0}
            max={Math.max(1, done)}
            onChange={(v) => setShown(Math.min(v, done))}
          />
          <Actions>
            {running ? (
              <Action onClick={pause}>Pause</Action>
            ) : (
              <Action onClick={() => runTo(done >= iterations ? Math.min(MAX_ITERATIONS, done + iterations) : iterations, true)}>
                {done >= MAX_ITERATIONS ? "Done" : done >= iterations ? `Run ${iterations} more` : done > 0 ? "Continue" : "Run Griffin-Lim"}
              </Action>
            )}
            <Action tone="quiet" onClick={reset}>
              Start over
            </Action>
          </Actions>
          <Actions>
            <Action
              tone="quiet"
              onClick={playing === "original" ? stop : () => play("original", SIGNAL, RATE, 0.5)}
            >
              {playing === "original" ? "Stop" : "Play the original"}
            </Action>
            <Action
              tone="quiet"
              onClick={
                playing === "rebuilt"
                  ? stop
                  : () => reconstruction && play("rebuilt", reconstruction, RATE, 0.5)
              }
            >
              {playing === "rebuilt" ? "Stop" : `Play iteration ${at}`}
            </Action>
          </Actions>
        </>
      }
    >
      <Plot
        height={printed ? 150 : 180}
        xDomain={[0, xMax]}
        yDomain={[lowest, 0]}
        xLabel="iteration"
        yLabel="spectral convergence (dB)"
        yTicks={Math.round(-lowest / tickStep)}
        tickFormat={(value) => value.toFixed(0)}
      >
        {() => (
          <>
            <Rule x={at} />
            {curves.plain && <Curve points={curves.plain} stroke="var(--ink-4)" width={1.4} dashed />}
            <Curve points={curves.main} width={1.8} />
            {curves.main[at] && <Points points={[curves.main[at]]} r={3.6} />}
          </>
        )}
      </Plot>
      <Legend
        items={[
          { label: fast ? `fast Griffin-Lim, momentum ${momentum.toFixed(2)}` : "Griffin-Lim", colour: "var(--demo)" },
          ...(fast ? [{ label: "plain Griffin-Lim", colour: "var(--ink-4)", dashed: true }] : []),
        ]}
      />

      <p className="mt-4 text-[12.5px] font-medium text-ink-3">The original</p>
      <div className="mt-1">
        <Spectrogram spec={spectrogram} sampleRate={RATE} floorDb={FLOOR_DB} height={printed ? 100 : 130} description="The original signal's spectrogram" />
      </div>
      <p className="mt-3 text-[12.5px] font-medium text-ink-3">
        Rebuilt from its magnitudes alone, iteration {at}
      </p>
      <div className="mt-1">
        {rebuiltSpectrogram && (
          <Spectrogram
            spec={rebuiltSpectrogram}
            sampleRate={RATE}
            floorDb={FLOOR_DB}
            height={printed ? 100 : 130}
            description={`The spectrogram of the Griffin-Lim reconstruction after ${at} iterations`}
          />
        )}
      </div>

      <Readouts>
        <Readout label="Iteration" value={at} hint={running ? "running" : `of ${done} run`} />
        <Readout label="Spectral convergence" value={`${dB(scAt).toFixed(1)} dB`} hint={scAt.toFixed(3)} tone="demo" />
        <Readout
          label="Plain, same iteration"
          value={plainAt !== undefined ? `${dB(plainAt).toFixed(1)} dB` : "n/a"}
          hint={fast ? "no momentum" : "momentum is 0"}
        />
        <Readout label="STFT" value={`${N_FFT} / ${HOP}`} hint="Hann window / hop" />
      </Readouts>
    </DemoFrame>
  );
}
