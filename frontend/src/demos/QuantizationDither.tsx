import { useMemo, useState } from "react";

import { fadeEdges, hann, rfftMagnitude, seededRandom } from "./dsp";
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
  Toggle,
  useAudioPlayer,
} from "./kit";

const RATE = 16000;
const SECONDS = 1;
const TONE = 440;
/** The span the waveform plots show: 6 ms, about two and a half cycles of the tone. */
const SHOWN_SAMPLES = 96;
const WELCH = 4096;
const PLAY_GAIN = 0.25;
/** TPDF dither adds noise of variance 2^2 / 6 on top of the error's 2^2 / 12. */
const TPDF_PENALTY_DB = 10 * Math.log10(3);

type Result = {
  clean: Float32Array;
  quantized: Float32Array;
  error: Float32Array;
  snrDb: number;
  clipped: number;
  spectrumDb: [number, number][];
};

/**
 * Quantizes a sine at `level` dBFS to `bits` bits the way 16-bit PCM does:
 * `round(x * 2^(N-1)) / 2^(N-1)`, clipped to the codes that exist. With
 * dither, triangular noise two steps wide is added before rounding.
 */
function quantizeTone(bits: number, dither: boolean, levelDb: number): Result {
  const n = RATE * SECONDS;
  const amplitude = 10 ** (levelDb / 20);
  const scale = 2 ** (bits - 1);
  const random = seededRandom(2);
  const clean = new Float32Array(n);
  const quantized = new Float32Array(n);
  const error = new Float32Array(n);
  let signalEnergy = 0;
  let errorEnergy = 0;
  let clipped = 0;
  for (let i = 0; i < n; i += 1) {
    const x = amplitude * Math.sin((2 * Math.PI * TONE * i) / RATE);
    // TPDF: the difference of two uniform draws, from -1 to 1 step.
    const d = dither ? random() - random() : 0;
    let code = Math.round(x * scale + d);
    if (code > scale - 1 || code < -scale) {
      clipped += 1;
      code = Math.max(-scale, Math.min(scale - 1, code));
    }
    const q = code / scale;
    clean[i] = x;
    quantized[i] = q;
    error[i] = q - x;
    signalEnergy += x * x;
    errorEnergy += (q - x) * (q - x);
  }

  // The error's spectrum, averaged over Hann frames (Welch), in dB relative
  // to a full-scale sine, so a tone at 0 dBFS would read 0 dB.
  const window = hann(WELCH);
  const coherent = window.reduce((sum, w) => sum + w, 0) / 2;
  const power = new Float64Array(WELCH / 2 + 1);
  const frame = new Float64Array(WELCH);
  let frames = 0;
  for (let start = 0; start + WELCH <= n; start += WELCH / 2) {
    for (let i = 0; i < WELCH; i += 1) frame[i] = error[start + i] * window[i];
    const magnitude = rfftMagnitude(frame);
    for (let k = 0; k < power.length; k += 1) power[k] += magnitude[k] * magnitude[k];
    frames += 1;
  }
  const spectrumDb: [number, number][] = [];
  for (let k = 0; k < power.length; k += 1) {
    const db = 10 * Math.log10(Math.max(1e-30, power[k] / frames) / (coherent * coherent));
    spectrumDb.push([(k * RATE) / WELCH, Math.max(-170, db)]);
  }

  return {
    clean,
    quantized,
    error,
    snrDb: 10 * Math.log10(signalEnergy / Math.max(1e-300, errorEnergy)),
    clipped,
    spectrumDb,
  };
}

const ms = (i: number) => (i / RATE) * 1000;

export default function QuantizationDither({ caption }: DemoProps) {
  const [bits, setBits] = useState(4);
  const [dither, setDither] = useState(false);
  const [levelDb, setLevelDb] = useState(-1);
  const { play, stop, playing } = useAudioPlayer();

  const result = useMemo(() => quantizeTone(bits, dither, levelDb), [bits, dither, levelDb]);
  const step = 2 ** (1 - bits);
  const theory = 6.02 * bits + 1.76 + levelDb;
  const expected = dither ? theory - TPDF_PENALTY_DB : theory;

  const { tone, steps, errorPoints } = useMemo(() => {
    const amplitude = 10 ** (levelDb / 20);
    const tone: [number, number][] = [];
    for (let i = 0; i <= 400; i += 1) {
      const at = (i / 400) * SHOWN_SAMPLES;
      tone.push([ms(at), amplitude * Math.sin((2 * Math.PI * TONE * at) / RATE)]);
    }
    const steps: [number, number][] = [];
    const errorPoints: [number, number][] = [];
    for (let i = 0; i < SHOWN_SAMPLES; i += 1) {
      steps.push([ms(i), result.quantized[i]], [ms(i + 1), result.quantized[i]]);
      // In steps of the quantizer, clamped to the plot so an overload stays visible.
      errorPoints.push([ms(i), Math.max(-1.9, Math.min(1.9, result.error[i] / step))]);
    }
    return { tone, steps, errorPoints };
  }, [result, levelDb, step]);

  function playClip(name: string, samples: Float32Array, boost = false) {
    const copy = Float32Array.from(samples);
    if (boost) {
      let peak = 0;
      for (const v of copy) peak = Math.max(peak, Math.abs(v));
      if (peak > 0) for (let i = 0; i < copy.length; i += 1) copy[i] /= peak;
    }
    play(name, fadeEdges(copy, RATE), RATE, PLAY_GAIN);
  }

  const button = (name: string, label: string, samples: Float32Array, boost = false) => (
    <Action tone={name === "clean" ? "primary" : "quiet"} onClick={playing === name ? stop : () => playClip(name, samples, boost)}>
      {playing === name ? "Stop" : label}
    </Action>
  );

  return (
    <DemoFrame
      title="Quantizing a signal, with and without dither"
      caption={caption}
      controls={
        <>
          <Slider label="Bits" value={bits} min={1} max={16} onChange={setBits} />
          <Slider
            label="Level of the sine"
            value={levelDb}
            min={-40}
            max={0}
            step={1}
            format={(v) => `${v} dBFS`}
            onChange={setLevelDb}
          />
          <Toggle label="Add TPDF dither before rounding" checked={dither} onChange={setDither} />
          <Actions>
            {button("clean", "Play the original", result.clean)}
            {button("quantized", "Play the quantized", result.quantized)}
            {button("error", "Play the error, boosted", result.error, true)}
          </Actions>
        </>
      }
    >
      <Plot
        height={170}
        xDomain={[0, ms(SHOWN_SAMPLES)]}
        xTicks={6}
        yDomain={[-1.1, 1.1]}
        xLabel="time (ms)"
        yLabel="amplitude"
        yTicks={2}
        tickFormat={(value, axis) => value.toFixed(axis === "x" ? 0 : 1)}
      >
        {() => (
          <>
            <Curve points={tone} stroke="var(--ink-4)" width={1.2} />
            <Curve points={steps} width={1.6} />
          </>
        )}
      </Plot>
      <Legend
        items={[
          { label: `a ${TONE} Hz sine at ${levelDb} dBFS`, colour: "var(--ink-4)" },
          { label: `quantized to ${bits} bit${bits === 1 ? "" : "s"}, ${2 ** bits} levels`, colour: "var(--demo)" },
        ]}
      />

      <div className="mt-4">
        <Plot
          height={120}
          xDomain={[0, ms(SHOWN_SAMPLES)]}
          xTicks={6}
          yDomain={[-2, 2]}
          xLabel="time (ms)"
          yLabel="error (steps)"
          yTicks={4}
          tickFormat={(value, axis) => value.toFixed(axis === "x" ? 0 : 1)}
        >
          {({ sx, sy }) => (
            <>
              {[-0.5, 0.5].map((y) => (
                <line
                  key={y}
                  x1={sx(0)}
                  x2={sx(ms(SHOWN_SAMPLES))}
                  y1={sy(y)}
                  y2={sy(y)}
                  stroke="var(--ink-4)"
                  strokeDasharray="3 3"
                />
              ))}
              <Curve points={errorPoints} stroke="var(--bad)" width={0.9} opacity={0.5} />
              <Points points={errorPoints} r={2.2} fill="var(--bad)" />
            </>
          )}
        </Plot>
      </div>

      <div className="mt-4">
        <Plot
          height={170}
          xDomain={[0, RATE / 2]}
          yDomain={[-160, 0]}
          xLabel="frequency (Hz)"
          yLabel="error power (dB)"
          yTicks={4}
          xTicks={4}
          tickFormat={(value) => value.toFixed(0)}
        >
          {() => (
            <>
              <Rule x={TONE} label={`${TONE} Hz`} />
              <Curve points={result.spectrumDb} stroke="var(--bad)" width={1.1} />
            </>
          )}
        </Plot>
      </div>
      <p className="mt-1 text-[12.5px] leading-snug text-ink-3">
        {dither
          ? "With dither the error is noise: its spectrum is flat, whatever the signal does."
          : "Without dither the error repeats whenever the signal does, so its spectrum is a comb of spikes: distortion, not noise."}
      </p>

      <Readouts>
        <Readout label="Step" value={step >= 0.001 ? step.toPrecision(3) : step.toExponential(2)} hint="2 / 2^N" />
        <Readout
          label="Measured SNR"
          value={`${result.snrDb.toFixed(1)} dB`}
          tone={Math.abs(result.snrDb - expected) < 1.5 ? "good" : "bad"}
        />
        <Readout
          label={dither ? "Expected, with dither" : "6.02N + 1.76 + level"}
          value={`${expected.toFixed(1)} dB`}
          hint={dither ? `the rule, less ${TPDF_PENALTY_DB.toFixed(2)} dB` : undefined}
        />
        <Readout
          label="Clipped samples"
          value={result.clipped}
          tone={result.clipped ? "bad" : "plain"}
          hint={result.clipped ? "past the top code" : "none"}
        />
      </Readouts>
    </DemoFrame>
  );
}
