import { useMemo, useState } from "react";

import { fadeEdges, sine } from "./dsp";
import type { DemoProps } from "./index";
import {
  Action,
  Actions,
  Curve,
  DemoFrame,
  Legend,
  Plot,
  Readout,
  Readouts,
  Rule,
  Slider,
  Stems,
  useAudioPlayer,
} from "./kit";

/** How many samples the time plot shows, whatever the rate. */
const SHOWN_SAMPLES = 20;
const CURVE_POINTS = 1200;
const PLAY_SECONDS = 1.2;
const PLAY_GAIN = 0.25;

/**
 * The frequency the samples of a sine at `f` encode, when taken at `fs`.
 *
 * Sampling cannot tell `f` from `f - k fs` for any whole number `k`, so the
 * samples are those of the one closest to zero. A negative result is the same
 * sine with its sign flipped.
 */
function aliasOf(f: number, fs: number): number {
  return f - fs * Math.round(f / fs);
}

const kHz = (hz: number) => (hz >= 1000 ? `${+(hz / 1000).toFixed(2)} kHz` : `${Math.round(hz)} Hz`);

export default function SamplingAndAliasing({ caption }: DemoProps) {
  const [frequency, setFrequency] = useState(5000);
  const [rate, setRate] = useState(8000);
  const { play, stop, playing } = useAudioPlayer();

  const nyquist = rate / 2;
  const alias = aliasOf(frequency, rate);
  const heard = Math.abs(alias);
  const aliased = frequency > nyquist;
  const span = SHOWN_SAMPLES / rate;

  const { wave, samples, rebuilt } = useMemo(() => {
    const wave: [number, number][] = [];
    const rebuilt: [number, number][] = [];
    for (let i = 0; i <= CURVE_POINTS; i += 1) {
      const t = (i / CURVE_POINTS) * span;
      wave.push([t * 1000, Math.sin(2 * Math.PI * frequency * t)]);
      rebuilt.push([t * 1000, Math.sin(2 * Math.PI * alias * t)]);
    }
    const samples: [number, number][] = [];
    for (let n = 0; n <= SHOWN_SAMPLES; n += 1) {
      samples.push([(n / rate) * 1000, Math.sin((2 * Math.PI * frequency * n) / rate)]);
    }
    return { wave, samples, rebuilt };
  }, [frequency, rate, alias, span]);

  const reconstruction = aliased ? "var(--bad)" : "var(--good)";
  const spectrumMax = Math.max(frequency, rate) * 1.08;

  function playTone() {
    // At the output rate, so this is the sine itself, before any sampling.
    const context = 48000;
    play("tone", fadeEdges(sine(frequency, context, PLAY_SECONDS, 1), context), context, PLAY_GAIN);
  }

  function playSamples() {
    // The samples themselves, at their own rate. The browser's resampler is
    // the reconstruction filter, so what comes out is what the samples encode.
    play("samples", fadeEdges(sine(frequency, rate, PLAY_SECONDS, 1), rate), rate, PLAY_GAIN);
  }

  return (
    <DemoFrame
      title="Sampling a sine, and the alias it can turn into"
      caption={caption}
      controls={
        <>
          <Slider
            label="Frequency of the sine"
            value={frequency}
            min={100}
            max={12000}
            step={50}
            format={kHz}
            onChange={setFrequency}
          />
          <Slider
            label="Sample rate"
            value={rate}
            min={4000}
            max={24000}
            step={500}
            format={kHz}
            onChange={setRate}
          />
          <Actions>
            <Action onClick={playing === "tone" ? stop : playTone}>
              {playing === "tone" ? "Stop" : "Play the sine"}
            </Action>
            <Action tone="quiet" onClick={playing === "samples" ? stop : playSamples}>
              {playing === "samples" ? "Stop" : "Play the samples"}
            </Action>
          </Actions>
        </>
      }
    >
      <Plot
        xDomain={[0, span * 1000]}
        yDomain={[-1.25, 1.25]}
        xLabel="time (ms)"
        yLabel="amplitude"
        yTicks={2}
        tickFormat={(value, axis) => (axis === "x" ? value.toFixed(2) : value.toFixed(1))}
      >
        {() => (
          <>
            <Curve points={wave} stroke="var(--ink-4)" width={1} />
            <Curve points={rebuilt} stroke={reconstruction} width={1.8} dashed={aliased} />
            <Stems points={samples} />
          </>
        )}
      </Plot>
      <Legend
        items={[
          { label: `the sine, ${kHz(frequency)}`, colour: "var(--ink-4)" },
          { label: "the samples", colour: "var(--demo)" },
          { label: `what they reconstruct, ${kHz(heard)}`, colour: reconstruction, dashed: aliased },
        ]}
      />

      <div className="mt-4">
        <Plot
          height={96}
          xDomain={[0, spectrumMax]}
          yDomain={[0, 1.15]}
          xLabel="frequency"
          yTicks={0}
          xTicks={4}
          tickFormat={(value) => kHz(value)}
        >
          {() => (
            <>
              <Rule x={nyquist} label="fs/2" />
              <Rule x={rate} label="fs" colour="var(--ink-4)" />
              <Curve points={[[frequency, 0], [frequency, 1]]} stroke="var(--ink-3)" width={2.4} />
              {aliased && (
                <Curve points={[[heard, 0], [heard, 0.8]]} stroke="var(--bad)" width={2.4} />
              )}
            </>
          )}
        </Plot>
      </div>

      <Readouts>
        <Readout label="Nyquist, fs/2" value={kHz(nyquist)} />
        <Readout label="The sine" value={kHz(frequency)} />
        <Readout label="The samples say" value={kHz(heard)} tone={aliased ? "bad" : "good"} />
        <Readout label="Aliased" value={aliased ? "yes" : "no"} tone={aliased ? "bad" : "good"} />
      </Readouts>
    </DemoFrame>
  );
}
