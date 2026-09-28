import { useMemo, useState, type ReactNode } from "react";

import { fadeEdges, istft, linearChirp, siSdr, stftComplex, toDb, type ComplexSpectrogram } from "./dsp";
import { useDebounced } from "./figure";
import type { DemoProps } from "./index";
import {
  Action,
  Actions,
  Choice,
  DemoFrame,
  HeatmapCanvas,
  Readout,
  Readouts,
  Slider,
  Spectrogram,
  quantize,
  useAudioPlayer,
} from "./kit";

const RATE = 16000;
const SECONDS = 1.2;
const N_FFT = 512;
const HOP = 128;
const FLOOR_DB = -70;
/** The band the spectrograms show, [first, last) bins: 0 to 4 kHz. */
const ROWS: [number, number] = [0, Math.round(4000 / (RATE / N_FFT)) + 1];

/** A rising chirp, 400 Hz to 3.2 kHz. */
const CHIRP = fadeEdges(linearChirp(400, 3200, RATE, SECONDS, 0.5), RATE, 0.03);

/**
 * A vowel-like tone: harmonics of 150 Hz with a little vibrato, weighted by
 * three formant resonances near those of "ah" (700, 1200, and 2600 Hz).
 */
const VOWEL = (() => {
  const n = Math.round(SECONDS * RATE);
  const out = new Float32Array(n);
  const formants = [
    [700, 110, 1],
    [1200, 120, 0.6],
    [2600, 160, 0.25],
  ];
  let phase = 0;
  for (let i = 0; i < n; i += 1) {
    const f0 = 150 + 3 * Math.sin((2 * Math.PI * 5 * i) / RATE);
    phase += (2 * Math.PI * f0) / RATE;
    let v = 0;
    for (let h = 1; h * 150 < 4000; h += 1) {
      const f = h * 150;
      const gain = formants.reduce((s, [centre, width, g]) => s + g / (1 + ((f - centre) / width) ** 2), 0.02);
      v += gain * Math.sin(h * phase);
    }
    out[i] = v;
  }
  const rms = Math.sqrt(out.reduce((s, x) => s + x * x, 0) / n);
  const chirpRms = Math.sqrt(CHIRP.reduce((s, x) => s + x * x, 0) / n);
  for (let i = 0; i < n; i += 1) out[i] *= chirpRms / rms;
  return fadeEdges(out, RATE, 0.03);
})();

type Mask = "ibm" | "irm";

function magnitude(spec: ComplexSpectrogram): Float32Array {
  const out = new Float32Array(spec.re.length);
  for (let i = 0; i < out.length; i += 1) out[i] = Math.hypot(spec.re[i], spec.im[i]);
  return out;
}

function applyMask(spec: ComplexSpectrogram, mask: Float32Array): ComplexSpectrogram {
  return {
    ...spec,
    re: spec.re.map((v, i) => v * mask[i]),
    im: spec.im.map((v, i) => v * mask[i]),
  };
}

function peakNormalize(x: ArrayLike<number>, peak = 0.9): Float32Array {
  let top = 1e-9;
  for (let i = 0; i < x.length; i += 1) top = Math.max(top, Math.abs(x[i]));
  return Float32Array.from(x, (v) => (v * peak) / top);
}

export default function SeparationPit({ caption }: DemoProps) {
  const [snr, setSnr] = useState(0);
  const [maskKind, setMaskKind] = useState<Mask>("irm");
  const [swapped, setSwapped] = useState(false);
  const { play, stop, playing } = useAudioPlayer();
  // Three STFTs and two inverses take a few tens of ms, so wait for the slider.
  const level = useDebounced(snr, 90);

  const result = useMemo(() => {
    // The chirp's level relative to the vowel, in dB.
    const gain = 10 ** (level / 20);
    const sources = [CHIRP.map((v) => v * gain), VOWEL];
    const mixture = sources[0].map((v, i) => v + sources[1][i]);
    const options = { nFft: N_FFT, hop: HOP, window: "hann" as const };
    const Y = stftComplex(mixture, options);
    const S = sources.map((s) => magnitude(stftComplex(s, options)));
    const masks = [0, 1].map((c) =>
      Float32Array.from(S[c], (a, i) => {
        const b = S[1 - c][i];
        // IBM with a 0 dB local criterion; IRM with beta = 0.5, on powers.
        if (maskKind === "ibm") return a > b ? 1 : 0;
        return Math.sqrt((a * a) / (a * a + b * b + 1e-12));
      }),
    );
    const estimates = masks.map((m) => Float32Array.from(istft(applyMask(Y, m), { length: mixture.length })));
    const ref = Math.max(...magnitude(Y));
    const db = (values: Float32Array) => ({ bins: Y.bins, frames: Y.frames, nFft: N_FFT, hop: HOP, data: toDb(values, { floorDb: FLOOR_DB, ref }) });
    return {
      sources,
      mixture,
      masks,
      estimates,
      mixtureDb: db(magnitude(Y)),
      estimateDb: estimates.map((_, c) => db(magnitude(applyMask(Y, masks[c])))),
      baseline: sources.map((s) => siSdr(mixture, s)),
    };
  }, [level, maskKind]);

  // The separator's outputs in the order it happens to emit them.
  const outputs = swapped ? [result.estimates[1], result.estimates[0]] : result.estimates;
  const outputMasks = swapped ? [result.masks[1], result.masks[0]] : result.masks;
  const outputDb = swapped ? [result.estimateDb[1], result.estimateDb[0]] : result.estimateDb;
  const pairwise = outputs.map((o) => result.sources.map((s) => siSdr(o, s)));
  const fixed = -(pairwise[0][0] + pairwise[1][1]) / 2;
  const crossed = -(pairwise[0][1] + pairwise[1][0]) / 2;
  const pit = Math.min(fixed, crossed);
  const best = fixed <= crossed ? "identity" : "swap";
  const truth = [0, 1].map((c) => siSdr(result.estimates[c], result.sources[c]));

  const sound = (name: string, samples: ArrayLike<number>) =>
    playing === name ? stop() : play(name, peakNormalize(samples), RATE, 0.5);
  const names = ["the chirp", "the vowel"];

  return (
    <DemoFrame
      title="Masks, mixtures, and the permutation problem"
      caption={caption}
      controls={
        <>
          <Slider
            label="Chirp level against the vowel"
            value={snr}
            min={-12}
            max={12}
            format={(v) => `${v > 0 ? "+" : ""}${v} dB`}
            onChange={setSnr}
          />
          <Choice
            label="Oracle mask"
            value={maskKind}
            options={[
              { value: "ibm", label: "Ideal binary" },
              { value: "irm", label: "Ideal ratio" },
            ]}
            onChange={setMaskKind}
          />
          <Actions>
            <Action onClick={() => setSwapped((s) => !s)}>Shuffle the outputs</Action>
          </Actions>
          <Actions>
            <Action tone="quiet" onClick={() => sound("mix", result.mixture)}>
              {playing === "mix" ? "Stop" : "Mixture"}
            </Action>
            <Action tone="quiet" onClick={() => sound("s0", result.sources[0])}>
              {playing === "s0" ? "Stop" : "Chirp"}
            </Action>
            <Action tone="quiet" onClick={() => sound("s1", result.sources[1])}>
              {playing === "s1" ? "Stop" : "Vowel"}
            </Action>
            <Action tone="quiet" onClick={() => sound("o0", outputs[0])}>
              {playing === "o0" ? "Stop" : "Output 1"}
            </Action>
            <Action tone="quiet" onClick={() => sound("o1", outputs[1])}>
              {playing === "o1" ? "Stop" : "Output 2"}
            </Action>
          </Actions>
        </>
      }
    >
      <div className="grid gap-x-4 gap-y-3 sm:grid-cols-2">
        <Panel title="The mixture">
          <Spectrogram spec={result.mixtureDb} sampleRate={RATE} floorDb={FLOOR_DB} rows={ROWS} height={150} description="The mixture's spectrogram" />
        </Panel>
        <Panel title={`Output 1's mask, ${maskKind === "ibm" ? "binary" : "ratio"}`}>
          <HeatmapCanvas
            levels={quantize(outputMasks[0].subarray(0, ROWS[1] * result.mixtureDb.frames), 0, 1)}
            rows={ROWS[1]}
            cols={result.mixtureDb.frames}
            origin="lower"
            colormap="viridis"
            x={{ label: "Time", unit: "s", min: 0, max: (result.mixtureDb.frames * HOP) / RATE }}
            y={{ label: "Frequency", unit: "Hz", min: 0, max: (ROWS[1] - 0.5) * (RATE / N_FFT) }}
            value={{ label: "mask", min: 0, max: 1 }}
            height={150}
            description="The mask for output 1"
          />
        </Panel>
        <Panel title="Output 1: mask × mixture">
          <Spectrogram spec={outputDb[0]} sampleRate={RATE} floorDb={FLOOR_DB} rows={ROWS} height={150} description="Output 1's spectrogram" />
        </Panel>
        <Panel title="Output 2">
          <Spectrogram spec={outputDb[1]} sampleRate={RATE} floorDb={FLOOR_DB} rows={ROWS} height={150} description="Output 2's spectrogram" />
        </Panel>
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <div className="min-w-0">
          <p className="mb-1 text-[12px] font-medium text-ink-3">SI-SDR of each output against each source, dB</p>
          <table className="w-full table-fixed border-collapse font-mono text-[12px] tabular-nums">
            <thead>
              <tr className="text-[11px] text-ink-3">
                <th className="py-1 text-left font-medium" />
                <th className="py-1 text-right font-medium">chirp</th>
                <th className="py-1 text-right font-medium">vowel</th>
              </tr>
            </thead>
            <tbody>
              {pairwise.map((row, i) => (
                <tr key={i} className="border-t border-rule">
                  <td className="py-1 text-[11.5px] text-ink-3">output {i + 1}</td>
                  {row.map((value, j) => {
                    const chosen = best === "identity" ? i === j : i !== j;
                    return (
                      <td key={j} className={`py-1 text-right ${chosen ? "font-semibold text-good" : "text-ink-3"}`}>
                        {value.toFixed(1)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-1 text-[11.5px] leading-snug text-ink-3">Green: the pairing PIT picks.</p>
        </div>
        <div className="min-w-0 text-[13px] leading-snug text-ink-2">
          <p>
            A fixed-order loss compares output 1 with the chirp and output 2 with the vowel:{" "}
            <span className={`font-mono ${fixed > pit + 0.01 ? "text-bad" : "text-ink"}`}>{fixed.toFixed(2)} dB</span>.
          </p>
          <p className="mt-2">
            PIT takes the better of both pairings, min over π of −SI-SDR:{" "}
            <span className="font-mono text-good">{pit.toFixed(2)} dB</span>, the same however the outputs are shuffled.
          </p>
          {swapped && (
            <p className="mt-2 text-ink-3">
              The outputs are shuffled now. Nothing about the separation changed, yet the fixed-order loss says it is
              terrible.
            </p>
          )}
        </div>
      </div>

      <Readouts>
        <Readout label="Chirp estimate" value={`${truth[0].toFixed(1)} dB`} hint={`SI-SDRi +${(truth[0] - result.baseline[0]).toFixed(1)} dB`} tone="good" />
        <Readout label="Vowel estimate" value={`${truth[1].toFixed(1)} dB`} hint={`SI-SDRi +${(truth[1] - result.baseline[1]).toFixed(1)} dB`} tone="good" />
        <Readout label="Fixed-order loss" value={fixed.toFixed(2)} hint="−SI-SDR, dB" tone={fixed > pit + 0.01 ? "bad" : "plain"} />
        <Readout label="PIT loss" value={pit.toFixed(2)} hint={best === "identity" ? "pairs 1–chirp, 2–vowel" : "pairs 1–vowel, 2–chirp"} tone="demo" />
      </Readouts>
      <p className="mt-3 text-[12.5px] leading-snug text-ink-3">
        The mixture alone scores {result.baseline[0].toFixed(1)} dB against {names[0]} and{" "}
        {result.baseline[1].toFixed(1)} dB against {names[1]}; the oracle masks use the true sources, so they are the
        ceiling a trained separator aims for.
      </p>
    </DemoFrame>
  );
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <p className="mb-1 text-[12px] font-medium text-ink-3">{title}</p>
      {children}
    </div>
  );
}
