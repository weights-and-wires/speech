import { useContext, useMemo, useState, type ReactNode } from "react";

import { dct2, fadeEdges, hamming, hzToMel, idct2, linspace, melFilterbank, melToHz, preEmphasis, rfftMagnitude } from "./dsp";
import type { DemoProps } from "./index";
import {
  Action,
  Actions,
  Bars,
  Choice,
  Curve,
  DemoFrame,
  Legend,
  Plot,
  Points,
  PrintedFigure,
  Readout,
  Readouts,
  Slider,
  Stems,
  Toggle,
  useAudioPlayer,
} from "./kit";
import { synthesizeVoice, vowelFormants } from "./synth";

const RATE = 16000;
/** 25 ms frames in a 512-point FFT, as HTK and Kaldi frame speech. */
const FRAME = 400;
const N_FFT = 512;
const N_MELS = 26;
const LIFTER = 22;
const ALPHA = 0.97;

const VOICES = {
  a: { label: "/ɑ/ hod", f1: 730, f2: 1090, f3: 2440 },
  i: { label: "/i/ heed", f1: 270, f2: 2290, f3: 3010 },
  u: { label: "/u/ who'd", f1: 300, f2: 870, f3: 2240 },
} as const;
type VoiceName = keyof typeof VOICES;

const BANK = melFilterbank({ sampleRate: RATE, nFft: N_FFT, nMels: N_MELS, formula: "htk", norm: null });
const CENTRES = Array.from(linspace(hzToMel(0, "htk"), hzToMel(RATE / 2, "htk"), N_MELS + 2), (m) => melToHz(m, "htk")).slice(
  1,
  -1,
);
const WINDOW = hamming(FRAME);
/** Each triangle's total weight, to turn a band's energy into power per bin. */
const LOG_AREAS = BANK.map((row) => Math.log(row.reduce((sum, w) => sum + w, 0)));

const ms = (i: number) => (i / RATE) * 1000;
const kHz = (hz: number) => (hz >= 1000 ? `${+(hz / 1000).toFixed(1)}k` : hz.toFixed(0));

function Stage({ step, title, children }: { step: number; title: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <p className="text-[12.5px] font-medium text-ink-2">
        <span className="mr-1.5 font-mono text-demo">{step}</span>
        {title}
      </p>
      <div className="mt-1">{children}</div>
    </div>
  );
}

export default function MfccPipeline({ caption }: DemoProps) {
  const [kept, setKept] = useState(13);
  const [voiceName, setVoiceName] = useState<VoiceName>("a");
  const [f0, setF0] = useState(120);
  const [emphasis, setEmphasis] = useState(true);
  const [lifter, setLifter] = useState(true);
  const { play, stop, playing } = useAudioPlayer();

  const voice = VOICES[voiceName];
  const sound = useMemo(
    () =>
      synthesizeVoice({
        sampleRate: RATE,
        seconds: 0.6,
        f0,
        formants: vowelFormants(voice.f1, voice.f2, voice.f3),
      }),
    [voice, f0],
  );

  const stages = useMemo(() => {
    const start = Math.floor((sound.length - FRAME) / 2);
    const raw = sound.slice(start, start + FRAME);
    const emphasised = emphasis ? preEmphasis(raw, ALPHA) : Float64Array.from(raw);
    const windowed = emphasised.map((x, i) => x * WINDOW[i]);
    const magnitude = rfftMagnitude(windowed, N_FFT);
    const power = magnitude.map((m) => m * m);
    const energies = BANK.map((row) => row.reduce((sum, w, k) => sum + w * power[k], 0));
    const logs = energies.map((e) => Math.log(Math.max(1e-10, e)));
    const cepstrum = dct2(logs);
    const liftered = cepstrum.map((c, n) => c * (1 + (LIFTER / 2) * Math.sin((Math.PI * n) / LIFTER)));
    return { raw, emphasised, windowed, power, energies, logs, cepstrum, liftered };
  }, [sound, emphasis]);

  const smooth = useMemo(() => idct2(stages.cepstrum.slice(0, kept), "ortho", N_MELS), [stages, kept]);

  const plots = useMemo(() => {
    const { raw, emphasised, windowed, power, energies, logs, cepstrum, liftered } = stages;
    const time = (x: ArrayLike<number>) => Array.from(x, (v, i): [number, number] => [ms(i), v]);
    const peak = (x: ArrayLike<number>) => Math.max(1e-12, ...Array.from(x, Math.abs));
    const rawPeak = peak(raw);
    const emphasisPeak = peak(emphasised);
    const windowedPeak = peak(windowed);
    const powerPeak = Math.max(...power);
    const spectrumDb = Array.from(power, (p, k): [number, number] => [
      (k * RATE) / N_FFT,
      Math.max(-80, 10 * Math.log10(Math.max(1e-20, p / powerPeak))),
    ]);
    const energyPeak = Math.max(...energies);
    // Per bin, so a wide band is not drawn louder just for being wide.
    const perBin = (values: ArrayLike<number>) => Array.from(values, (v, m) => v - LOG_AREAS[m]);
    const logPeak = Math.max(...perBin(logs));
    const toDbAtCentres = (values: ArrayLike<number>) =>
      perBin(values).map((v, m): [number, number] => [CENTRES[m], Math.max(-80, (10 / Math.LN10) * (v - logPeak))]);
    return {
      raw: time(Array.from(raw, (v) => v / rawPeak)),
      emphasised: time(Array.from(emphasised, (v) => v / emphasisPeak)),
      windowed: time(Array.from(windowed, (v) => v / windowedPeak)),
      window: time(WINDOW),
      spectrumDb,
      energyBars: Array.from(energies, (e, m) => ({ x0: m - 0.4, x1: m + 0.4, y: e / energyPeak })),
      logs: Array.from(logs, (v, m): [number, number] => [m, v]),
      smooth: Array.from(smooth, (v, m): [number, number] => [m, v]),
      cepstrum: Array.from(cepstrum, (c, n): [number, number] => [n, c]),
      liftered: Array.from(liftered, (c, n): [number, number] => [n, c]),
      logsHz: toDbAtCentres(logs),
      smoothHz: toDbAtCentres(smooth),
    };
  }, [stages, smooth]);

  const logRange: [number, number] = useMemo(() => {
    const values = [...plots.logs, ...plots.smooth].map((p) => p[1]);
    // Even bounds four whole steps apart, so the four ticks land on integers.
    const lo = Math.floor((Math.min(...values) - 0.5) / 2) * 2;
    let hi = Math.ceil((Math.max(...values) + 0.5) / 2) * 2;
    if ((hi - lo) % 4) hi += 2;
    return [lo, hi];
  }, [plots]);
  const cepRange = Math.max(1, ...plots.cepstrum.slice(1).map((p) => Math.abs(p[1]))) * 1.15;
  const lifRange = Math.max(1, ...plots.liftered.slice(1).map((p) => Math.abs(p[1]))) * 1.15;
  const keptC = (points: [number, number][]) => points.slice(1, kept);
  const droppedC = (points: [number, number][]) => points.slice(kept);

  let residual = 0;
  for (let m = 0; m < N_MELS; m += 1) residual += (stages.logs[m] - smooth[m]) ** 2;
  residual = Math.sqrt(residual / N_MELS);

  // A printed page is shorter than a screen: shorter plots, and no readouts.
  const printed = useContext(PrintedFigure) !== null;
  const small = { width: 360, height: printed ? 130 : 180 };

  return (
    <DemoFrame
      title="From a frame of speech to thirteen numbers"
      caption={caption}
      controls={
        <>
          <Slider label="Coefficients kept" value={kept} min={1} max={N_MELS} onChange={setKept} />
          <Choice
            label="Vowel"
            value={voiceName}
            options={(Object.keys(VOICES) as VoiceName[]).map((v) => ({ value: v, label: VOICES[v].label }))}
            onChange={setVoiceName}
          />
          <Slider label="Pitch, F0" value={f0} min={80} max={260} step={5} format={(v) => `${v} Hz`} onChange={setF0} />
          <div className="flex flex-col gap-2">
            <Toggle label={`Pre-emphasis, α = ${ALPHA}`} checked={emphasis} onChange={setEmphasis} />
            <Toggle label={`Sinusoidal lifter, L = ${LIFTER}`} checked={lifter} onChange={setLifter} />
          </div>
          <Actions>
            <Action onClick={playing ? stop : () => play("vowel", fadeEdges(Float32Array.from(sound), RATE, 0.03), RATE, 0.5)}>
              {playing ? "Stop" : "Play the vowel"}
            </Action>
          </Actions>
        </>
      }
    >
      <div className="grid gap-x-5 gap-y-4 sm:grid-cols-2">
        <Stage step={1} title={emphasis ? "Frame, and pre-emphasized" : "Frame"}>
          <Plot {...small} xDomain={[0, ms(FRAME)]} yDomain={[-1.1, 1.1]} xLabel="time (ms)" yTicks={2} xTicks={5} tickFormat={(v) => v.toFixed(0)}>
            {() => (
              <>
                <Curve points={plots.raw} stroke="var(--ink-4)" width={1} />
                {emphasis && <Curve points={plots.emphasised} width={1.1} />}
              </>
            )}
          </Plot>
        </Stage>

        <Stage step={2} title="Hamming window">
          <Plot {...small} xDomain={[0, ms(FRAME)]} yDomain={[-1.1, 1.1]} xLabel="time (ms)" yTicks={2} xTicks={5} tickFormat={(v) => v.toFixed(0)}>
            {() => (
              <>
                <Curve points={plots.window} stroke="var(--ink-4)" width={1} dashed />
                <Curve points={plots.windowed} width={1.1} />
              </>
            )}
          </Plot>
        </Stage>

        <Stage step={3} title="Power spectrum, 512-point FFT">
          <Plot {...small} xDomain={[0, RATE / 2]} yDomain={[-80, 0]} xLabel="frequency (Hz)" yLabel="dB" yTicks={4} xTicks={4} tickFormat={(v, a) => (a === "x" ? kHz(v) : v.toFixed(0))}>
            {() => <Curve points={plots.spectrumDb} width={1} />}
          </Plot>
        </Stage>

        <Stage step={4} title={`${N_MELS} mel energies (HTK triangles)`}>
          <Plot {...small} xDomain={[0, N_MELS - 1]} yDomain={[0, 1.08]} xLabel="mel band" yLabel="relative power" yTicks={2} xTicks={5} tickFormat={(v) => v.toFixed(v < 1 && v > 0 ? 1 : 0)}>
            {() => <Bars bins={plots.energyBars} />}
          </Plot>
        </Stage>

        <Stage step={5} title="Log, and the curve the kept coefficients rebuild">
          <Plot {...small} xDomain={[0, N_MELS - 1]} yDomain={logRange} xLabel="mel band" yLabel="ln energy" yTicks={4} xTicks={5} tickFormat={(v) => v.toFixed(0)}>
            {() => (
              <>
                <Curve points={plots.logs} stroke="var(--ink-4)" width={1} />
                <Points points={plots.logs} r={2.2} fill="var(--ink-3)" />
                <Curve points={plots.smooth} stroke="var(--bad)" width={1.8} />
              </>
            )}
          </Plot>
        </Stage>

        <Stage step={6} title="DCT-II, orthonormal: the cepstrum">
          <Plot {...small} xDomain={[0, N_MELS - 1]} yDomain={[-cepRange, cepRange]} xLabel="coefficient" yTicks={2} xTicks={5} tickFormat={(v) => v.toFixed(0)}>
            {({ sx, sy }) => (
              <>
                <Stems points={droppedC(plots.cepstrum)} colour="var(--ink-4)" r={2} />
                <Stems points={keptC(plots.cepstrum)} r={2.4} />
                <text x={sx(N_MELS - 1)} y={sy(cepRange) + 11} textAnchor="end" fontSize={10} fontFamily="var(--font-mono)" fill="var(--ink-3)">
                  c0 = {stages.cepstrum[0].toFixed(1)}, off the scale
                </text>
              </>
            )}
          </Plot>
        </Stage>

        {lifter && (
          <Stage step={7} title={`Liftered, 1 + (L/2) sin(πn/L)`}>
            <Plot {...small} xDomain={[0, N_MELS - 1]} yDomain={[-lifRange, lifRange]} xLabel="coefficient" yTicks={2} xTicks={5} tickFormat={(v) => v.toFixed(0)}>
              {() => (
                <>
                  <Stems points={droppedC(plots.liftered)} colour="var(--ink-4)" r={2} />
                  <Stems points={keptC(plots.liftered)} r={2.4} />
                </>
              )}
            </Plot>
          </Stage>
        )}

        <Stage step={lifter ? 8 : 7} title="The envelope, per bin, on the frequency axis">
          <Plot {...small} xDomain={[0, RATE / 2]} yDomain={[-80, 0]} xLabel="frequency (Hz)" yLabel="dB" yTicks={4} xTicks={4} tickFormat={(v, a) => (a === "x" ? kHz(v) : v.toFixed(0))}>
            {() => (
              <>
                <Curve points={plots.spectrumDb} stroke="var(--ink-4)" width={0.8} />
                <Points points={plots.logsHz} r={2} fill="var(--ink-3)" />
                <Curve points={plots.smoothHz} stroke="var(--bad)" width={1.8} />
              </>
            )}
          </Plot>
        </Stage>
      </div>
      <Legend
        items={[
          { label: `the ${kept} kept coefficients`, colour: "var(--demo)" },
          { label: "dropped", colour: "var(--ink-4)" },
          { label: `the log-mel curve rebuilt from ${kept} coefficient${kept === 1 ? "" : "s"}`, colour: "var(--bad)" },
        ]}
      />

      {!printed && (
        <Readouts>
          <Readout label="MFCCs" value={kept} hint={`of ${N_MELS} mel bands`} tone="demo" />
          <Readout label="c0" value={stages.cepstrum[0].toFixed(2)} hint="the frame's log energy, scaled" />
          <Readout label="c1" value={stages.cepstrum[1].toFixed(2)} hint="the spectral tilt" />
          <Readout label="Rebuild error" value={residual.toFixed(2)} hint="RMS, in ln units" />
        </Readouts>
      )}
    </DemoFrame>
  );
}
