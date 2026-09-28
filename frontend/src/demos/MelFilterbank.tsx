import { useContext, useMemo, useState } from "react";

import {
  gaussian,
  hann,
  hzToMel,
  isPowerOfTwo,
  linspace,
  melFilterbank,
  melToHz,
  rfftMagnitude,
  seededRandom,
  type MelFormula,
} from "./dsp";
import type { DemoProps } from "./index";
import {
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
  Toggle,
} from "./kit";
import { synthesizeVoice, vowelFormants } from "./synth";

const RATES = ["8000", "16000", "22050", "44100"] as const;
type Rate = (typeof RATES)[number];
const FFTS = ["256", "400", "512", "1024", "2048"] as const;
type Fft = (typeof FFTS)[number];
const FLOOR_DB = -80;

/** The power spectrum of a frame of any length, by FFT when it can and by DFT when it cannot. */
function powerSpectrum(frame: Float64Array): Float64Array {
  const n = frame.length;
  const bins = Math.floor(n / 2) + 1;
  if (isPowerOfTwo(n)) return rfftMagnitude(frame, n).map((m) => m * m);
  const out = new Float64Array(bins);
  for (let k = 0; k < bins; k += 1) {
    let re = 0;
    let im = 0;
    for (let i = 0; i < n; i += 1) {
      const angle = (-2 * Math.PI * k * i) / n;
      re += frame[i] * Math.cos(angle);
      im += frame[i] * Math.sin(angle);
    }
    out[k] = re * re + im * im;
  }
  return out;
}

const kHz = (hz: number) => (hz >= 1000 ? `${+(hz / 1000).toFixed(1)}k` : hz.toFixed(0));
const dB = (power: number, ref: number) => Math.max(FLOOR_DB, 10 * Math.log10(Math.max(1e-20, power / ref)));

export default function MelFilterbank({ caption }: DemoProps) {
  const [rate, setRate] = useState<Rate>("16000");
  const [nFft, setNFft] = useState<Fft>("512");
  const [nMels, setNMels] = useState(26);
  const [fmin, setFmin] = useState(0);
  const [fmax, setFmax] = useState(8000);
  const [formula, setFormula] = useState<MelFormula>("slaney");
  const [areaNorm, setAreaNorm] = useState(true);

  // A printed page is shorter than a screen, so the plots are too.
  const printed = useContext(PrintedFigure) !== null;
  const sampleRate = Number(rate);
  const nyquist = sampleRate / 2;
  const top = Math.min(fmax, nyquist);
  const low = Math.min(fmin, top - 100);
  const n = Number(nFft);

  const vowel = useMemo(() => {
    const voice = synthesizeVoice({
      sampleRate,
      // Long enough for a 2048-sample frame at 8 kHz.
      seconds: 0.3,
      f0: 125,
      formants: vowelFormants(660, 1720, 2410, 3500),
      noise: 0.05,
    });
    // A little room noise, 60 dB down, as any recording has.
    const random = seededRandom(9);
    for (let i = 0; i < voice.length; i += 1) voice[i] += 1e-3 * gaussian(random);
    return voice;
  }, [sampleRate]);

  const spectrum = useMemo(() => {
    const window = hann(n);
    const start = Math.floor((vowel.length - n) / 2);
    const frame = Float64Array.from({ length: n }, (_, i) => vowel[start + i] * window[i]);
    return powerSpectrum(frame);
  }, [vowel, n]);

  const bank = useMemo(
    () =>
      melFilterbank({
        sampleRate,
        nFft: n,
        nMels,
        fmin: low,
        fmax: top,
        formula,
        norm: areaNorm ? "slaney" : null,
      }),
    [sampleRate, n, nMels, low, top, formula, areaNorm],
  );

  const figure = useMemo(() => {
    const bins = spectrum.length;
    const binHz = sampleRate / n;
    const edges = Array.from(linspace(hzToMel(low, formula), hzToMel(top, formula), nMels + 2), (m) =>
      melToHz(m, formula),
    );

    let maxWeight = 0;
    let empty = 0;
    let fewest = Infinity;
    const triangles = bank.map((row) => {
      let first = -1;
      let last = -1;
      let count = 0;
      for (let k = 0; k < bins; k += 1) {
        if (row[k] > 0) {
          if (first < 0) first = k;
          last = k;
          count += 1;
          maxWeight = Math.max(maxWeight, row[k]);
        }
      }
      if (count === 0) empty += 1;
      fewest = Math.min(fewest, count);
      if (first < 0) return [] as [number, number][];
      const points: [number, number][] = [];
      for (let k = Math.max(0, first - 1); k <= Math.min(bins - 1, last + 1); k += 1) points.push([k * binHz, row[k]]);
      return points;
    });

    const energies = bank.map((row) => row.reduce((sum, w, k) => sum + w * spectrum[k], 0));
    const peakPower = Math.max(...spectrum);
    const peakEnergy = Math.max(...energies, 1e-30);
    const spectrumDb = Array.from(spectrum, (p, k): [number, number] => [k * binHz, dB(p, peakPower)]);
    const melDb = energies.map((e, m): [number, number] => [edges[m + 1], dB(e, peakEnergy)]);
    const bars = energies.map((e, m) => ({ x0: m, x1: m + 1, y: dB(e, peakEnergy) - FLOOR_DB }));

    const mapping: [number, number][] = [];
    const other: MelFormula = formula === "slaney" ? "htk" : "slaney";
    const otherMapping: [number, number][] = [];
    const scale = hzToMel(nyquist, formula) / hzToMel(nyquist, other);
    for (let i = 0; i <= 200; i += 1) {
      const hz = (i / 200) * nyquist;
      mapping.push([hz, hzToMel(hz, formula)]);
      otherMapping.push([hz, hzToMel(hz, other) * scale]);
    }
    const edgePoints = edges.map((hz): [number, number] => [hz, hzToMel(hz, formula)]);

    return {
      triangles,
      maxWeight,
      empty,
      fewest,
      spectrumDb,
      melDb,
      bars,
      mapping,
      otherMapping,
      edgePoints,
      lowestWidth: edges[2] - edges[0],
      binHz,
    };
  }, [bank, spectrum, sampleRate, n, nMels, low, top, formula, nyquist]);

  const melTop = hzToMel(nyquist, formula);
  const melAxis = useMemo(() => {
    const raw = melTop / 5;
    const power = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((v) => v >= raw) ?? raw;
    const ticks = Math.ceil(melTop / step);
    return { top: ticks * step, ticks };
  }, [melTop]);
  const guides = nMels <= 40;
  const other = formula === "slaney" ? "HTK" : "Slaney";

  return (
    <DemoFrame
      title="Laying triangles along the mel scale"
      caption={caption}
      controls={
        <>
          <Choice
            label="Sample rate"
            value={rate}
            options={RATES.map((r) => ({ value: r, label: `${+(Number(r) / 1000).toFixed(2)} kHz` }))}
            onChange={(value) => {
              setRate(value);
              setFmax(Number(value) / 2);
              setFmin((f) => Math.min(f, Number(value) / 4));
            }}
          />
          <Choice
            label="n_fft"
            value={nFft}
            options={FFTS.map((f) => ({ value: f, label: f }))}
            onChange={setNFft}
          />
          <Slider label="Mel bands" value={nMels} min={4} max={128} onChange={setNMels} />
          <Choice
            label="Formula"
            value={formula}
            options={[
              { value: "slaney", label: "Slaney (librosa)" },
              { value: "htk", label: "HTK (torchaudio)" },
            ]}
            onChange={setFormula}
          />
          <Slider
            label="fmin"
            value={fmin}
            min={0}
            max={Math.min(1000, nyquist / 2)}
            step={10}
            format={(v) => `${v} Hz`}
            onChange={setFmin}
          />
          <Slider
            label="fmax"
            value={Math.min(fmax, nyquist)}
            min={1000}
            max={nyquist}
            step={50}
            format={(v) => `${v} Hz`}
            onChange={setFmax}
          />
          <Toggle
            label="Slaney area normalization: each triangle's area is the same"
            checked={areaNorm}
            onChange={setAreaNorm}
          />
        </>
      }
    >
      <Plot
        height={printed ? 140 : 170}
        xDomain={[0, nyquist]}
        yDomain={[0, (figure.maxWeight || 1) * 1.08]}
        xLabel="frequency (Hz)"
        yLabel={areaNorm ? "weight, area-normalized" : "weight"}
        xTicks={4}
        yTicks={2}
        tickFormat={(value, axis) =>
          axis === "x" ? kHz(value) : value === 0 ? "0" : value < 0.01 ? value.toExponential(0) : value.toFixed(value >= 1 ? 1 : 2)
        }
      >
        {() => (
          <>
            {figure.triangles.map((points, m) => (
              <Curve
                key={m}
                points={points}
                stroke={m % 2 ? "var(--series-2)" : "var(--demo)"}
                width={1.2}
                fill={m % 2 ? "color-mix(in oklab, var(--series-2) 12%, transparent)" : "color-mix(in oklab, var(--demo) 12%, transparent)"}
              />
            ))}
          </>
        )}
      </Plot>

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <div className="min-w-0">
          <Plot
            width={360}
            height={printed ? 190 : 240}
            xDomain={[0, nyquist]}
            yDomain={[0, melAxis.top]}
            xLabel="frequency (Hz)"
            yLabel="mel"
            xTicks={4}
            yTicks={melAxis.ticks}
            tickFormat={(value, axis) => (axis === "x" ? kHz(value) : value.toFixed(0))}
          >
            {({ sx, sy }) => (
              <>
                {guides &&
                  figure.edgePoints.map(([hz, mel], i) => (
                    <g key={i} stroke="var(--rule-strong)" strokeWidth={0.7}>
                      <line x1={sx(0)} x2={sx(hz)} y1={sy(mel)} y2={sy(mel)} />
                      <line x1={sx(hz)} x2={sx(hz)} y1={sy(mel)} y2={sy(0)} />
                    </g>
                  ))}
                <Curve points={figure.otherMapping} stroke="var(--ink-4)" width={1.2} dashed />
                <Curve points={figure.mapping} width={1.8} />
                <Points points={figure.edgePoints} r={2.4} />
              </>
            )}
          </Plot>
          <Legend
            items={[
              { label: `${formula === "slaney" ? "Slaney" : "HTK"}, with the band edges`, colour: "var(--demo)" },
              { label: `${other}, rescaled to match at ${kHz(nyquist)}`, colour: "var(--ink-4)", dashed: true },
            ]}
          />
        </div>
        <div className="min-w-0">
          <Plot
            width={360}
            height={printed ? 190 : 240}
            xDomain={[0, nMels]}
            yDomain={[0, -FLOOR_DB]}
            xLabel="mel band"
            yLabel="energy (dB)"
            xTicks={Math.min(4, nMels - 1)}
            yTicks={4}
            tickFormat={(value, axis) => (axis === "x" ? value.toFixed(0) : (value + FLOOR_DB).toFixed(0))}
          >
            {() => <Bars bins={figure.bars} gap={nMels > 60 ? 0 : 1} />}
          </Plot>
          <p className="mt-2 text-[12px] leading-snug text-ink-3">
            {printed ? "The same energies, one bar per band." : "The same energies, one bar per band: evenly spaced, the way a model sees them."}
          </p>
        </div>
      </div>

      <div className="mt-4">
        <Plot
          height={printed ? 140 : 170}
          xDomain={[0, nyquist]}
          yDomain={[FLOOR_DB, 0]}
          xLabel="frequency (Hz)"
          yLabel="dB"
          xTicks={4}
          yTicks={4}
          tickFormat={(value, axis) => (axis === "x" ? kHz(value) : value.toFixed(0))}
        >
          {() => (
            <>
              <Curve points={figure.spectrumDb} stroke="var(--ink-4)" width={0.9} />
              <Curve points={figure.melDb} width={1.6} />
              <Points points={figure.melDb} r={2.4} />
            </>
          )}
        </Plot>
      </div>
      <Legend
        items={[
          { label: `one ${n}-sample frame of a vowel, its power spectrum`, colour: "var(--ink-4)" },
          { label: "its mel energies, at each band's centre", colour: "var(--demo)" },
        ]}
      />

      <Readouts>
        <Readout label="Bin spacing" value={`${figure.binHz.toFixed(1)} Hz`} hint="fs / n_fft" />
        <Readout label="Lowest band" value={`${figure.lowestWidth.toFixed(0)} Hz`} hint="wide, edge to edge" />
        <Readout
          label="Fewest bins in a band"
          value={figure.fewest}
          tone={figure.fewest <= 1 ? "bad" : "plain"}
          hint={figure.fewest <= 1 ? "a band samples one bin" : undefined}
        />
        <Readout
          label="Empty bands"
          value={figure.empty}
          tone={figure.empty ? "bad" : "good"}
          hint={figure.empty ? "librosa warns about these" : "none"}
        />
      </Readouts>
    </DemoFrame>
  );
}
