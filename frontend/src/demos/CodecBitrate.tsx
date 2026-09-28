import { useContext, useState } from "react";

import type { DemoProps } from "./index";
import { Choice, DemoFrame, PrintedFigure, Readout, Readouts, Slider } from "./kit";

type Codec = {
  sampleRate: number;
  /** Samples per frame: the encoder's total stride. */
  hop: number;
  /** Codebooks per frame, or per coarsest frame when `strides` is set. */
  codebooks: number;
  size: number;
  /**
   * For a multi-scale codec, how many frames each level spans: SNAC's levels
   * run at 1/4, 1/2, and 1 times the frame rate. Null means every codebook
   * runs at the frame rate.
   */
  strides: number[] | null;
};

type Preset = { id: string; name: string; note: string; codec: Codec };

/**
 * The presets, every number from docs/research/03-tts-vocoders-codecs.md,
 * "Neural audio codecs" and its "Codec comparison" table:
 *
 * - EnCodec 24 kHz: stride 320, so 75 Hz; codebooks of 1024 entries; 8 of
 *   them give 6 kbps and 600 tokens a second (Défossez et al. 2022).
 * - DAC 44.1 kHz: stride 512, so 86.13 Hz; 9 codebooks of 1024, 7,752 bps
 *   (Kumar et al. 2023).
 * - Mimi: 24 kHz, stride 1920, so 12.5 Hz; 8 codebooks of 2048 in Moshi,
 *   1,100 bps and 100 tokens a second (Défossez et al. 2024).
 * - SNAC 24 kHz: hop 512 and VQ strides [4, 2, 1], so levels at 11.72,
 *   23.44, and 46.875 Hz, with 4096 entries each: 82.03 tokens a second and
 *   984 bps (Siuzdak et al. 2024).
 * - WavTokenizer: one codebook of 4096 at 24 kHz and 75 Hz (stride 320),
 *   900 bps (Ji et al. 2025).
 */
const PRESETS: Preset[] = [
  {
    id: "encodec",
    name: "EnCodec 24 kHz",
    note: "at 6 kbps; it runs from 2 to 32 codebooks, 1.5 to 24 kbps",
    codec: { sampleRate: 24000, hop: 320, codebooks: 8, size: 1024, strides: null },
  },
  {
    id: "dac",
    name: "DAC 44.1 kHz",
    note: "9 codebooks at full depth",
    codec: { sampleRate: 44100, hop: 512, codebooks: 9, size: 1024, strides: null },
  },
  {
    id: "mimi",
    name: "Mimi",
    note: "8 codebooks as Moshi uses it; the release has 32",
    codec: { sampleRate: 24000, hop: 1920, codebooks: 8, size: 2048, strides: null },
  },
  {
    id: "snac",
    name: "SNAC 24 kHz",
    note: "three levels at three frame rates",
    codec: { sampleRate: 24000, hop: 512, codebooks: 3, size: 4096, strides: [4, 2, 1] },
  },
  {
    id: "wavtokenizer",
    name: "WavTokenizer",
    note: "a single codebook, at 75 Hz",
    codec: { sampleRate: 24000, hop: 320, codebooks: 1, size: 4096, strides: null },
  },
];

const RATES = [16000, 24000, 32000, 44100, 48000];
const HOPS = [160, 240, 320, 480, 512, 600, 640, 960, 1920, 3840];
const SIZES = [256, 1024, 2048, 4096, 8192, 16384, 65536];
const CONTEXTS = [4096, 8192, 32768, 131072];

function measure(codec: Codec) {
  const frameRate = codec.sampleRate / codec.hop;
  const levelRates = codec.strides ? codec.strides.map((s) => frameRate / s) : new Array(codec.codebooks).fill(frameRate);
  const tokens = levelRates.reduce((a, b) => a + b, 0);
  const bits = tokens * Math.log2(codec.size);
  // One sequence position per frame: the codebooks of a frame share it, as
  // when their embeddings are summed or a depth transformer handles them.
  // For a multi-scale codec, a frame is the coarsest level's.
  const perFrame = codec.strides ? frameRate / Math.max(...codec.strides) : frameRate;
  return { frameRate, tokens, bits, perFrame, pcm: codec.sampleRate * 16 };
}

const hz = (v: number) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(2).replace(/\.?0+$/, "") : v.toFixed(2));
const kbps = (bits: number) => (bits >= 1000 ? `${(bits / 1000).toFixed(bits >= 10000 ? 1 : 2)} kbps` : `${Math.round(bits)} bps`);
const duration = (seconds: number) =>
  seconds >= 3600
    ? `${(seconds / 3600).toFixed(1)} h`
    : seconds >= 60
      ? `${(seconds / 60).toFixed(1)} min`
      : `${seconds.toFixed(0)} s`;

type Metric = "bits" | "tokens" | "minutes";

export default function CodecBitrate({ caption }: DemoProps) {
  const [codec, setCodec] = useState<Codec>(PRESETS[2].codec);
  const [context, setContext] = useState(4096);
  const [counting, setCounting] = useState<"tokens" | "frames">("tokens");
  const [metric, setMetric] = useState<Metric>("minutes");
  const printed = useContext(PrintedFigure);

  const preset = PRESETS.find((p) => JSON.stringify(p.codec) === JSON.stringify(codec));
  const m = measure(codec);
  const perSecond = counting === "tokens" ? m.tokens : m.perFrame;
  const fits = context / perSecond;

  const edit = (change: Partial<Codec>) => setCodec((c) => ({ ...c, ...change }));

  const rows = [
    ...PRESETS.map((p) => ({ id: p.id, name: p.name, codec: p.codec })),
    ...(preset ? [] : [{ id: "yours", name: "Your settings", codec }]),
  ].map((row) => {
    const r = measure(row.codec);
    const rate = counting === "tokens" ? r.tokens : r.perFrame;
    return { ...row, bits: r.bits, tokens: r.tokens, minutes: context / rate / 60 };
  });
  const top = Math.max(...rows.map((r) => r[metric]));
  const format = (row: (typeof rows)[number]) =>
    metric === "bits" ? kbps(row.bits) : metric === "tokens" ? `${hz(row.tokens)}/s` : duration(row.minutes * 60);

  return (
    <DemoFrame
      title="Frames, codebooks, and bits per second"
      caption={caption}
      controls={
        <>
          <Choice
            label="Start from"
            value={preset?.id ?? "custom"}
            options={[
              ...PRESETS.map((p) => ({ value: p.id, label: p.name })),
              ...(preset ? [] : [{ value: "custom", label: "Your settings" }]),
            ]}
            onChange={(id) => {
              const found = PRESETS.find((p) => p.id === id);
              if (found) setCodec(found.codec);
            }}
          />
          <Choice
            label="Sample rate"
            value={String(codec.sampleRate)}
            options={RATES.map((r) => ({ value: String(r), label: `${r / 1000} kHz` }))}
            onChange={(v) => edit({ sampleRate: Number(v) })}
          />
          <Slider
            label="Hop, the encoder's total stride"
            value={Math.max(0, HOPS.indexOf(codec.hop))}
            min={0}
            max={HOPS.length - 1}
            format={() => `${codec.hop} samples, ${hz(m.frameRate)} Hz`}
            onChange={(i) => edit({ hop: HOPS[i] })}
          />
          <Slider
            label="Codebooks per frame"
            value={codec.codebooks}
            min={1}
            max={32}
            // A multi-scale codec's levels are fixed, so more codebooks means a plain one.
            onChange={(v) => edit({ codebooks: v, strides: null })}
          />
          <Choice
            label="Codebook size"
            value={String(codec.size)}
            options={SIZES.map((s) => ({ value: String(s), label: s >= 1024 ? `${s / 1024}k` : String(s) }))}
            onChange={(v) => edit({ size: Number(v) })}
          />
          <Choice
            label="LLM context"
            value={String(context)}
            options={CONTEXTS.map((c) => ({ value: String(c), label: `${c / 1024}k` }))}
            onChange={(v) => setContext(Number(v))}
          />
          <Choice
            label="The context holds"
            value={counting}
            options={[
              { value: "tokens", label: "every token, flattened" },
              { value: "frames", label: "one position per frame" },
            ]}
            onChange={setCounting}
          />
        </>
      }
    >
      <p className="font-mono text-[12.5px] leading-relaxed text-ink-2">
        {codec.strides ? (
          <>
            tokens/s = {codec.strides.map((s) => hz(m.frameRate / s)).join(" + ")} = {hz(m.tokens)}
            <br />
            bits/s = {hz(m.tokens)} × log2 {codec.size.toLocaleString("en-US")} = {Math.round(m.bits).toLocaleString("en-US")}
          </>
        ) : (
          <>
            bits/s = (fs / hop) × codebooks × log2 K = ({codec.sampleRate.toLocaleString("en-US")} / {codec.hop}) ×{" "}
            {codec.codebooks} × {Math.log2(codec.size)} = {Math.round(m.bits).toLocaleString("en-US")}
          </>
        )}
      </p>
      {preset && <p className="mt-1 text-[12px] text-ink-3">{preset.name}, {preset.note}.</p>}

      <Readouts>
        <Readout label="Frame rate" value={`${hz(m.frameRate)} Hz`} hint={`${((1000 * codec.hop) / codec.sampleRate).toFixed(1)} ms a frame`} />
        <Readout label="Tokens a second" value={hz(m.tokens)} hint={counting === "frames" ? `${hz(m.perFrame)} positions a second` : "one position each"} />
        <Readout label="Bitrate" value={kbps(m.bits)} hint={`${Math.round(m.pcm / m.bits)}× smaller than 16-bit PCM`} tone="demo" />
        <Readout label={`In ${context / 1024}k positions`} value={duration(fits)} hint="of audio, with no text" />
      </Readouts>

      <div className="mt-5">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <p className="text-[12px] font-medium text-ink-3">
            {metric === "bits"
              ? "Bitrate"
              : metric === "tokens"
                ? "Tokens a second"
                : `Audio that fits in ${context / 1024}k positions, ${counting === "tokens" ? "flattened" : "one per frame"}`}
          </p>
          {!printed && (
          <Choice
            label="Compare"
            value={metric}
            options={[
              { value: "minutes", label: "Context" },
              { value: "tokens", label: "Tokens/s" },
              { value: "bits", label: "Bitrate" },
            ]}
            onChange={setMetric}
          />
          )}
        </div>
        <div className="grid gap-1.5">
          {rows.map((row) => {
            const active = (preset && preset.id === row.id) || row.id === "yours";
            return (
              <div key={row.id} className="grid grid-cols-[7.5rem_minmax(0,1fr)] items-center gap-2 text-[12px]">
                <span className={active ? "font-medium text-ink" : "text-ink-3"}>{row.name}</span>
                <div className="flex min-w-0 items-center gap-2">
                  <div
                    className="h-3.5 shrink-0 rounded-sm"
                    style={{
                      width: `${Math.max(1, (row[metric] / top) * 72)}%`,
                      background: active ? "var(--demo)" : "color-mix(in oklab, var(--demo) 35%, transparent)",
                    }}
                  />
                  <span className="whitespace-nowrap font-mono text-[11.5px] text-ink-2">{format(row)}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </DemoFrame>
  );
}
