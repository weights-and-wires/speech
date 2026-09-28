import { useState } from "react";

import { useColumnWidth } from "./figure";
import type { DemoProps } from "./index";
import { DemoFrame, Readout, Readouts, Slider, Toggle } from "./kit";

/** Moshi's frame: Mimi runs at 12.5 Hz (Défossez et al., 2024). */
const FRAME_MS = 80;
/** The average gap between turns in human conversation (Stivers et al., 2009). */
const HUMAN_GAP_MS = 200;
/** How long the user's turn lasts, for an ASR that waits for all of it. */
const UTTERANCE_MS = 1800;
/** The real-time factor of a non-streaming ASR, as a stand-in. */
const BATCH_RTF = 0.2;
/** Text the LLM streams before the TTS has a phrase worth speaking. */
const FIRST_PHRASE_MS = 120;
/** How much of the user's speech the chart shows before they stop. */
const LEAD_MS = 640;

type Bar = { from: number; to: number; tone: "demo" | "ink" | "net" | "wait" | "user" | "out"; thin?: boolean };
type Row = { label: string; bars: Bar[] };

const TONES: Record<Bar["tone"], string> = {
  demo: "var(--demo)",
  ink: "var(--ink-3)",
  net: "var(--series-4)",
  wait: "var(--warn)",
  user: "var(--series-3)",
  out: "var(--series-2)",
};

export default function LatencyTimeline({ caption }: DemoProps) {
  // Illustrative defaults for a hosted cascade; they are sliders, not measurements.
  const [silence, setSilence] = useState(500);
  const [streaming, setStreaming] = useState(true);
  const [asrFinal, setAsrFinal] = useState(150);
  const [ttft, setTtft] = useState(400);
  const [ttsFirst, setTtsFirst] = useState(200);
  const [rtt, setRtt] = useState(100);
  const [compute, setCompute] = useState(40);
  const [tau, setTau] = useState(1);

  const up = rtt / 2;
  // The cascade, in ms after the user stops speaking.
  const endpoint = up + silence;
  const asrDone = endpoint + (streaming ? asrFinal : asrFinal + BATCH_RTF * UTTERANCE_MS);
  const firstToken = asrDone + ttft;
  const ttsStart = firstToken + FIRST_PHRASE_MS;
  const firstAudioOut = ttsStart + ttsFirst;
  const cascade = firstAudioOut + up;

  // The full-duplex model: the first frame of silence has to finish and
  // reach it, one step computes, and the acoustic tokens trail the semantic
  // one by tau frames. With no network and no compute that is Moshi's
  // theoretical 160 ms: one 80 ms frame plus one frame of acoustic delay.
  const lastFrameEnd = up + FRAME_MS;
  const decided = lastFrameEnd + compute;
  const duplexOut = decided + tau * FRAME_MS;
  const duplex = duplexOut + up;
  const realtime = compute < FRAME_MS;

  const end = Math.max(cascade, duplex) + 500;
  const rows: { title: string; total: number; rows: Row[] }[] = [
    {
      title: "Cascade: VAD, ASR, LLM, and TTS",
      total: cascade,
      rows: [
        { label: "user speaks", bars: [{ from: -LEAD_MS, to: 0, tone: "user" }] },
        { label: "network up", bars: [{ from: 0, to: up, tone: "net" }] },
        { label: `VAD waits ${silence} ms of silence`, bars: [{ from: up, to: endpoint, tone: "wait" }] },
        {
          label: streaming ? "streaming ASR, then its final result" : "ASR on the whole turn, after the endpoint",
          bars: streaming
            ? [
                { from: -LEAD_MS, to: endpoint, tone: "ink", thin: true },
                { from: endpoint, to: asrDone, tone: "demo" },
              ]
            : [{ from: endpoint, to: asrDone, tone: "demo" }],
        },
        {
          label: "LLM to its first token, then streaming",
          bars: [
            { from: asrDone, to: firstToken, tone: "demo" },
            { from: firstToken, to: end, tone: "demo", thin: true },
          ],
        },
        {
          label: "TTS to its first audio, after a phrase of text",
          bars: [
            { from: firstToken, to: ttsStart, tone: "ink", thin: true },
            { from: ttsStart, to: firstAudioOut, tone: "demo" },
          ],
        },
        {
          label: "network down, then the reply plays",
          bars: [
            { from: firstAudioOut, to: cascade, tone: "net" },
            { from: cascade, to: end, tone: "out" },
          ],
        },
      ],
    },
    {
      title: `Full duplex: one model step every ${FRAME_MS} ms`,
      total: duplex,
      rows: [
        {
          label: "user audio, a frame at a time, as it reaches the model",
          bars: frameStarts(-LEAD_MS, FRAME_MS).map((start) => ({
            from: start + up,
            to: start + FRAME_MS + up - 3,
            tone: start < 0 ? ("user" as const) : ("ink" as const),
          })),
        },
        {
          label: `a model step once each frame is in, ${compute} ms each`,
          bars: frameStarts(-LEAD_MS, end).map((start) => ({
            from: start + FRAME_MS + up,
            to: start + FRAME_MS + up + Math.min(compute, FRAME_MS - 4),
            tone: "demo" as const,
          })),
        },
        {
          label: `the model's own stream, acoustic tokens ${tau * FRAME_MS} ms behind`,
          bars: [
            { from: -LEAD_MS, to: duplex, tone: "ink", thin: true },
            { from: decided, to: duplexOut, tone: "wait" },
          ],
        },
        {
          label: "network down, then the reply plays",
          bars: [
            { from: duplexOut, to: duplex, tone: "net" },
            { from: duplex, to: end, tone: "out" },
          ],
        },
      ],
    },
  ];

  return (
    <DemoFrame
      title="Where the time goes in a voice agent"
      caption={caption}
      controls={
        <>
          <Slider label="VAD silence before the endpoint" value={silence} min={100} max={1200} step={50} format={(v) => `${v} ms`} onChange={setSilence} />
          <Slider label="ASR final result" value={asrFinal} min={20} max={500} step={10} format={(v) => `${v} ms`} onChange={setAsrFinal} />
          <Slider label="LLM time to first token" value={ttft} min={50} max={1500} step={25} format={(v) => `${v} ms`} onChange={setTtft} />
          <Slider label="TTS time to first audio" value={ttsFirst} min={30} max={800} step={10} format={(v) => `${v} ms`} onChange={setTtsFirst} />
          <Slider label="Network round trip" value={rtt} min={0} max={400} step={10} format={(v) => `${v} ms`} onChange={setRtt} />
          <Toggle label="Streaming ASR, which transcribes while the user talks" checked={streaming} onChange={setStreaming} />
          <Slider label="Duplex compute per frame" value={compute} min={10} max={120} step={5} format={(v) => `${v} ms`} onChange={setCompute} />
          <Slider label="Acoustic delay, τ" value={tau} min={0} max={2} format={(v) => `${v} ${v === 1 ? "frame" : "frames"}`} onChange={setTau} />
        </>
      }
    >
      <Gantt sections={rows} from={-LEAD_MS} to={end} />

      <Readouts>
        <Readout label="Cascade, first audio" value={`${Math.round(cascade)} ms`} hint="after the user stops" tone={cascade > 1000 ? "bad" : "plain"} />
        <Readout label="Duplex, first audio" value={`${Math.round(duplex)} ms`} hint={`${FRAME_MS} + ${compute} + ${tau * FRAME_MS} + network`} tone="demo" />
        <Readout label="Human gap" value={`${HUMAN_GAP_MS} ms`} hint="the average between turns" />
        <Readout
          label="Duplex keeps up"
          value={realtime ? "yes" : "no"}
          hint={`${compute} ms of compute per ${FRAME_MS} ms frame`}
          tone={realtime ? "good" : "bad"}
        />
      </Readouts>
      <p className="mt-3 text-[12.5px] leading-snug text-ink-3">
        The VAD's wait is the cascade's largest fixed cost: it cannot tell a pause from the end of a turn without
        waiting. The duplex model never takes turns; it emits a token every frame and learns when to speak. With no
        network and no compute, its floor is the frame plus the acoustic delay: {FRAME_MS + tau * FRAME_MS} ms.
      </p>
    </DemoFrame>
  );
}

/** The start of every frame, aligned to 0, from before `from` up to `to`. */
function frameStarts(from: number, to: number): number[] {
  const out: number[] = [];
  for (let start = Math.floor(from / FRAME_MS) * FRAME_MS; start < to; start += FRAME_MS) out.push(start);
  return out;
}

/** A halo in the card's colour, so a label stays readable over the lines. */
const HALO = { paintOrder: "stroke", stroke: "var(--raise)", strokeWidth: 3, strokeLinejoin: "round" } as const;

const ROW_H = 30;
const HEAD_H = 38;
const AXIS_H = 26;

function Gantt({ sections, from, to }: { sections: { title: string; total: number; rows: Row[] }[]; from: number; to: number }) {
  const [box, width] = useColumnWidth();
  const left = 6;
  const right = 10;
  const x = (ms: number) => left + ((Math.max(from, Math.min(to, ms)) - from) / (to - from)) * (width - left - right);
  const TOP = 20;
  const height = sections.reduce((h, s) => h + HEAD_H + s.rows.length * ROW_H, 0) + AXIS_H + TOP;
  const stepMs = (to - from) / Math.max(1, width / 70) > 250 ? 500 : 250;
  const ticks: number[] = [];
  for (let t = 0; t <= to; t += stepMs) ticks.push(t);
  for (let t = -stepMs; t >= from; t -= stepMs) ticks.unshift(t);

  let y = TOP;
  const layout = sections.map((section) => {
    const top = y;
    y += HEAD_H + section.rows.length * ROW_H;
    return { section, top };
  });
  const bottom = height - AXIS_H;

  return (
    <div ref={box} className="w-full min-w-0">
      {width > 0 && (
        <svg width={width} height={height} role="img" aria-label="Latency timelines of a cascaded voice agent and a full-duplex model">
          {ticks.map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={TOP} y2={bottom} stroke="var(--rule)" strokeWidth={1} />
              <text x={x(t)} y={bottom + 14} textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--ink-3)">
                {t === 0 ? "0" : `${t > 0 ? "+" : ""}${t}`}
              </text>
            </g>
          ))}
          <text x={(left + width - right) / 2} y={height - 2} textAnchor="middle" fontSize={10} fill="var(--ink-3)">
            ms after the user stops speaking
          </text>

          <line x1={x(0)} x2={x(0)} y1={TOP - 6} y2={bottom} stroke="var(--ink-2)" strokeWidth={1.2} />
          <text x={x(0) - 3} y={TOP - 9} textAnchor="end" fontSize={10} fill="var(--ink-2)">
            user stops
          </text>
          <line x1={x(HUMAN_GAP_MS)} x2={x(HUMAN_GAP_MS)} y1={TOP - 6} y2={bottom} stroke="var(--good)" strokeWidth={1.2} strokeDasharray="4 3" />
          <text x={x(HUMAN_GAP_MS) + 3} y={TOP - 9} fontSize={10} fill="var(--good)">
            human gap, {HUMAN_GAP_MS} ms
          </text>

          {layout.map(({ section, top }) => (
            <g key={section.title}>
              <text x={left} y={top + 14} fontSize={12} fontWeight={600} fill="var(--ink)" {...HALO}>
                {section.title}
              </text>
              {section.rows.map((row, r) => {
                const rowTop = top + HEAD_H + r * ROW_H;
                return (
                  <g key={row.label}>
                    <text x={left} y={rowTop + 9} fontSize={10.5} fill="var(--ink-3)" {...HALO}>
                      {row.label}
                    </text>
                    {row.bars.map((bar, b) => {
                      const x1 = x(bar.from);
                      const x2 = x(bar.to);
                      if (x2 - x1 < 0.5) return null;
                      const h = bar.thin ? 3 : 10;
                      return (
                        <rect
                          key={b}
                          x={x1}
                          y={rowTop + 13 + (10 - h) / 2}
                          width={x2 - x1}
                          height={h}
                          rx={bar.thin ? 1.5 : 2.5}
                          fill={TONES[bar.tone]}
                          opacity={bar.thin ? 0.55 : 0.9}
                        />
                      );
                    })}
                  </g>
                );
              })}
              <line
                x1={x(section.total)}
                x2={x(section.total)}
                y1={top + HEAD_H - 4}
                y2={top + HEAD_H + section.rows.length * ROW_H}
                stroke="var(--series-2)"
                strokeWidth={1.6}
              />
              <text
                x={width - right}
                y={top + 29}
                textAnchor="end"
                fontSize={11}
                fontFamily="var(--font-mono)"
                fill="var(--series-2)"
                {...HALO}
              >
                first audio at {Math.round(section.total)} ms
              </text>
            </g>
          ))}
        </svg>
      )}
    </div>
  );
}
