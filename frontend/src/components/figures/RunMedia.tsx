import { useMemo } from "react";

import type { AudioEvent, HeatmapEvent, LineEvent, MediaEvent } from "../../lib/api";
import { isColormap } from "../../lib/colormap";
import { decodeBase64 } from "../../lib/plotting";
import AudioClip from "./AudioClip";
import HeatmapCanvas from "./HeatmapCanvas";
import LineCanvas from "./LineCanvas";

/**
 * One `audio` or `figure` event from a run, in the console, where it arrived.
 *
 * Each sits in a card with its label, what it is, the check that produced it
 * in check mode, and any note `speechlab` attached, such as a clip that was cut
 * at its length limit.
 */
export default function RunMedia({ event }: { event: MediaEvent }) {
  return (
    <figure className="fade-in my-2 min-w-0 overflow-hidden rounded-lg bg-raise px-3 py-2.5 font-sans shadow-[inset_0_0_0_1px_var(--rule)]">
      <figcaption className="mb-2 flex min-w-0 flex-wrap items-baseline gap-x-2 gap-y-0.5">
        <span className="min-w-0 truncate text-[13px] font-medium text-ink">{event.label ?? defaultTitle(event)}</span>
        <span className="text-[11.5px] tabular-nums text-ink-4">{describe(event)}</span>
        {event.check != null && <span className="text-[11.5px] text-ink-4">from check {event.check + 1}</span>}
      </figcaption>
      {event.type === "audio" ? (
        <AudioClip wavBase64={event.wav_base64} label={event.label ?? "clip"} />
      ) : event.kind === "heatmap" ? (
        <Heatmap event={event} />
      ) : (
        <Line event={event} />
      )}
      {event.note && <p className="mt-1.5 text-[12px] leading-snug text-warn">{event.note}</p>}
    </figure>
  );
}

function defaultTitle(event: MediaEvent): string {
  if (event.type === "audio") return "Audio";
  return event.kind === "heatmap" ? "Heatmap" : "Plot";
}

function describe(event: MediaEvent): string {
  if (event.type === "audio") return audioSummary(event);
  if (event.kind === "heatmap") return `${event.shape[0]} × ${event.shape[1]}`;
  const points = event.series.reduce((sum, s) => sum + s.points, 0);
  return `${event.series.length} series, ${points.toLocaleString()} points`;
}

function audioSummary(event: AudioEvent): string {
  const rate = event.sample_rate >= 1000 ? `${+(event.sample_rate / 1000).toFixed(2)} kHz` : `${event.sample_rate} Hz`;
  const layout = event.channels === 2 ? "stereo" : "mono";
  return `${event.seconds.toFixed(2)} s · ${rate} · ${layout}`;
}

function Heatmap({ event }: { event: HeatmapEvent }) {
  const levels = useMemo(() => decodeBase64(event.data_base64), [event.data_base64]);
  const axis = (a: HeatmapEvent["x"], fallback: number) => ({
    label: a.label,
    unit: a.unit,
    min: a.min ?? 0,
    max: a.max ?? fallback,
    scale: a.scale,
    mel: a.mel,
    ticks: a.ticks,
  });
  return (
    <HeatmapCanvas
      levels={levels}
      rows={event.rows}
      cols={event.cols}
      origin={event.origin}
      colormap={isColormap(event.colormap) ? event.colormap : "magma"}
      x={axis(event.x, event.cols)}
      y={axis(event.y, event.rows)}
      value={
        event.value.min != null && event.value.max != null
          ? { label: event.value.label, min: event.value.min, max: event.value.max }
          : undefined
      }
      description={event.label ?? "A heatmap from your run"}
    />
  );
}

function Line({ event }: { event: LineEvent }) {
  return (
    <LineCanvas
      series={event.series}
      xlabel={event.xlabel}
      ylabel={event.ylabel}
      xScale={event.x_scale}
      yScale={event.y_scale}
      xlim={event.xlim}
      ylim={event.ylim}
      description={event.label ?? "A plot from your run"}
    />
  );
}
