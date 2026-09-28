import { useMemo, useState } from "react";

import { CanvasFigure, type Paint } from "./figure";
import type { DemoProps } from "./index";
import { Action, Actions, Choice, DemoFrame, Readout, Readouts, Slider } from "./kit";

/** Positions drawn one dot each; past this the figure draws taps as ticks. */
const DOT_LIMIT = 160;

/**
 * The dilations of a WaveNet stack, 1, 2, 4, ... 2^(M - 1), repeated for
 * each of `stacks` stacks, from the layer nearest the input up.
 */
function dilations(layers: number, stacks: number): number[] {
  const out: number[] = [];
  for (let b = 0; b < stacks; b += 1) for (let m = 0; m < layers; m += 1) out.push(2 ** m);
  return out;
}

/**
 * Which positions each level depends on, for the output at the last
 * position. Level 0 is the input and level `d.length` the output; a causal
 * layer with kernel K and dilation d reads positions p, p - d, ... p - (K - 1) d.
 */
function taps(kernel: number, dils: number[], window: number): Uint8Array[] {
  const levels = dils.length + 1;
  const out = Array.from({ length: levels }, () => new Uint8Array(window));
  out[levels - 1][window - 1] = 1;
  for (let level = levels - 1; level > 0; level -= 1) {
    const d = dils[level - 1];
    const above = out[level];
    const below = out[level - 1];
    for (let p = 0; p < window; p += 1) {
      if (!above[p]) continue;
      for (let k = 0; k < kernel; k += 1) if (p - k * d >= 0) below[p - k * d] = 1;
    }
  }
  return out;
}

const RATES = [
  { value: "16000", label: "16 kHz" },
  { value: "22050", label: "22.05 kHz" },
  { value: "24000", label: "24 kHz" },
] as const;

export default function WavenetReceptiveField({ caption }: DemoProps) {
  const [kernel, setKernel] = useState(2);
  const [layers, setLayers] = useState(4);
  const [stacks, setStacks] = useState(2);
  const [rate, setRate] = useState<(typeof RATES)[number]["value"]>("16000");

  const dils = useMemo(() => dilations(layers, stacks), [layers, stacks]);
  const sum = dils.reduce((a, b) => a + b, 0);
  const field = 1 + (kernel - 1) * sum;
  const plain = 1 + (kernel - 1) * dils.length;
  const margin = Math.max(3, Math.round(field * 0.08));
  const window = field + margin;
  const tapped = useMemo(() => taps(kernel, dils, window), [kernel, dils, window]);
  const ms = (field / Number(rate)) * 1000;

  const levels = dils.length + 1;
  const rowHeight = (width: number) => Math.max(7, Math.min(26, (width < 500 ? 240 : 300) / levels));
  const LEFT = 56;
  const TOP = 8;
  const BOTTOM = 34;

  function draw({ ctx, width, height, colour }: Paint) {
    const plotW = width - LEFT - 10;
    const rh = (height - TOP - BOTTOM) / levels;
    const x = (p: number) => LEFT + ((p + 0.5) / window) * plotW;
    // Level 0, the input, at the bottom.
    const y = (level: number) => TOP + (levels - 1 - level + 0.5) * rh;
    const demo = colour("--demo");
    const dots = window <= DOT_LIMIT;
    const r = Math.max(1.2, Math.min(3.2, (plotW / window) * 0.32, rh * 0.28));

    ctx.font = `10px "JetBrains Mono Variable", ui-monospace, monospace`;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const labelEvery = rh >= 10 ? 1 : Math.max(1, layers);
    for (let level = 0; level < levels; level += 1) {
      let text = "";
      if (level === 0) text = "input";
      else if (level === levels - 1 && rh >= 10) text = `d=${dils[level - 1]}`;
      else if ((level - 1) % labelEvery === 0) text = rh >= 10 ? `d=${dils[level - 1]}` : `stack ${Math.floor((level - 1) / layers) + 1}`;
      ctx.fillStyle = level === 0 ? colour("--ink") : colour("--ink-3");
      if (text) ctx.fillText(text, LEFT - 6, y(level));
    }

    // Stacks, as bands behind their layers.
    for (let b = 0; b < stacks; b += 1) {
      if (b % 2 === 1) {
        const topLevel = (b + 1) * layers;
        ctx.fillStyle = colour("--chrome");
        ctx.fillRect(LEFT, y(topLevel) - rh / 2, plotW, layers * rh);
      }
    }

    if (dots) {
      // Every position faintly, then the edges and positions the output depends on.
      ctx.fillStyle = colour("--rule-strong");
      for (let level = 0; level < levels; level += 1) {
        for (let p = 0; p < window; p += 1) {
          ctx.beginPath();
          ctx.arc(x(p), y(level), r * 0.7, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
      ctx.strokeStyle = demo;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = 1;
      for (let level = levels - 1; level > 0; level -= 1) {
        const d = dils[level - 1];
        for (let p = 0; p < window; p += 1) {
          if (!tapped[level][p]) continue;
          for (let k = 0; k < kernel; k += 1) {
            if (p - k * d < 0) continue;
            ctx.beginPath();
            ctx.moveTo(x(p), y(level));
            ctx.lineTo(x(p - k * d), y(level - 1));
            ctx.stroke();
          }
        }
      }
      ctx.globalAlpha = 1;
      ctx.fillStyle = demo;
      for (let level = 0; level < levels; level += 1) {
        for (let p = 0; p < window; p += 1) {
          if (!tapped[level][p]) continue;
          ctx.beginPath();
          ctx.arc(x(p), y(level), r, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    } else {
      // Too many positions for dots: one tick per position the output depends on.
      ctx.fillStyle = demo;
      const w = Math.max(1, plotW / window);
      for (let level = 0; level < levels; level += 1) {
        for (let p = 0; p < window; p += 1) {
          if (tapped[level][p]) ctx.fillRect(x(p) - w / 2, y(level) - rh * 0.3, w, rh * 0.6);
        }
      }
    }

    // The output sample.
    ctx.beginPath();
    ctx.arc(x(window - 1), y(levels - 1), r + 1.6, 0, 2 * Math.PI);
    ctx.strokeStyle = colour("--series-2");
    ctx.lineWidth = 2;
    ctx.stroke();

    // The receptive field, bracketed under the input row.
    const from = x(window - field) - (plotW / window) * 0.5;
    const to = x(window - 1) + (plotW / window) * 0.5;
    const by = y(0) + rh / 2 + 6;
    ctx.strokeStyle = colour("--series-2");
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(from, by - 4);
    ctx.lineTo(from, by);
    ctx.lineTo(to, by);
    ctx.lineTo(to, by - 4);
    ctx.stroke();
    ctx.fillStyle = colour("--ink-2");
    ctx.font = `11px "Inter Variable", ui-sans-serif, system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillText(`R = ${field.toLocaleString("en-US")} samples, ${ms.toFixed(1)} ms`, (from + to) / 2, by + 4);
  }

  return (
    <DemoFrame
      title="How far back a dilated stack can see"
      caption={caption}
      controls={
        <>
          <Slider label="Kernel size, K" value={kernel} min={2} max={4} onChange={setKernel} />
          <Slider
            label="Layers per stack, M"
            value={layers}
            min={1}
            max={10}
            format={(v) => `${v}, dilations 1 to ${2 ** (v - 1)}`}
            onChange={setLayers}
          />
          <Slider label="Stacks, B" value={stacks} min={1} max={4} onChange={setStacks} />
          <Choice label="Sample rate" value={rate} options={RATES} onChange={setRate} />
          <Actions>
            <Action
              tone="quiet"
              onClick={() => {
                setKernel(2);
                setLayers(4);
                setStacks(2);
              }}
            >
              Small stack
            </Action>
            <Action
              tone="quiet"
              onClick={() => {
                setKernel(2);
                setLayers(10);
                setStacks(3);
                setRate("24000");
              }}
            >
              WaveNet: 3 stacks of 1 to 512
            </Action>
          </Actions>
        </>
      }
    >
      <CanvasFigure
        height={(w) => Math.round(8 + 34 + levels * rowHeight(w))}
        deps={[tapped, kernel, dils, window, field, ms, layers, stacks]}
        draw={draw}
        description={`A dilated causal stack of ${dils.length} layers with a receptive field of ${field} samples`}
      />
      <p className="mt-2 text-[12px] leading-snug text-ink-3">
        The ringed sample at the top right is one output. The coloured positions below it are everything it depends on,
        layer by layer; nothing to its right, because every layer is causal.
      </p>
      <p className="mt-3 font-mono text-[12.5px] leading-relaxed text-ink-2">
        R = 1 + (K − 1) · Σd = 1 + {kernel - 1} · {sum.toLocaleString("en-US")} = {field.toLocaleString("en-US")}
      </p>
      <Readouts>
        <Readout label="Receptive field" value={field.toLocaleString("en-US")} hint="samples" tone="demo" />
        <Readout label="In time" value={`${ms < 10 ? ms.toFixed(2) : ms.toFixed(1)} ms`} hint={`at ${Number(rate) / 1000} kHz`} />
        <Readout label="Layers" value={dils.length} hint={`${stacks} × ${layers}`} />
        <Readout label="Without dilation" value={plain} hint="the same layers, d = 1" />
      </Readouts>
    </DemoFrame>
  );
}
