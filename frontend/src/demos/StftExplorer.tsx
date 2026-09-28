import { useMemo, useState } from "react";

import { fadeEdges, linearChirp, stft, toDb, type WindowName } from "./dsp";
import type { DemoProps } from "./index";
import {
  Action,
  Actions,
  Choice,
  DemoFrame,
  Plot,
  Readout,
  Readouts,
  Spectrogram,
  Waveform,
  useAudioPlayer,
  useMicrophone,
} from "./kit";

const RATE = 16000;
const SECONDS = 1;
const FLOOR_DB = -80;
/** The two steady tones, 80 Hz apart: a long window separates them, a short one cannot. */
const TONES = [2000, 2080];
/** The two clicks, 20 ms apart: a short window separates them, a long one cannot. */
const CLICKS = [0.5, 0.52];
/** The band the tone zoom shows. */
const ZOOM_HZ: [number, number] = [1700, 2400];

const LENGTHS = [64, 128, 256, 512, 1024, 2048] as const;
type Length = (typeof LENGTHS)[number];
type HopShare = "8" | "4" | "2" | "1";

/**
 * The test signal: a chirp from 300 Hz to 6 kHz, two steady tones close in
 * frequency, and two clicks close in time. Each part tests one side of the
 * trade-off, so no single window length shows all of them sharply.
 */
function testSignal(): Float32Array {
  const signal = linearChirp(300, 6000, RATE, SECONDS, 0.25);
  for (let i = 0; i < signal.length; i += 1) {
    for (const hz of TONES) signal[i] += 0.12 * Math.sin((2 * Math.PI * hz * i) / RATE);
  }
  for (const at of CLICKS) {
    const start = Math.round(at * RATE);
    // A click four samples long, shaped so it is broadband but not harsh.
    [0.5, 1, 1, 0.5].forEach((g, k) => {
      signal[start + k] += 0.8 * g * (k % 2 ? -1 : 1);
    });
  }
  // Faded in and out, so the ends of the clip do not draw two more clicks.
  return fadeEdges(signal, RATE, 0.03);
}

const TEST_SIGNAL = testSignal();

export default function StftExplorer({ caption }: DemoProps) {
  const [length, setLength] = useState<Length>(512);
  const [hopShare, setHopShare] = useState<HopShare>("4");
  const [windowName, setWindowName] = useState<WindowName>("hann");
  const [zoom, setZoom] = useState<"all" | "tones">("all");
  const [recorded, setRecorded] = useState<Float32Array | null>(null);
  const { play, stop, playing } = useAudioPlayer();
  const microphone = useMicrophone();

  const signal = recorded ?? TEST_SIGNAL;
  const hop = Math.max(1, length / Number(hopShare));

  const spec = useMemo(() => {
    const magnitude = stft(signal, { nFft: length, hop, window: windowName });
    return { ...magnitude, data: toDb(magnitude.data, { floorDb: FLOOR_DB }) };
  }, [signal, length, hop, windowName]);

  const binHz = RATE / length;
  const rows: [number, number] =
    zoom === "tones"
      ? [Math.floor(ZOOM_HZ[0] / binHz), Math.min(spec.bins, Math.ceil(ZOOM_HZ[1] / binHz) + 1)]
      : [0, spec.bins];

  async function listenToMicrophone() {
    stop();
    const samples = await microphone.record(SECONDS * 2, RATE);
    if (samples) setRecorded(samples);
  }

  const micNote: Record<string, string> = {
    asking: "Waiting for permission to use the microphone.",
    recording: "Recording two seconds. Speak, whistle, or clap.",
    denied: "The browser was not allowed to use the microphone, so the demo keeps its test signal.",
    unsupported: "This browser has no microphone to offer here, so the demo keeps its test signal.",
    failed: "The recording could not be read, so the demo keeps its test signal.",
  };

  return (
    <DemoFrame
      title="Trading time for frequency in a spectrogram"
      caption={caption}
      controls={
        <>
          <Choice
            label="Window length"
            value={String(length)}
            options={LENGTHS.map((n) => ({ value: String(n), label: String(n) }))}
            onChange={(value) => setLength(Number(value) as Length)}
          />
          <Choice
            label="Hop"
            value={hopShare}
            options={[
              { value: "8", label: "N/8" },
              { value: "4", label: "N/4" },
              { value: "2", label: "N/2" },
              { value: "1", label: "N" },
            ]}
            onChange={setHopShare}
          />
          <Choice
            label="Window"
            value={windowName}
            options={[
              { value: "rectangular", label: "Rectangular" },
              { value: "hann", label: "Hann" },
              { value: "hamming", label: "Hamming" },
            ]}
            onChange={setWindowName}
          />
          <Choice
            label="Show"
            value={zoom}
            options={[
              { value: "all", label: "0 to 8 kHz" },
              { value: "tones", label: "The two tones" },
            ]}
            onChange={setZoom}
          />
          <Actions>
            <Action onClick={playing ? stop : () => play("signal", signal, RATE, 0.5)}>
              {playing ? "Stop" : "Listen"}
            </Action>
            {recorded ? (
              <Action tone="quiet" onClick={() => setRecorded(null)}>
                Back to the test signal
              </Action>
            ) : (
              <Action tone="quiet" onClick={listenToMicrophone}>
                Record from the microphone
              </Action>
            )}
          </Actions>
          {microphone.state !== "idle" && (
            <p className="text-[12.5px] leading-snug text-ink-3 sm:col-span-2" role="status">
              {micNote[microphone.state]}
            </p>
          )}
        </>
      }
    >
      <Plot
        height={90}
        xDomain={[0, signal.length / RATE]}
        yDomain={[-1, 1]}
        yTicks={2}
        xTicks={4}
        tickFormat={(value) => value.toFixed(value === 0 ? 0 : 2)}
      >
        {() => <Waveform samples={signal} sampleRate={RATE} />}
      </Plot>
      <div className="mt-2">
        <Spectrogram
          spec={spec}
          sampleRate={RATE}
          floorDb={FLOOR_DB}
          rows={rows}
          description={`A spectrogram with a ${length}-sample ${windowName} window`}
        />
      </div>
      <Readouts>
        <Readout label="Window" value={`${((length / RATE) * 1000).toFixed(1)} ms`} hint={`${length} samples`} />
        <Readout label="Bin spacing" value={`${binHz.toFixed(1)} Hz`} hint="fs / N" />
        <Readout label="Hop" value={`${((hop / RATE) * 1000).toFixed(1)} ms`} hint={`${hop} samples`} />
        <Readout label="Frames" value={spec.frames} hint={`${spec.bins} bins each`} />
      </Readouts>
      <p className="mt-3 text-[13px] leading-snug text-ink-3">
        {recorded
          ? "This is your recording, at 16 kHz."
          : `The test signal holds a chirp, two tones ${TONES[1] - TONES[0]} Hz apart near 2 kHz, and two clicks ${Math.round(
              (CLICKS[1] - CLICKS[0]) * 1000,
            )} ms apart at half a second.`}
      </p>
    </DemoFrame>
  );
}
