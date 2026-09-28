/**
 * Sound for the demos: play a buffer of samples, and record from the
 * microphone when the reader asks.
 *
 * Browsers start an `AudioContext` only after a click or a key press, so
 * nothing here runs on load: `play` and `record` are for event handlers.
 * The microphone is never opened except by `record`, which a reader starts
 * with a button, and the stream is closed as soon as the recording ends.
 */

import { useCallback, useEffect, useRef, useState } from "react";

let shared: AudioContext | null = null;

/** The page's one audio context, created on first use. */
export function audioContext(): AudioContext {
  if (!shared || shared.state === "closed") shared = new AudioContext();
  if (shared.state === "suspended") void shared.resume();
  return shared;
}

/** The sample rates every browser accepts for an `AudioBuffer`. */
export const MIN_BUFFER_RATE = 3000;
export const MAX_BUFFER_RATE = 384000;

/**
 * Plays `samples`, taken at `sampleRate`, and returns a function that stops
 * them. The browser resamples to the output rate with its own band-limited
 * interpolation, so a buffer at 8 kHz sounds like what those samples encode,
 * aliases included.
 */
export function playSamples(
  samples: Float32Array,
  sampleRate: number,
  options: { gain?: number; onEnded?: () => void } = {},
): () => void {
  const context = audioContext();
  const rate = Math.min(MAX_BUFFER_RATE, Math.max(MIN_BUFFER_RATE, Math.round(sampleRate)));
  const buffer = context.createBuffer(1, Math.max(1, samples.length), rate);
  buffer.copyToChannel(samples as Float32Array<ArrayBuffer>, 0);
  const source = context.createBufferSource();
  source.buffer = buffer;
  const gain = context.createGain();
  gain.gain.value = options.gain ?? 0.6;
  source.connect(gain).connect(context.destination);
  source.onended = () => options.onEnded?.();
  source.start();
  return () => {
    try {
      source.stop();
    } catch {
      // Already stopped.
    }
  };
}

/**
 * Plays one sound at a time. Starting a new one stops the last, and leaving
 * the page stops whatever is playing. `playing` names what is playing now.
 */
export function useAudioPlayer() {
  const stopRef = useRef<(() => void) | null>(null);
  const [playing, setPlaying] = useState<string | null>(null);

  const stop = useCallback(() => {
    stopRef.current?.();
    stopRef.current = null;
    setPlaying(null);
  }, []);

  const play = useCallback(
    (name: string, samples: Float32Array, sampleRate: number, gain?: number) => {
      stopRef.current?.();
      setPlaying(name);
      const stopThis = playSamples(samples, sampleRate, {
        gain,
        onEnded: () => {
          if (stopRef.current === stopThis) {
            stopRef.current = null;
            setPlaying(null);
          }
        },
      });
      stopRef.current = stopThis;
    },
    [],
  );

  useEffect(() => () => stopRef.current?.(), []);
  return { play, stop, playing };
}

export type MicrophoneState = "idle" | "asking" | "recording" | "unsupported" | "denied" | "failed";

/**
 * Records a few seconds from the microphone, resampled to `sampleRate`.
 *
 * `record` asks for permission the first time, so call it only from a click.
 * It resolves to the samples, or to null when the reader said no, the browser
 * has no microphone, or the page is not served over HTTPS; `state` says
 * which, so the demo can explain and carry on with its built-in signal.
 */
export function useMicrophone() {
  const [state, setState] = useState<MicrophoneState>("idle");
  const stream = useRef<MediaStream | null>(null);

  useEffect(() => () => stream.current?.getTracks().forEach((track) => track.stop()), []);

  const record = useCallback(async (seconds: number, sampleRate: number): Promise<Float32Array | null> => {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setState("unsupported");
      return null;
    }
    setState("asking");
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
    } catch (error) {
      setState(error instanceof DOMException && error.name === "NotAllowedError" ? "denied" : "unsupported");
      return null;
    }

    try {
      setState("recording");
      const recorder = new MediaRecorder(stream.current);
      const chunks: Blob[] = [];
      recorder.ondataavailable = (event) => chunks.push(event.data);
      const stopped = new Promise<void>((resolve) => {
        recorder.onstop = () => resolve();
      });
      recorder.start();
      await new Promise((resolve) => window.setTimeout(resolve, seconds * 1000));
      recorder.stop();
      await stopped;
      stream.current.getTracks().forEach((track) => track.stop());
      stream.current = null;

      const encoded = await new Blob(chunks, { type: recorder.mimeType }).arrayBuffer();
      const decoded = await audioContext().decodeAudioData(encoded);
      // An offline context resamples to the rate the demo works at.
      const length = Math.ceil(decoded.duration * sampleRate);
      const offline = new OfflineAudioContext(1, Math.max(1, length), sampleRate);
      const source = offline.createBufferSource();
      source.buffer = decoded;
      source.connect(offline.destination);
      source.start();
      const rendered = await offline.startRendering();
      setState("idle");
      return rendered.getChannelData(0).slice(0, Math.round(seconds * sampleRate));
    } catch {
      stream.current?.getTracks().forEach((track) => track.stop());
      stream.current = null;
      setState("failed");
      return null;
    }
  }, []);

  return { state, record };
}
