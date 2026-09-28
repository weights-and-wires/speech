/** The typed client for everything under /api. */

export type LessonMeta = {
  slug: string;
  title: string;
  part: string;
  summary: string;
  minutes: number | null;
  objectives: string[];
  prereqs: string[];
  exercise: string | null;
  gpu: boolean;
};

export type ExerciseDetail = {
  id: string;
  title: string;
  brief: string;
  needs_gpu: boolean;
  timeout: number;
  packages: string[];
  hints: string[];
  starter: string;
};

export type Lesson = LessonMeta & {
  body: string;
  prev: string | null;
  next: string | null;
  index: number;
  count: number;
  exercise_detail?: ExerciseDetail;
  draft?: string | null;
};

export type Part = { part: string; lessons: LessonMeta[] };

/** The whole course as one PDF, built at deploy time. */
export type BookFile = {
  title: string;
  pages: number;
  bytes: number;
  built: string;
  commit: string;
  url: string;
};

export type ProgressMap = Record<
  string,
  { status: "started" | "solved"; updated_at: number }
>;

/**
 * One axis of a figure from `speechlab`. `min` and `max` are the edges of the
 * data, `scale` says how rows are spaced (a mel axis is even in mels, not Hz),
 * and `ticks` names each row or column, such as an attention map's tokens.
 */
export type FigureAxis = {
  label: string | null;
  unit: string | null;
  min: number | null;
  max: number | null;
  scale?: "linear" | "mel";
  mel?: "htk" | "slaney";
  ticks?: string[] | null;
};

/** `speechlab.play`: a 16-bit PCM WAV file, base64-encoded. */
export type AudioEvent = {
  type: "audio";
  label: string | null;
  sample_rate: number;
  channels: number;
  seconds: number;
  original_seconds?: number;
  truncated?: boolean;
  note?: string | null;
  wav_base64: string;
  /** The check that played it, in check mode. */
  check?: number;
};

/**
 * `speechlab.show_spectrogram` and `show_image`: `rows * cols` bytes, row by
 * row, each a level from 0 (`value.min`) to 255 (`value.max`). With
 * `origin: "lower"`, row 0 is drawn at the bottom.
 */
export type HeatmapEvent = {
  type: "figure";
  kind: "heatmap";
  label: string | null;
  shape: [number, number];
  rows: number;
  cols: number;
  origin: "lower" | "upper";
  colormap: string;
  data_base64: string;
  x: FigureAxis;
  y: FigureAxis;
  value: { label: string | null; min: number | null; max: number | null };
  note?: string | null;
  check?: number;
};

/** `speechlab.show_plot`: up to six series; a null breaks the line. */
export type LineEvent = {
  type: "figure";
  kind: "line";
  label: string | null;
  xlabel: string | null;
  ylabel: string | null;
  x_scale: "linear" | "log";
  y_scale: "linear" | "log";
  xlim: [number | null, number | null] | null;
  ylim: [number | null, number | null] | null;
  series: { label: string | null; x: (number | null)[]; y: (number | null)[]; points: number }[];
  note?: string | null;
  check?: number;
};

export type MediaEvent = AudioEvent | HeatmapEvent | LineEvent;

export type RunEvent =
  | { type: "begin"; backend: string; mode: string }
  | MediaEvent
  | { type: "status"; text: string }
  | { type: "plan"; total: number }
  | { type: "stdout"; text: string }
  | { type: "stderr"; text: string }
  | {
      type: "check";
      index: number;
      description: string;
      status: "passed" | "failed";
      seconds: number;
      message?: string;
      output?: string;
      timings?: Record<string, number>;
    }
  | { type: "error"; message: string; detail?: string | null }
  | {
      type: "done";
      ok: boolean;
      passed?: number;
      total?: number;
      seconds?: number;
      timeout?: boolean;
    };

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    credentials: "same-origin",
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // A non-JSON error body still gives us the status text.
    }
    throw new ApiError(detail, response.status);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  session: () =>
    request<{ signed_in: boolean; username: string | null }>("/api/session"),

  login: (username: string, password: string) =>
    request<{ signed_in: boolean; username: string }>("/api/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),

  logout: () => request<{ signed_in: boolean }>("/api/logout", { method: "POST" }),

  lessons: () =>
    request<{ parts: Part[]; progress: ProgressMap }>("/api/lessons"),

  lesson: (slug: string) => request<Lesson>(`/api/lessons/${slug}`),

  solution: (exerciseId: string) =>
    request<{ id: string; solution: string }>(
      `/api/exercises/${exerciseId}/solution`,
    ),

  saveDraft: (exerciseId: string, code: string) =>
    request<{ saved: boolean }>(`/api/drafts/${exerciseId}`, {
      method: "PUT",
      body: JSON.stringify({ code }),
    }),

  resetDraft: (exerciseId: string) =>
    request<{ saved: boolean }>(`/api/drafts/${exerciseId}`, {
      method: "DELETE",
    }),

  setProgress: (slug: string, status: "started" | "solved") =>
    request<{ slug: string; status: string }>(`/api/progress/${slug}`, {
      method: "PUT",
      body: JSON.stringify({ status }),
    }),

  me: () =>
    request<{
      username: string;
      stats: { solved: number; runs: number; compute_seconds: number };
      runs: {
        exercise_id: string;
        mode: string;
        backend: string;
        ok: number;
        passed: number | null;
        total: number | null;
        seconds: number | null;
        started_at: number;
      }[];
      gpu: boolean;
    }>("/api/me"),

  compute: () =>
    request<{
      local: { workers: number; memory_mb: number; timeout_ceiling: number };
      runpod: Record<string, unknown>;
    }>("/api/compute"),

  book: () => request<{ book: BookFile | null }>("/api/book"),

  health: () =>
    request<{
      ok: boolean;
      commit: string;
      lessons: number;
      exercises: number;
      gpu: boolean;
      book_pages: number;
    }>("/api/health"),
};

/**
 * Start a run and yield its events as they arrive.
 *
 * The response is an SSE stream that stays open for the length of the run, so
 * this reads the body incrementally rather than awaiting `response.json()`.
 * Aborting the signal closes the connection, and the server kills the process.
 */
export async function* runStream(
  body: {
    exercise_id: string;
    mode: "run" | "check";
    code: string;
    prefer_gpu?: boolean;
  },
  signal: AbortSignal,
): AsyncGenerator<RunEvent> {
  const response = await fetch("/api/run", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok || !response.body) {
    let detail = response.statusText;
    try {
      detail = String((await response.json())?.detail ?? detail);
    } catch {
      // Keep the status text.
    }
    throw new ApiError(detail, response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE separates events with a blank line.
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        try {
          yield JSON.parse(line.slice(6)) as RunEvent;
        } catch {
          // A half-written frame cannot happen here, but a malformed one
          // should not take the whole run down.
        }
      }
      boundary = buffer.indexOf("\n\n");
    }
  }
}
