import { useState, type FormEvent } from "react";

import { api, ApiError } from "../lib/api";
import Mark from "./Mark";

export default function Login({ onSignedIn }: { onSignedIn: (name: string) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const result = await api.login(username.trim(), password);
      onSignedIn(result.username);
    } catch (exc) {
      setError(
        exc instanceof ApiError
          ? exc.message
          : "Could not reach the server. Try again.",
      );
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-full items-center justify-center bg-canvas px-5 py-12">
      <div className="w-full max-w-[22rem]">
        <Mark size={32} />
        <h1 className="mt-5 font-serif text-[2rem] font-semibold leading-tight tracking-[-0.015em] text-ink">
          Speech and audio
        </h1>
        <p className="mt-2 font-serif text-[1.05rem] leading-relaxed text-ink-3">
          Lessons you can hear, from a sampled waveform to models that
          listen and speak, and every one ends with code you write yourself.
        </p>

        <form onSubmit={submit} className="mt-8 space-y-4">
          <Field
            label="Username"
            value={username}
            onChange={setUsername}
            autoComplete="username"
            autoFocus
          />
          <Field
            label="Password"
            value={password}
            onChange={setPassword}
            type="password"
            autoComplete="current-password"
          />

          {error && (
            <p
              role="alert"
              className="rounded-md bg-[color-mix(in_oklab,var(--bad)_8%,transparent)] px-3 py-2 text-[13.5px] text-bad"
            >
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={busy || !username || !password}
            className="btn btn-primary min-h-10! w-full text-[14px]!"
          >
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </div>
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
  type = "text",
  autoComplete,
  autoFocus,
}: {
  label: string;
  value: string;
  onChange: (next: string) => void;
  type?: string;
  autoComplete?: string;
  autoFocus?: boolean;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-[13px] font-medium text-ink-2">{label}</span>
      <input
        type={type}
        value={value}
        autoComplete={autoComplete}
        autoFocus={autoFocus}
        onChange={(event) => onChange(event.target.value)}
        className="field"
      />
    </label>
  );
}
