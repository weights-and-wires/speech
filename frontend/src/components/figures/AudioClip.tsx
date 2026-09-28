import { useEffect, useMemo } from "react";

import { decodeBase64 } from "../../lib/plotting";

/**
 * A WAV clip from `speechlab.play`, as the browser's own audio player.
 *
 * The clip becomes a blob URL for as long as the player is on the page, and
 * is released when it goes, so a reader pressing Run fifty times does not
 * hold fifty clips in memory.
 */
export default function AudioClip({ wavBase64, label }: { wavBase64: string; label: string }) {
  const url = useMemo(() => {
    const bytes = decodeBase64(wavBase64);
    return URL.createObjectURL(new Blob([bytes.buffer as ArrayBuffer], { type: "audio/wav" }));
  }, [wavBase64]);

  useEffect(() => () => URL.revokeObjectURL(url), [url]);

  return (
    <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1">
      <audio controls preload="metadata" src={url} aria-label={label} className="h-9 w-full min-w-0 max-w-[480px] flex-1" />
      <a href={url} download={`${label.replace(/[^\w.-]+/g, "-").slice(0, 40) || "clip"}.wav`} className="text-[12px] text-ink-3 underline-offset-2 hover:text-ink hover:underline">
        Download WAV
      </a>
    </div>
  );
}
