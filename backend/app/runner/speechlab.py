"""Audio and figure output for a run: play a clip, show a spectrogram or a plot.

A starter file's `__main__` block, or a check, imports these helpers:

    from speechlab import play, show_spectrogram, show_plot, show_image

Each call sends one event through the harness's JSON-lines channel, beside the
`stdout` and `check` events and never through `sys.stdout`, so a reader's own
prints cannot be mistaken for one. The console on the page renders each event
inline, in the order the calls happened: an audio player for `play`, and a
canvas for the figures.

Every payload is bounded, so one call cannot flood the page. A clip is cut at
30 seconds or about 2 MB of WAV, a heatmap is pooled down to at most 256 rows
by 512 columns and sent as 8-bit levels, and a line plot keeps at most 2000
points per series. A run gets at most `MAX_EVENTS` figures and clips, and at
most `MAX_RUN_BYTES` of them in total. The event keeps the original `shape`
or point count, so the page can say how much was pooled; when a clip is cut
or clipped, the event carries a note that says so, and the reader sees it.

The helpers also find the files the GPU worker bakes in at build time, through
`SPEECH_ASSETS`: see `asset_path` and `has_asset`.

The harness connects this module to its channel with `connect`. Outside the
harness, for example when a reader runs a file with plain `python`, each call
prints a one-line description to stderr instead.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
import wave
from pathlib import Path
from typing import Any, Callable, Sequence

__all__ = [
    "play",
    "show_spectrogram",
    "show_image",
    "show_plot",
    "show_waveform",
    "asset_path",
    "has_asset",
    "assets_root",
    "read_audio",
    "MAX_AUDIO_SECONDS",
    "MAX_AUDIO_BYTES",
    "MAX_HEATMAP_ROWS",
    "MAX_HEATMAP_COLS",
    "MAX_SERIES",
    "MAX_POINTS",
    "MAX_EVENTS",
    "MAX_RUN_BYTES",
]

# Limits ----------------------------------------------------------------

MAX_AUDIO_SECONDS = 30.0
#: The largest WAV file one `play` call sends, header included.
MAX_AUDIO_BYTES = 2_000_000
MAX_AUDIO_CHANNELS = 2
MAX_HEATMAP_ROWS = 256
MAX_HEATMAP_COLS = 512
MAX_SERIES = 6
MAX_POINTS = 2000
#: Tick labels on a `show_image` axis, such as the tokens of an attention map.
MAX_TICK_LABELS = 64
#: Figures and clips in one run.
MAX_EVENTS = 32
#: Encoded bytes of figures and clips in one run. The backend enforces a
#: slightly larger ceiling of its own, so this is the limit a reader meets.
MAX_RUN_BYTES = 8_000_000
MAX_LABEL = 200

COLORMAPS = ("magma", "viridis", "inferno", "gray")

# The channel -----------------------------------------------------------

_EMIT: Callable[[dict[str, Any]], None] | None = None
#: Merged into every event. The harness sets `check` while a check runs, so
#: the page can say which check drew a figure.
_CONTEXT: dict[str, Any] = {}
_SENT = {"events": 0, "bytes": 0, "refused": 0}


def connect(emit: Callable[[dict[str, Any]], None]) -> None:
    """Route events to `emit`. The harness calls this once, at start-up."""
    global _EMIT
    _EMIT = emit
    reset_budget()


def reset_budget() -> None:
    _SENT.update(events=0, bytes=0, refused=0)


def _send(event: dict[str, Any]) -> bool:
    """Send one media event, or a note when the run's budget is spent."""
    event = {**event, **_CONTEXT}
    size = len(json.dumps(event, default=str))
    what = event.get("kind") or event["type"]
    if _SENT["events"] >= MAX_EVENTS or _SENT["bytes"] + size > MAX_RUN_BYTES:
        _SENT["refused"] += 1
        # One note is enough; a loop that calls play() a hundred times should
        # not bury the output in a hundred copies of it.
        if _SENT["refused"] == 1:
            _note(
                f"speechlab: this run has sent its limit of figures and clips "
                f"({MAX_EVENTS} items or {MAX_RUN_BYTES / 1e6:.0f} MB), so the "
                f"{what} labelled {event.get('label')!r} and anything after it "
                "is not shown."
            )
        return False
    _SENT["events"] += 1
    _SENT["bytes"] += size
    if _EMIT is None:
        label = f" {event['label']!r}" if event.get("label") else ""
        sys.stderr.write(f"[speechlab] {what}{label} ({size / 1e3:.0f} kB) is shown only in the course runner\n")
        return True
    _EMIT(event)
    return True


def _note(text: str) -> None:
    if _EMIT is None:
        sys.stderr.write(text + "\n")
    else:
        _EMIT({"type": "stderr", "text": text, **_CONTEXT})


# Helpers ---------------------------------------------------------------


def _np():
    import numpy as np

    return np


def _array(value: Any, name: str):
    """A numpy array from an array-like, a list, or a torch tensor on any device."""
    np = _np()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        if hasattr(value, "float") and str(getattr(value, "dtype", "")) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    try:
        return np.asarray(value)
    except Exception as exc:  # noqa: BLE001 - the message is what matters
        raise TypeError(f"{name} must be array-like, got {type(value).__name__}: {exc}") from None


def _label(label: Any) -> str | None:
    if label is None:
        return None
    text = str(label)
    return text if len(text) <= MAX_LABEL else text[: MAX_LABEL - 1] + "…"


def _num(value: float) -> float | None:
    """A float with six significant digits, or None when it is not finite."""
    if not math.isfinite(value):
        return None
    return float(f"{value:.6g}")


def _edges(n: int, target: int):
    np = _np()
    return np.linspace(0, n, target + 1).round().astype(int)


def _pool(matrix, rows: int, cols: int, how: str):
    """Pool a 2-D array down to at most `rows` by `cols` cells.

    `max` keeps thin features, such as a click or a harmonic, that averaging
    would wash out; `mean` suits a map whose cells are weights.
    """
    np = _np()
    out = matrix
    for axis, limit in ((0, rows), (1, cols)):
        n = out.shape[axis]
        if n <= limit:
            continue
        starts = _edges(n, limit)[:-1]
        if how == "max":
            out = np.maximum.reduceat(out, starts, axis=axis)
        else:
            sums = np.add.reduceat(out, starts, axis=axis)
            counts = np.diff(np.append(starts, n))
            shape = [1, 1]
            shape[axis] = counts.size
            out = sums / counts.reshape(shape)
    return out


def _quantize(values, lo: float | None, hi: float | None):
    np = _np()
    finite = np.isfinite(values)
    if not finite.any():
        values = np.zeros_like(values)
        finite = np.ones_like(values, dtype=bool)
    low = float(values[finite].min()) if lo is None else float(lo)
    high = float(values[finite].max()) if hi is None else float(hi)
    if not high > low:
        high = low + 1.0
    filled = np.where(finite, values, low)
    levels = np.clip((filled - low) / (high - low), 0.0, 1.0)
    data = np.round(levels * 255).astype(np.uint8)
    return base64.b64encode(np.ascontiguousarray(data).tobytes()).decode("ascii"), low, high


# Audio -----------------------------------------------------------------


def play(samples: Any, sample_rate: int, label: str | None = None) -> None:
    """Send a clip to the page, where it appears as an audio player.

    `samples` is a 1-D array, or a 2-D array shaped (channels, samples) with one
    or two channels. Floats are taken to lie in [-1, 1] and are clipped to it;
    int16 is sent as it is. The clip is encoded as 16-bit PCM WAV, and cut at
    30 seconds or about 2 MB, whichever comes first, with a note saying so.
    """
    np = _np()
    rate = int(sample_rate)
    if not 1 <= rate <= 384_000:
        raise ValueError(f"sample_rate must be between 1 and 384000 Hz, got {sample_rate!r}")
    audio = _array(samples, "samples")
    if audio.ndim == 1:
        audio = audio[None, :]
    elif audio.ndim != 2:
        raise ValueError(f"samples must be 1-D or (channels, samples), got shape {audio.shape}")
    channels, frames = audio.shape
    if channels > MAX_AUDIO_CHANNELS:
        raise ValueError(
            f"samples has shape {audio.shape}, which reads as {channels} channels. "
            "Pass (channels, samples) with one or two channels; for a (samples, "
            "channels) array, pass samples.T."
        )
    if frames == 0:
        raise ValueError("samples is empty, so there is nothing to play")
    if np.iscomplexobj(audio) or audio.dtype == bool:
        raise TypeError(f"samples must be real numbers, got dtype {audio.dtype}")

    notes: list[str] = []
    if audio.dtype == np.int16:
        pcm = audio
    elif audio.dtype == np.uint8:
        pcm = ((audio.astype(np.int16) - 128) * 256).astype(np.int16)
    elif np.issubdtype(audio.dtype, np.integer):
        raise TypeError(
            f"integer samples must be int16, got {audio.dtype}. Scale them to "
            "floats in [-1, 1] first."
        )
    else:
        floats = audio.astype(np.float64)
        bad = ~np.isfinite(floats)
        if bad.any():
            notes.append(f"{int(bad.sum())} samples were NaN or infinite and were set to 0.")
            floats = np.where(bad, 0.0, floats)
        clipped = int((np.abs(floats) > 1.0).sum())
        if clipped:
            notes.append(
                f"{clipped} samples were outside [-1, 1] and were clipped; the peak "
                f"was {float(np.abs(floats).max()):.3g}."
            )
        pcm = np.round(np.clip(floats, -1.0, 1.0) * 32767).astype(np.int16)

    limit = min(int(MAX_AUDIO_SECONDS * rate), (MAX_AUDIO_BYTES - 44) // (2 * channels))
    truncated = frames > limit
    if truncated:
        notes.append(
            f"The clip is {frames / rate:.1f} s long, so only the first "
            f"{limit / rate:.1f} s is sent."
        )
        pcm = pcm[:, :limit]

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(np.ascontiguousarray(pcm.T).astype("<i2").tobytes())

    kept = pcm.shape[1]
    _send(
        {
            "type": "audio",
            "label": _label(label),
            "sample_rate": rate,
            "channels": channels,
            "seconds": kept / rate,
            "original_seconds": frames / rate,
            "truncated": truncated,
            "note": " ".join(notes) or None,
            "wav_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
    )


# Heatmaps --------------------------------------------------------------


def _axis(label: str | None, unit: str | None, low: float, high: float, **extra: Any) -> dict:
    return {"label": label, "unit": unit, "min": _num(low), "max": _num(high), **extra}


def show_spectrogram(
    matrix: Any,
    sample_rate: float | None = None,
    hop_length: int | None = None,
    label: str | None = None,
    db: bool = True,
    y_axis: str = "linear",
    *,
    fmin: float = 0.0,
    fmax: float | None = None,
    htk: bool = False,
    power: bool = False,
    top_db: float | None = 80.0,
    colormap: str = "magma",
    value_label: str | None = None,
) -> None:
    """Send a spectrogram to the page, drawn as a heatmap with its axes.

    `matrix` is shaped (frequency bins, frames), the way `numpy.fft.rfft` along
    each frame and a transpose produce it, with row 0 the lowest frequency. It
    may be complex, in which case its magnitude is shown.

    With `db=True` the values are converted to decibels, `20 log10` of a
    magnitude or `10 log10` of a power when `power=True`, and everything more
    than `top_db` below the loudest cell is shown as the floor. With `db=False`
    the values are shown as they are, which suits a matrix already in dB.

    `y_axis` says what the rows are: `"linear"` for FFT bins from `fmin` to
    `fmax`, which default to 0 and `sample_rate / 2`, `"mel"` for mel bands from `fmin` to `fmax` (Slaney's
    formula, or HTK's with `htk=True`), or `"bins"` for plain row numbers. The
    time axis is in seconds when `sample_rate` and `hop_length` are both given,
    and in frames otherwise.
    """
    np = _np()
    values = _array(matrix, "matrix")
    if values.ndim != 2:
        raise ValueError(f"matrix must be 2-D (frequency bins, frames), got shape {values.shape}")
    if 0 in values.shape:
        raise ValueError(f"matrix is empty, shape {values.shape}")
    if y_axis not in ("linear", "mel", "bins"):
        raise ValueError(f"y_axis must be 'linear', 'mel', or 'bins', got {y_axis!r}")
    if colormap not in COLORMAPS:
        raise ValueError(f"colormap must be one of {', '.join(COLORMAPS)}, got {colormap!r}")
    if np.iscomplexobj(values):
        values = np.abs(values)
    values = values.astype(np.float64)
    bins, frames = values.shape

    notes: list[str] = []
    floor = None
    if db:
        if (values < 0).any():
            notes.append("Negative values were treated as zero before taking decibels.")
        scale = 10.0 if power else 20.0
        values = scale * np.log10(np.maximum(values, 1e-10))
        if top_db is not None:
            peak = float(np.nanmax(np.where(np.isfinite(values), values, -np.inf)))
            floor = peak - float(top_db)
            values = np.maximum(values, floor)

    pooled = _pool(values, MAX_HEATMAP_ROWS, MAX_HEATMAP_COLS, "max")
    data, low, high = _quantize(pooled, floor, None)

    if sample_rate and hop_length:
        x = _axis("Time", "s", 0.0, frames * hop_length / float(sample_rate))
    else:
        x = _axis("Frame", None, 0.0, float(frames))

    if y_axis == "linear" and sample_rate:
        # A matrix cut to its low bins passes the top of what it kept as fmax.
        top = fmax if fmax is not None else float(sample_rate) / 2
        y = _axis("Frequency", "Hz", float(fmin), float(top), scale="linear")
    elif y_axis == "mel":
        top = fmax if fmax is not None else (float(sample_rate) / 2 if sample_rate else None)
        if top is None:
            raise ValueError("a mel axis needs fmax, or a sample_rate to take fmax from")
        y = _axis("Frequency", "Hz", float(fmin), float(top), scale="mel", mel="htk" if htk else "slaney")
    else:
        y = _axis("Bin", None, 0.0, float(bins), scale="linear")

    _send(
        {
            "type": "figure",
            "kind": "heatmap",
            "label": _label(label),
            "shape": [bins, frames],
            "rows": int(pooled.shape[0]),
            "cols": int(pooled.shape[1]),
            "origin": "lower",
            "colormap": colormap,
            "data_base64": data,
            "x": x,
            "y": y,
            "value": {"label": value_label or ("dB" if db else None), "min": _num(low), "max": _num(high)},
            "note": " ".join(notes) or None,
        }
    )


def show_image(
    matrix: Any,
    label: str | None = None,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    x_labels: Sequence[str] | None = None,
    y_labels: Sequence[str] | None = None,
    colormap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    origin: str = "upper",
    value_label: str | None = None,
) -> None:
    """Send a generic heatmap, such as an attention or alignment map.

    Row 0 is drawn at the top unless `origin="lower"`. `x_labels` and
    `y_labels` name the columns and rows, the tokens or frames of an attention
    map, and are kept when there are at most 64 of them and the matrix needs no
    pooling. The matrix is averaged down to at most 256 by 512 cells.
    """
    np = _np()
    values = _array(matrix, "matrix")
    if values.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {values.shape}")
    if 0 in values.shape:
        raise ValueError(f"matrix is empty, shape {values.shape}")
    if origin not in ("upper", "lower"):
        raise ValueError(f"origin must be 'upper' or 'lower', got {origin!r}")
    if colormap not in COLORMAPS:
        raise ValueError(f"colormap must be one of {', '.join(COLORMAPS)}, got {colormap!r}")
    if np.iscomplexobj(values):
        values = np.abs(values)
    values = values.astype(np.float64)
    rows, cols = values.shape

    notes: list[str] = []
    pooled = _pool(values, MAX_HEATMAP_ROWS, MAX_HEATMAP_COLS, "mean")
    data, low, high = _quantize(pooled, vmin, vmax)

    def ticks(names: Sequence[str] | None, n: int, pooled_n: int) -> list[str] | None:
        if names is None:
            return None
        names = [str(name)[:24] for name in names]
        if len(names) != n:
            raise ValueError(f"got {len(names)} tick labels for an axis of length {n}")
        if n > MAX_TICK_LABELS or pooled_n != n:
            notes.append(f"The {n} tick labels on one axis were dropped, because there are too many to read.")
            return None
        return names

    _send(
        {
            "type": "figure",
            "kind": "heatmap",
            "label": _label(label),
            "shape": [rows, cols],
            "rows": int(pooled.shape[0]),
            "cols": int(pooled.shape[1]),
            "origin": origin,
            "colormap": colormap,
            "data_base64": data,
            "x": _axis(xlabel, None, 0.0, float(cols), ticks=ticks(x_labels, cols, pooled.shape[1])),
            "y": _axis(ylabel, None, 0.0, float(rows), ticks=ticks(y_labels, rows, pooled.shape[0]), scale="linear"),
            "value": {"label": value_label, "min": _num(low), "max": _num(high)},
            "note": " ".join(notes) or None,
        }
    )


# Line plots ------------------------------------------------------------


def _series(item: Any, index: int):
    """Split one positional argument into (x, y) arrays."""
    np = _np()
    if isinstance(item, tuple) and len(item) == 2:
        x, y = _array(item[0], "x"), _array(item[1], "y")
        if x.ndim == 1 and y.ndim == 1:
            if x.shape != y.shape:
                raise ValueError(f"series {index}: x has {x.size} points and y has {y.size}")
            return x.astype(np.float64), y.astype(np.float64)
    y = _array(item, "y")
    if y.ndim != 1:
        raise ValueError(
            f"series {index} has shape {y.shape}. Pass each series as a 1-D array "
            "of y values, or as an (x, y) tuple of 1-D arrays."
        )
    return np.arange(y.size, dtype=np.float64), y.astype(np.float64)


def _thin(x, y):
    """At most MAX_POINTS points that still show every peak and trough.

    With x in order, each of MAX_POINTS / 2 buckets keeps its lowest and its
    highest point, so a waveform keeps its envelope instead of aliasing. With x
    out of order, every k-th point is kept.
    """
    np = _np()
    n = y.size
    if n <= MAX_POINTS:
        return x, y
    if not np.all(np.diff(x) >= 0):
        keep = np.linspace(0, n - 1, MAX_POINTS).round().astype(int)
        return x[keep], y[keep]
    buckets = MAX_POINTS // 2
    edges = _edges(n, buckets)
    xs, ys = [], []
    filled = np.where(np.isfinite(y), y, np.nan)
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        chunk = filled[start:stop]
        if np.all(np.isnan(chunk)):
            xs.append(x[start])
            ys.append(np.nan)
            continue
        lo, hi = int(np.nanargmin(chunk)), int(np.nanargmax(chunk))
        for k in sorted({lo, hi}):
            xs.append(x[start + k])
            ys.append(chunk[k])
    return np.asarray(xs), np.asarray(ys)


def show_plot(
    *series: Any,
    label: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    labels: Sequence[str] | None = None,
    x_scale: str = "linear",
    y_scale: str = "linear",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Send a line plot of up to six series.

    Each positional argument is one series: a 1-D array of y values, plotted
    against its index, or an `(x, y)` tuple of 1-D arrays. `labels` names the
    series for the legend. A series longer than 2000 points keeps the lowest
    and highest point of each of 1000 buckets. `x_scale` and `y_scale` are
    `"linear"` or `"log"`; a log axis skips values that are not positive.
    """
    if not series:
        raise ValueError("show_plot needs at least one series")
    if len(series) > MAX_SERIES:
        raise ValueError(f"show_plot draws at most {MAX_SERIES} series, got {len(series)}")
    for scale in (x_scale, y_scale):
        if scale not in ("linear", "log"):
            raise ValueError(f"an axis scale must be 'linear' or 'log', got {scale!r}")
    if labels is not None and len(labels) != len(series):
        raise ValueError(f"got {len(labels)} labels for {len(series)} series")

    notes: list[str] = []
    out = []
    for index, item in enumerate(series):
        x, y = _series(item, index)
        if y.size == 0:
            raise ValueError(f"series {index} is empty")
        tx, ty = _thin(x, y)
        out.append(
            {
                "label": _label(labels[index]) if labels is not None else None,
                "x": [_num(float(v)) for v in tx],
                "y": [_num(float(v)) for v in ty],
                "points": int(y.size),
            }
        )

    def limits(pair: tuple[float, float] | None) -> list[float | None] | None:
        if pair is None:
            return None
        lo, hi = pair
        return [_num(float(lo)), _num(float(hi))]

    _send(
        {
            "type": "figure",
            "kind": "line",
            "label": _label(label),
            "xlabel": _label(xlabel),
            "ylabel": _label(ylabel),
            "x_scale": x_scale,
            "y_scale": y_scale,
            "xlim": limits(xlim),
            "ylim": limits(ylim),
            "series": out,
            "note": " ".join(notes) or None,
        }
    )


def show_waveform(samples: Any, sample_rate: float, label: str | None = None) -> None:
    """Plot a clip against time in seconds, one line per channel."""
    np = _np()
    audio = _array(samples, "samples").astype(np.float64)
    if audio.ndim == 1:
        audio = audio[None, :]
    if audio.ndim != 2 or audio.shape[0] > MAX_SERIES:
        raise ValueError(f"samples must be 1-D or (channels, samples), got shape {audio.shape}")
    t = np.arange(audio.shape[1]) / float(sample_rate)
    names = None if audio.shape[0] == 1 else [f"channel {i + 1}" for i in range(audio.shape[0])]
    show_plot(
        *[(t, channel) for channel in audio],
        label=label,
        xlabel="Time (s)",
        ylabel="Amplitude",
        labels=names,
    )


# Assets ----------------------------------------------------------------


def assets_root() -> Path | None:
    """The directory `SPEECH_ASSETS` names, or None when it is unset."""
    root = os.environ.get("SPEECH_ASSETS", "").strip()
    return Path(root) if root else None


def has_asset(name: str) -> bool:
    """Whether the asset, or a file inside it such as `whisper-tiny/config.json`, is here."""
    root = assets_root()
    return bool(root) and _resolve(root, name).exists()


def asset_path(name: str) -> Path:
    """The path of a baked-in asset, or a file inside one.

    The GPU worker downloads every entry in `gpu/assets.json` into
    `/assets/<name>` when its image is built, so a lab can load a model or a
    clip there without the network. On the CPU runner `SPEECH_ASSETS` points at
    the repository's `assets/` directory, which is empty unless someone ran
    `python gpu/fetch_assets.py gpu/assets.json assets`. Check `has_asset` first
    when a lab should fall back to something smaller instead of failing.
    """
    root = assets_root()
    if root is None:
        raise FileNotFoundError("SPEECH_ASSETS is not set, so this runner has no assets")
    path = _resolve(root, name)
    if not path.exists():
        raise FileNotFoundError(
            f"The asset {name!r} is not on this runner. Assets listed in "
            "gpu/assets.json are baked into the GPU worker; on the CPU runner, "
            "fetch them with `python gpu/fetch_assets.py gpu/assets.json assets`."
        )
    return path


def _resolve(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise ValueError(f"asset names stay inside the assets directory, got {name!r}")
    return path


def read_audio(path: str | os.PathLike[str]) -> tuple[Any, int]:
    """Read a sound file as float32 in [-1, 1], with its sample rate.

    Returns a 1-D array for a mono file and a (channels, samples) array
    otherwise, the same layout `play` takes. It uses `soundfile` when it is
    installed, which reads WAV, FLAC, and Ogg, and falls back to the standard
    library for 16-bit PCM WAV.
    """
    np = _np()
    try:
        import soundfile
    except ImportError:
        soundfile = None
    if soundfile is not None:
        data, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        data = data.T
    else:
        with wave.open(str(path), "rb") as source:
            if source.getsampwidth() != 2:
                raise ValueError("without soundfile, only 16-bit PCM WAV can be read")
            rate = source.getframerate()
            channels = source.getnchannels()
            raw = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
        data = (raw.reshape(-1, channels).T / 32768.0).astype(np.float32)
    return (data[0] if data.shape[0] == 1 else data), int(rate)
