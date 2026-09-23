"""Record or replay every neural-model output the pipeline uses.

Set TKH_MODEL_OUTPUTS=record:DIR to save each embedding and NLI call's
output to DIR as it is computed, or TKH_MODEL_OUTPUTS=replay:DIR to return
the saved outputs instead of running the models (the models are never
loaded in replay). Unset, the models run normally.

Outputs are keyed by the whole call, not by text: model revision, inputs
in order, batch size and sequence length. The same text can embed
differently in the last float bits depending on its batch, so only the
whole call reproduces bit for bit. See DESIGN_NOTES.md section 23.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ENV_VAR = "TKH_MODEL_OUTPUTS"


def mode():
    """(mode, dir) from TKH_MODEL_OUTPUTS, or (None, None) when unset."""
    value = os.environ.get(ENV_VAR, "")
    if not value:
        return None, None
    m, _, path = value.partition(":")
    if m not in ("record", "replay") or not path:
        raise ValueError(f"{ENV_VAR} must be record:DIR or replay:DIR, got {value!r}")
    return m, Path(path)


def call_key(kind, payload):
    blob = json.dumps({"kind": kind, **payload}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def cached_array(kind, payload, compute):
    """compute() normally; in record mode also save its output; in replay
    mode return the saved output without calling compute()."""
    m, root = mode()
    if m is None:
        return compute()
    path = root / f"{kind}_{call_key(kind, payload)}.npy"
    if m == "replay":
        if not path.exists():
            raise KeyError(f"no recorded {kind} output for this call ({path.name}) in {root}. "
                           f"Replay only covers calls the published pipeline makes.")
        return np.load(path)
    out = np.asarray(compute())
    root.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # a repeated identical call must give identical output; log if not
        if not np.array_equal(np.load(path), out):
            with open(root / "conflicts.log", "a", encoding="utf-8") as log:
                log.write(f"{path.name}: repeated call gave a different output\n")
    else:
        np.save(path, out)
    return out


def cached_json(name, compute):
    """Small metadata (e.g. an NLI label map) recorded next to the outputs."""
    m, root = mode()
    if m is None:
        return compute()
    path = root / f"{name}.json"
    if m == "replay":
        if not path.exists():
            raise KeyError(f"no recorded {name} in {root}")
        return json.loads(path.read_text(encoding="utf-8"))
    out = compute()
    root.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
