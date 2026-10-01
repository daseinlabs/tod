"""Minimal inference for the UTOD picker (utod-picker-12b-v1).

    scripts/fetch_weights.sh                   # bundle -> ./picker (sha256-verified)
    python inference.py --example examples/choice.json
    python inference.py --state "It is raining hard." --options "umbrella" "sunglasses" "sandals"
    python inference.py --state "What does the chart show?" --options "rising" "falling" --image chart.png

Prints calibrated probabilities over ALL options (N unbounded, order preserved, sums to 1).
Everything below is a thin CLI over ``tod.release.load`` -- the exact harness two-stage path.
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default="picker", help="release bundle dir (contains picker.json)")
    ap.add_argument("--example", help="JSON file with {state, options, question?, images?}")
    ap.add_argument("--state", help="decision state (text)")
    ap.add_argument("--options", nargs="+", help="option labels")
    ap.add_argument("--question", help="instructions (default: 'Select the single best option.')")
    ap.add_argument("--image", action="append", dest="images", help="image path (repeatable)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None, help="override serve dtype (bundle default: bfloat16)")
    ap.add_argument("--attn", default=None, choices=["chunked_eager", "eager"],
                    help="override attention (never sdpa on gemma-4-12B: wrong logits)")
    a = ap.parse_args(argv)

    if a.example:
        with open(a.example, "r", encoding="utf-8") as fh:
            ex = json.load(fh)
    elif a.state is not None and a.options:
        ex = {"state": a.state, "options": a.options, "question": a.question, "images": a.images}
    else:
        ap.error("give --example FILE, or --state and --options")

    from tod.release.load import load_picker
    picker = load_picker(a.bundle, device=a.device, dtype=a.dtype, attn=a.attn)
    probs = picker.predict(ex["state"], ex["options"], question=ex.get("question"),
                           images=ex.get("images"))

    labels = list(ex["options"].keys()) if isinstance(ex["options"], dict) else list(ex["options"])
    for lab, p in sorted(zip(labels, probs), key=lambda t: -t[1]):
        print(f"{p:.4f}  {lab}")
    json.dump({"labels": labels, "probs": probs,
               "pick": labels[max(range(len(probs)), key=probs.__getitem__)]}, sys.stderr)
    sys.stderr.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
