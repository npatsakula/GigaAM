"""
Classify how a streaming model's output differs from the offline model's.

Runs both models on a manifest (batched, GPU), aligns the normalized
transcripts word by word and buckets every difference by *kind* and by *where
in the clip* it happens (using the offline model's own word timestamps), so you
can tell errors that context tricks might fix (utterance start/end, first or
last character of a word) from errors that only more training fixes
(mid-utterance substitutions, number formatting).

    python error_analysis.py --streaming_ckpt <ckpt> --offline v3_e2e_ctc \\
        --manifest /data/golos/manifest_val.tsv --limit 2000 --dump errors.tsv

``--dump`` saves (path, duration, dataset text, offline, streaming, offline
word times) so the analysis can be re-run offline with ``--from_dump errors.tsv``.
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_PUNCT = str.maketrans("", "", ".,?!:;\"'«»()[]-–—")


def normalize(text: str) -> str:
    return " ".join(text.lower().replace("ё", "е").translate(_PUNCT).split())


def align(ref: Sequence[str], hyp: Sequence[str]) -> List[Tuple[str, int, int]]:
    """Levenshtein alignment with backtrace. Returns ops as
    (kind, ref_index, hyp_index) with kind in {ok, sub, del, ins};
    ref_index / hyp_index are -1 when the op has no word on that side."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
    ops: List[Tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1, -1))
            i -= 1
        else:
            ops.append(("ins", -1, j - 1))
            j -= 1
    return ops[::-1]


def char_dist(a: str, b: str) -> int:
    import editdistance

    return editdistance.eval(a, b)


_DIGITS = re.compile(r"\d")


def classify(kind: str, r: Optional[str], h: Optional[str]) -> str:
    """Kind of difference for one aligned pair."""
    if kind == "del":
        return "word deleted"
    if kind == "ins":
        return "word inserted"
    assert r is not None and h is not None
    if bool(_DIGITS.search(r)) != bool(_DIGITS.search(h)):
        return "number formatting (digits vs words)"
    if h and r.endswith(h) and len(r) - len(h) <= 2:
        return "first char(s) dropped"
    if h and r.startswith(h) and len(r) - len(h) <= 2:
        return "last char(s) dropped"
    if r and h.startswith(r) and len(h) - len(r) <= 2:
        return "extra char(s) at end"
    if char_dist(r, h) == 1:
        return "one-char substitution"
    if char_dist(r, h) <= 2:
        return "two-char edit"
    return "other substitution"


def position(
    t: Optional[float], dur: float, start_s: float = 0.6, end_s: float = 0.6
) -> str:
    if t is None:
        return "unknown"
    if t < start_s:
        return "clip start"
    if t > dur - end_s:
        return "clip end"
    return "middle"


# ------------------------------------------------------------------ inference
def run_models(args) -> List[Dict]:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    import gigaam

    data_dir = Path(args.manifest).resolve().parent
    with open(args.manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if args.limit:
        rows = rows[: args.limit]
    paths = [
        str(
            (
                Path(r["path"])
                if Path(r["path"]).is_absolute()
                else data_dir / r["path"]
            ).resolve()
        )
        for r in rows
    ]
    offline = gigaam.load_model(
        args.offline, device=args.device, fp16_encoder=False
    ).eval()
    if Path(args.streaming_ckpt).is_file():
        stream = gigaam.load_model(
            args.streaming_ckpt, device=args.device, fp16_encoder=False
        )
    else:
        stream = gigaam.load_model(
            args.streaming_ckpt, device=args.device, fp16_encoder=False
        )
        gigaam.enable_streaming(stream)
    stream.eval()
    out: List[Dict] = []
    for s in range(0, len(paths), args.batch_size):
        chunk = paths[s : s + args.batch_size]
        wavs = [gigaam.load_audio(p).to(offline._device) for p in chunk]
        lens = torch.tensor([w.shape[-1] for w in wavs], device=offline._device)
        with torch.inference_mode():
            enc, enc_len = offline.forward(
                pad_sequence(wavs, batch_first=True).to(offline._dtype), lens
            )
            dec = offline._decode(enc, enc_len, lens, word_timestamps=True)
        hyps = stream.transcribe_streaming_batch(
            chunk,
            head_padding_seconds=args.head_padding,
            tail_padding_seconds=args.tail_padding,
            padding_noise=args.padding_noise,
        )
        for row, (text, words), hyp, wav in zip(
            rows[s : s + args.batch_size], dec, hyps, wavs
        ):
            out.append(
                {
                    "path": row["path"],
                    "duration": wav.shape[-1] / 16000,
                    "dataset": row.get("transcription", ""),
                    "offline": text,
                    "streaming": hyp.text,
                    "offline_words": [
                        (w.text, round(w.start, 3), round(w.end, 3))
                        for w in words or []
                    ],
                }
            )
        print(f"  {min(s + args.batch_size, len(paths))}/{len(paths)}", flush=True)
    return out


def save_dump(path: str, items: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(
            ["path", "duration", "dataset", "offline", "streaming", "offline_words"]
        )
        for it in items:
            w.writerow(
                [
                    it["path"],
                    f"{it['duration']:.3f}",
                    it["dataset"],
                    it["offline"],
                    it["streaming"],
                    json.dumps(it["offline_words"], ensure_ascii=False),
                ]
            )


def load_dump(path: str) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as f:
        items = []
        for r in csv.DictReader(f, delimiter="\t"):
            r["duration"] = float(r["duration"])
            r["offline_words"] = json.loads(r["offline_words"])
            items.append(r)
    return items


# ------------------------------------------------------------------- analysis
def analyze(items: List[Dict], top: int = 12) -> None:
    kinds: Counter = Counter()
    kind_pos: Dict[str, Counter] = defaultdict(Counter)
    pos_err: Counter = Counter()
    pos_words: Counter = Counter()
    dur_err: Counter = Counter()
    dur_words: Counter = Counter()
    examples: Dict[str, List[str]] = defaultdict(list)
    n_words = n_err = 0
    n_exact = n_format_only = 0
    for it in items:
        ref_raw, hyp_raw = it["offline"], it["streaming"]
        ref = normalize(ref_raw).split()
        hyp = normalize(hyp_raw).split()
        if ref_raw.strip() == hyp_raw.strip():
            n_exact += 1
        elif ref == hyp:
            n_format_only += 1
        # time of each normalized ref word: the offline words normalize to the
        # same token sequence (word count can differ when a word is only
        # punctuation; fall back to None then)
        times: List[Optional[float]] = []
        owords = [(normalize(w), s, e) for w, s, e in it["offline_words"]]
        owords = [(w, s, e) for w, s, e in owords if w]
        if len(owords) == len(ref):
            times = [s for _, s, _ in owords]
        else:
            times = [None] * len(ref)
        dur = it["duration"]
        dbucket = (
            "<2 s"
            if dur < 2
            else "2-4 s" if dur < 4 else "4-8 s" if dur < 8 else ">8 s"
        )
        n_words += len(ref)
        dur_words[dbucket] += len(ref)
        for i, t in enumerate(times):
            pos_words[position(t, dur)] += 1
        for kind, ri, hi in align(ref, hyp):
            if kind == "ok":
                continue
            r = ref[ri] if ri >= 0 else None
            h = hyp[hi] if hi >= 0 else None
            k = classify(kind, r, h)
            t = (
                times[ri]
                if ri >= 0
                else (times[min(hi, len(times) - 1)] if times and hi >= 0 else None)
            )
            p = position(t, dur)
            kinds[k] += 1
            kind_pos[k][p] += 1
            pos_err[p] += 1
            dur_err[dbucket] += 1
            n_err += 1
            if len(examples[k]) < top:
                examples[k].append(
                    f"{r!r} -> {h!r}   [{p}, {dur:.1f} s]  OFF: {ref_raw}  |  STR: {hyp_raw}"
                )

    print(
        f"\nclips: {len(items)}  exact match: {n_exact}  formatting-only diff: {n_format_only}  "
        f"differ after normalization: {len(items) - n_exact - n_format_only}"
    )
    print(
        f"normalized word gap: {100 * n_err / max(n_words, 1):.2f}%  ({n_err} errors / {n_words} words)\n"
    )
    print("errors by kind (share of all errors; split by position in clip):")
    for k, c in kinds.most_common():
        pp = kind_pos[k]
        split = ", ".join(
            f"{p}: {pp[p]}"
            for p in ("clip start", "middle", "clip end", "unknown")
            if pp[p]
        )
        print(f"  {100 * c / max(n_err, 1):5.1f}%  {c:6d}  {k:40s}  {split}")
    print("\nerror rate by position in clip (errors / words in that region):")
    for p in ("clip start", "middle", "clip end", "unknown"):
        if pos_words[p]:
            print(
                f"  {p:10s}  {100 * pos_err[p] / pos_words[p]:6.2f}%   ({pos_err[p]} / {pos_words[p]})"
            )
    print("\nerror rate by clip duration:")
    for b in ("<2 s", "2-4 s", "4-8 s", ">8 s"):
        if dur_words[b]:
            print(
                f"  {b:6s}  {100 * dur_err[b] / dur_words[b]:6.2f}%   ({dur_err[b]} / {dur_words[b]})"
            )
    for k, _ in kinds.most_common():
        print(f"\n--- examples: {k}")
        for e in examples[k]:
            print("  " + e)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--streaming_ckpt")
    p.add_argument("--offline", default="v3_e2e_ctc")
    p.add_argument("--manifest")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--head_padding", type=float, default=0.0)
    p.add_argument("--tail_padding", type=float, default=0.0)
    p.add_argument(
        "--padding_noise",
        type=float,
        default=0.0,
        help="noise amplitude for the padding (0 = zeros)",
    )
    p.add_argument(
        "--dump", help="write per-clip outputs here (TSV) for offline re-analysis"
    )
    p.add_argument("--from_dump", help="skip inference; analyze this dump")
    p.add_argument("--top", type=int, default=12, help="examples per error kind")
    args = p.parse_args()
    if args.from_dump:
        items = load_dump(args.from_dump)
    else:
        if not (args.streaming_ckpt and args.manifest):
            p.error("--streaming_ckpt and --manifest are required without --from_dump")
        items = run_models(args)
        if args.dump:
            save_dump(args.dump, items)
            print(f"wrote {args.dump}")
    analyze(items, top=args.top)


if __name__ == "__main__":
    main()
