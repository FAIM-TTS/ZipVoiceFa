#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

def parse_args():
    p = argparse.ArgumentParser("TSV builder")
    p.add_argument("--audio-dirs", nargs="+", required=True, help="One or more roots to search for audio")
    p.add_argument("--csv-path", required=True, help="CSV/TSV with columns for text and file path")
    p.add_argument("--text-col", default="text_normalized")
    p.add_argument("--file-col", default="audio_filepath")
    p.add_argument("--sep", default=",", help="CSV separator (use '\\t' for TSV)")
    p.add_argument("--dev-frac", type=float, default=0.02)
    p.add_argument("--out-dir", default="data/raw")
    p.add_argument("--audio-exts", default=".wav,.mp3,.flac,.m4a", help="Comma-separated list")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--utt-prefix", default="", help="Optional prefix for utt_id")
    return p.parse_args()

def main():
    a = parse_args()
    AUDIO_DIRS = [Path(d) for d in a.audio_dirs]
    CSV_PATH   = Path(a.csv_path)
    OUT_DIR    = Path(a.out_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_EXTS = {e.strip().lower() if e.strip().startswith(".") else "."+e.strip().lower()
                  for e in a.audio_exts.split(",") if e.strip()}

    # load CSV
    df = pd.read_csv(CSV_PATH, sep=a.sep, engine="c")
    if a.file_col not in df.columns or a.text_col not in df.columns:
        raise SystemExit(f"Missing required columns: '{a.file_col}', '{a.text_col}'. Found: {list(df.columns)}")
    df[a.file_col] = df[a.file_col].astype(str).str.strip()
    df[a.text_col] = df[a.text_col].astype(str).fillna("").str.strip()

    # index audio
    name_map, stem_map = {}, {}
    for root in AUDIO_DIRS:
        if not root.exists(): 
            tqdm.write(f"Skip missing dir: {root}")
            continue
        for p in tqdm(root.rglob("*"), desc=f"Indexing {root}", leave=False):
            if p.suffix.lower() in AUDIO_EXTS:
                name_map.setdefault(p.name, str(p.resolve()))
                stem_map.setdefault(p.stem, str(p.resolve()))

    # resolver
    def resolve_path(pstr: str):
        p = Path(pstr)
        if p.is_absolute() and p.exists():
            return str(p.resolve())
        for root in AUDIO_DIRS:
            q = root / p
            if q.exists():
                return str(q.resolve())
        hit = name_map.get(p.name)
        if hit:
            return hit
        return stem_map.get(p.stem)

    # map + report
    df["wav_path"] = df[a.file_col].map(resolve_path)
    ok = df.dropna(subset=["wav_path"]).copy()
    miss = len(df) - len(ok)
    print(f"Matched: {len(ok):,} | Missing: {miss:,}")
    if miss:
        print("Missing examples (first 5):", ", ".join(df.loc[df["wav_path"].isna(), a.file_col].head(5).astype(str)))

    # build TSVs
    def make_utt_id(p: str) -> str:
        base = Path(p).stem
        return f"{a.utt_prefix}{base}" if a.utt_prefix else base

    ok["utt_id"] = ok["wav_path"].map(make_utt_id)
    tsv = ok[["utt_id", a.text_col, "wav_path"]].rename(columns={a.text_col: "text"}).sample(frac=1.0, random_state=a.seed)

    n_dev = max(1, int(round(a.dev_frac * len(tsv)))) if len(tsv) > 0 else 0
    dev_df  = tsv.iloc[:n_dev].copy()
    train_df= tsv.iloc[n_dev:].copy()

    train_path = OUT_DIR / "custom_train.tsv"
    dev_path   = OUT_DIR / "custom_dev.tsv"
    if len(train_df):
        train_df.to_csv(train_path, sep="\t", header=False, index=False)
        print(f"Wrote: {train_path} ({len(train_df):,})")
    if len(dev_df):
        dev_df.to_csv(dev_path,   sep="\t", header=False, index=False)
        print(f"Wrote: {dev_path}   ({len(dev_df):,})")

    if len(train_df):
        print("\nSample (train):")
        for line in Path(train_path).read_text(encoding="utf-8").splitlines()[:5]:
            print(line)

if __name__ == "__main__":
    main()
