"""
Add spectral-centre-of-gravity features to the acoustic table.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth


INPUT_PATH = Path("data/pipeline/features_acoustic.csv")
OUTPUT_PATH = Path("data/pipeline/features_acoustic_extended.csv")
SPEAKERS_DIR = Path("ru-fr_interference/2/wav_et_textgrids/FRcorp_textgrids_only")
MID_WINDOW_SECONDS = 0.025


def find_wav(speaker: str, sentence_id: int) -> Path:
    pattern = SPEAKERS_DIR / speaker / f"*FRcorp{sentence_id}.wav"
    return Path(glob.glob(str(pattern))[0])


def extract_part(sound, start: float, end: float):
    start = max(start, sound.xmin)
    end = min(end, sound.xmax)
    if end <= start:
        return None
    return sound.extract_part(from_time=start, to_time=end, preserve_times=False)


def scg(segment) -> float:
    if segment is None:
        return np.nan
    try:
        value = segment.to_spectrum().get_centre_of_gravity(power=2.0)
        return float(value) if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    sound_cache = {}
    scg_mean = []
    scg_mid = []
    half_window = MID_WINDOW_SECONDS / 2.0

    for idx, row in df.iterrows():
        speaker = str(row["speaker"])
        sentence_id = int(row["sentence_id"])
        onset = float(row["onset"])
        offset = float(row["offset"])
        midpoint = (onset + offset) / 2.0

        key = (speaker, sentence_id)
        if key not in sound_cache:
            sound_cache[key] = parselmouth.Sound(str(find_wav(speaker, sentence_id)))

        sound = sound_cache[key]

        scg_mean.append(scg(extract_part(sound, onset, offset)))
        scg_mid.append(scg(extract_part(sound, midpoint - half_window, midpoint + half_window)))

        if (idx + 1) % 5000 == 0:
            print(f"processed {idx + 1:,}/{len(df):,} tokens")

    df["SCG_mean"] = scg_mean
    df["SCG_mid"] = scg_mid

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()