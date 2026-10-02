"""
Extract F1/F2 midpoint and long-vowel trajectory formants from phoneme intervals.
Input: data/pipeline/phoneme_tokens.csv
Output: data/pipeline/features_acoustic.csv
"""

import glob
import os

import numpy as np
import pandas as pd
import parselmouth

# Paths
INPUT_PATH   = os.path.join("data", "pipeline", "phoneme_tokens.csv")
OUTPUT_PATH  = os.path.join("data", "pipeline", "features_acoustic.csv")
SPEAKERS_DIR = os.path.join(
    "ru-fr_interference", "2",
    "wav_et_textgrids", "FRcorp_textgrids_only",
)

# LPC parameters
N_FORMANTS         = 5
MAX_FORMANT_FEMALE = 5000   # Hz
MAX_FORMANT_MALE   = 4500   # Hz
WINDOW_LENGTH      = 0.025  # seconds (Praat default)
PRE_EMPHASIS       = 50.0   # Hz (Praat default)

# Trajectory threshold
LONG_VOWEL_MS = 80   # tokens with duration_ms > this get 25%/75% extraction


def get_max_formant(gender: str) -> int:
    """Return the appropriate max_formant ceiling based on speaker gender."""
    if isinstance(gender, str) and gender.strip().lower() == "f":
        return MAX_FORMANT_FEMALE
    return MAX_FORMANT_MALE


def find_wav(speaker_dir: str, sentence_id: int) -> str | None:
    """
    Glob for the unique WAV file matching *FRcorp{sentence_id}.wav inside
    speaker_dir.  Returns the path, or None if not found.
    """
    pattern = os.path.join(speaker_dir, f"*FRcorp{sentence_id}.wav")
    matches = glob.glob(pattern)
    if len(matches) == 1:
        return matches[0]
    return None


def safe_formant_value(formant: parselmouth.Formant, number: int, time: float) -> float:
    """
    Query a formant object at a given time point.
    Returns NaN on any failure (tracker undefined, out-of-range time, etc.).
    """
    value = formant.get_value_at_time(number, time)
    return float(value) if (value is not None and not np.isnan(value)) else np.nan

def main() -> None:
    # Load Stage 1 output
    tokens = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(tokens)} phoneme tokens from '{INPUT_PATH}'.")

    f1_mid_vals: list[float] = []
    f2_mid_vals: list[float] = []
    f1_25_vals:  list[float] = []
    f2_25_vals:  list[float] = []
    f1_75_vals:  list[float] = []
    f2_75_vals:  list[float] = []

    # Caches to avoid loading the same Sound/Formant multiple times per speaker/sentence
    sound_cache: dict[tuple[str, int], parselmouth.Sound | None] = {}
    formant_cache: dict[tuple[str, int], parselmouth.Formant | None] = {}

    # Iterate over tokens
    for _, row in tokens.iterrows():
        speaker     = row["speaker"]
        sent_id     = int(row["sentence_id"])
        onset       = float(row["onset"])
        offset      = float(row["offset"])
        duration_ms = float(row["duration_ms"])
        gender      = row["gender"]
        max_formant = get_max_formant(gender)

        midpoint = (onset + offset) / 2.0
        t25      = onset + 0.25 * (offset - onset)
        t75      = onset + 0.75 * (offset - onset)
        is_long  = duration_ms > LONG_VOWEL_MS

        cache_key = (speaker, sent_id)

        # Load Sound once per (speaker, sentence)
        if cache_key not in sound_cache:
            speaker_dir = os.path.join(SPEAKERS_DIR, speaker)
            wav_path = find_wav(speaker_dir, sent_id)
            sound_cache[cache_key] = parselmouth.Sound(wav_path)

        # Compute Formant once per (speaker, sentence)
        if cache_key not in formant_cache:
            snd = sound_cache[cache_key]
            if snd is None:
                formant_cache[cache_key] = None
            else:
                formant_cache[cache_key] = snd.to_formant_burg(
                    time_step=None,
                    max_number_of_formants=N_FORMANTS,
                    maximum_formant=max_formant,
                    window_length=WINDOW_LENGTH,
                    pre_emphasis_from=PRE_EMPHASIS,
                )

        formant = formant_cache[cache_key]

        # Extract values
        f1_mid_vals.append(safe_formant_value(formant, 1, midpoint))
        f2_mid_vals.append(safe_formant_value(formant, 2, midpoint))

        if is_long:
            f1_25_vals.append(safe_formant_value(formant, 1, t25))
            f2_25_vals.append(safe_formant_value(formant, 2, t25))
            f1_75_vals.append(safe_formant_value(formant, 1, t75))
            f2_75_vals.append(safe_formant_value(formant, 2, t75))
        else:
            f1_25_vals.append(np.nan)
            f2_25_vals.append(np.nan)
            f1_75_vals.append(np.nan)
            f2_75_vals.append(np.nan)

    # Assemble output DataFrame
    result = tokens.copy()
    result["F1_mid"] = f1_mid_vals
    result["F2_mid"] = f2_mid_vals
    result["F1_25"]  = f1_25_vals
    result["F2_25"]  = f2_25_vals
    result["F1_75"]  = f1_75_vals
    result["F2_75"]  = f2_75_vals

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(result)} rows to '{OUTPUT_PATH}'.")

if __name__ == "__main__":
    main()