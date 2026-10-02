"""
Parse TextGrid phoneme intervals and speaker metadata into
data/pipeline/phoneme_tokens.csv.
"""

import os
import re
import pandas as pd
import tgt

# Paths
CORPUS_ROOT   = os.path.join("ru-fr_interference", "2")
METADATA_PATH = os.path.join(CORPUS_ROOT, "metadata_RUFR.csv")
SPEAKERS_DIR  = os.path.join(
    CORPUS_ROOT, "wav_et_textgrids", "FRcorp_textgrids_only"
)
OUTPUT_PATH   = os.path.join("data", "pipeline", "phoneme_tokens.csv")

# Regex to extract the sentence number from filenames like:
#   ab_fra_list1_FRcorp3.TextGrid
SENTENCE_ID_RE = re.compile(r"FRcorp(\d+)\.TextGrid$", re.IGNORECASE)


def parse_textgrid(tg_path: str) -> list[tuple[str, float, float]]:
    """
    Extract all non-empty intervals from the 'phones' tier.
    Returns list of (phoneme_label, onset_s, offset_s)
    """
    tg = tgt.io.read_textgrid(tg_path, encoding="utf-8")
    phones_tier = tg.get_tier_by_name("phones")
    tokens = []
    for interval in phones_tier.intervals:
        label = interval.text.strip()
        if label:                          # skip silent / empty intervals
            tokens.append((label, interval.start_time, interval.end_time))
    return tokens


def main() -> None:
    # Load metadata
    metadata = pd.read_csv(METADATA_PATH, sep=",")
    # Build lookup dicts keyed by speaker id (as stored in the CSV)
    speaker_l1     = dict(zip(metadata["spk"], metadata["L1"]))
    speaker_gender = dict(zip(metadata["spk"], metadata["Gender"]))

    rows = []

    # Walk the corpus
    for speaker_id in sorted(os.listdir(SPEAKERS_DIR)):
        speaker_dir = os.path.join(SPEAKERS_DIR, speaker_id)
        if not os.path.isdir(speaker_dir):
            continue

        l1 = speaker_l1[speaker_id]     
        gender = speaker_gender[speaker_id]

        # rep_counter[(sentence_id, phoneme)] -> next repetition index (1-based)
        rep_counter: dict[tuple[int, str], int] = {}

        for filename in sorted(os.listdir(speaker_dir)):
            m = SENTENCE_ID_RE.search(filename)
            if m is None:
                continue # not a TextGrid of interest

            sentence_id = int(m.group(1))
            tg_path = os.path.join(speaker_dir, filename)

            tokens = parse_textgrid(tg_path)

            for phoneme, onset, offset in tokens:
                key = (sentence_id, phoneme)
                rep_idx = rep_counter.get(key, 1)
                rep_counter[key] = rep_idx + 1

                duration_ms = round((offset - onset) * 1000, 3)

                rows.append({
                    "speaker":          speaker_id,
                    "sentence_id":      sentence_id,
                    "repetition_index": rep_idx,
                    "phoneme":          phoneme,
                    "onset":            round(onset, 6),
                    "offset":           round(offset, 6),
                    "duration_ms":      duration_ms,
                    "L1":               l1,
                    "gender":           gender,
                })

    # Build and save DataFrame
    columns = [
        "speaker", "sentence_id", "repetition_index",
        "phoneme", "onset", "offset", "duration_ms",
        "L1", "gender",
    ]
    manifest = pd.DataFrame(rows, columns=columns)
    manifest = manifest.sort_values(
        ["speaker", "sentence_id", "phoneme", "repetition_index"]
    ).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    manifest.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(manifest)} phoneme tokens to '{OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()