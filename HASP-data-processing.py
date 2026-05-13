"""
HASP-data-processing.py — pre-processing of two SiPM teensy CSVs into a unified
per-event DataFrame for downstream coincidence-spectrum analysis.

Steps:
  1. Load both teensy CSVs.
  2. Compute a per-row event_time (earliest non-zero signal_time on that row).
  3. Merge the two teensies row-by-row by nearest event_time within a tolerance.
  4. Detect orphan rows on either side (events that didn't match across teensies).
  5. Validate that the 16-bit trigger_binary pattern agrees on matched rows.
  6. Build a unified DataFrame plus a drift-diagnostic histogram.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === File paths (edit per run) ============================================
teensy1_fp = '/path/to/sipm_teensy_1.csv'
teensy2_fp = '/path/to/sipm_teensy_2.csv'


# All plots get saved to ./results/ relative to wherever this script is run.
# exist_ok=True so re-runs don't crash if the folder is already there.
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


# === Config ===============================================================
# Max allowed time gap between matched events on the two teensies.
# Units = whatever trigger_NN_signal_time uses (TBD; assumed seconds).
# PLACEHOLDER — tune once you've seen the real drift distribution.
MERGE_TOLERANCE = 0.01

# Column-name groups, used repeatedly below.
TRIGGER_BIN_COLS  = [f'trigger_{i:02d}_binary'      for i in range(1, 17)]
TRIGGER_TIME_COLS = [f'trigger_{i:02d}_signal_time' for i in range(1, 17)]

# === Load raw teensy CSVs =================================================
df1 = pd.read_csv(teensy1_fp)
df2 = pd.read_csv(teensy2_fp)

# === Per-row event_time ===================================================
# merge_asof needs ONE time column per row to match on, but a row can have
# multiple triggers fire (coincidence → multiple signal_times). Collapse
# the 16 signal_time cols into one scalar per row. Arbitrarily choosing the min of the row,
# but specific choice of min/max/mean is arbitrary; what matters is applying the SAME rule
# to both teensies.
#   .replace(0, np.nan) → mask "did not fire" sentinels so they don't win the min
#   .min(axis=1)        → earliest signal_time among triggers that fired
df1['event_time'] = df1[TRIGGER_TIME_COLS].replace(0, np.nan).min(axis=1)
df2['event_time'] = df2[TRIGGER_TIME_COLS].replace(0, np.nan).min(axis=1)

# === Sort by event_time (required by merge_asof) ==========================
df1 = df1.sort_values('event_time').reset_index(drop=True)
df2 = df2.sort_values('event_time').reset_index(drop=True)

# === Tag original row indices so we can detect orphans ====================
# After merging, any df2 row whose original index doesn't appear in the
# merge output is a teensy-2 orphan (no df1 row within tolerance).
df1['_t1_orig_idx'] = df1.index
df2['_t2_orig_idx'] = df2.index

# === Rename teensy 2 columns to disambiguate from teensy 1 ================
# - signal_time:    suffixed on both teensies (kept per-teensy for drift)
# - cpu_temperature: suffixed on both teensies
# - binary cols:    teensy 1 kept as canonical; teensy 2 suffixed for now,
#                   used for pattern validation, then dropped.
df1 = df1.rename(columns={c: c + '_t1' for c in TRIGGER_TIME_COLS})
df1 = df1.rename(columns={'cpu_temperature': 'cpu_temperature_t1'})

df2 = df2.rename(columns={c: c + '_t2' for c in TRIGGER_TIME_COLS})
df2 = df2.rename(columns={c: c + '_t2' for c in TRIGGER_BIN_COLS})
df2 = df2.rename(columns={'cpu_temperature': 'cpu_temperature_t2'})


# === Merge: nearest event_time within tolerance ===========================
# For every df1 row, find the df2 row with the closest event_time;
# if none is within MERGE_TOLERANCE, df2 columns come back as NaN.
merged = pd.merge_asof(
    df1, df2,
    on='event_time',
    direction='nearest',
    tolerance=MERGE_TOLERANCE,
)

# === Append teensy-2 orphans ==============================================
# df2 rows whose original index didn't make it into the merge are appended
# with NaN in all teensy-1 columns, so the final frame keeps every event.
matched_t2_idx = merged['_t2_orig_idx'].dropna().astype(int).tolist()
t2_orphans = df2[~df2['_t2_orig_idx'].isin(matched_t2_idx)].copy()
merged = pd.concat([merged, t2_orphans], ignore_index=True, sort=False)

# === Build match_status column ============================================
# Four possible states per row:
#   matched          : both teensies have data and trigger patterns agree
#   pattern_mismatch : both teensies have data but patterns disagree
#   orphan_t1        : df1 row, no df2 match within tolerance
#   orphan_t2        : df2 row appended above with no df1 match
has_t1 = merged['_t1_orig_idx'].notna()
has_t2 = merged['_t2_orig_idx'].notna()

t1_pattern_arr  = merged[TRIGGER_BIN_COLS].values
t2_pattern_arr  = merged[[c + '_t2' for c in TRIGGER_BIN_COLS]].values
patterns_agree  = np.all(t1_pattern_arr == t2_pattern_arr, axis=1)

merged['match_status'] = np.select(
    condlist=[
        has_t1 & has_t2 &  patterns_agree,
        has_t1 & has_t2 & ~patterns_agree,
        has_t1 & ~has_t2,
        ~has_t1 & has_t2,
    ],
    choicelist=['matched', 'pattern_mismatch', 'orphan_t1', 'orphan_t2'],
    default='unknown',
)

# === Drop redundant teensy-2 binary columns (kept teensy-1's as canonical) ===
merged = merged.drop(columns=[c + '_t2' for c in TRIGGER_BIN_COLS])

# === Final tidy: sort by event_time =======================================
merged = merged.sort_values('event_time').reset_index(drop=True)

# === Summary print ========================================================
print("Match status counts:")
print(merged['match_status'].value_counts())
print(f"\nTotal rows in unified DataFrame: {len(merged)}")

# === Inter-teensy drift diagnostic ========================================
# For every matched event and every trigger that fired on that event,
# compute Δt = signal_time_t1 - signal_time_t2 and aggregate.
matched_only = merged[merged['match_status'] == 'matched']
drift_pieces = []
for nn in range(1, 17):
    fired = matched_only[f'trigger_{nn:02d}_binary'] == 1
    delta = (
        matched_only.loc[fired, f'trigger_{nn:02d}_signal_time_t1']
        - matched_only.loc[fired, f'trigger_{nn:02d}_signal_time_t2']
    )
    drift_pieces.append(delta)
drift = pd.concat(drift_pieces, ignore_index=True).dropna()

print("\nInter-teensy signal-time drift (t1 - t2) stats:")
print(drift.describe())

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(drift, bins=100, color='steelblue', edgecolor='black')
ax.set_xlabel('Δt = signal_time_t1 - signal_time_t2 (per fired trigger, matched events)')
ax.set_ylabel('Count')
ax.set_title('Inter-teensy signal-time drift distribution')
plt.tight_layout()
plt.show()

# `merged` is now the unified per-event DataFrame, ready to hand off to
# Scintillator_Processing along with a coincidence-group config dict.