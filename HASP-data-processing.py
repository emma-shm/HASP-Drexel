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

# === Coincidence-group config ============================================
# Maps each "column" (physical stack of 4 scints) to the trigger numbers
# believed to belong to it. EDIT THIS once colleagues confirm wiring.
# Order within each list matters: it defines the cumulative coincidence
# chain (e.g. col1_CW_1&2 uses the first two, col1_CW_1&2&3 the first three).
COINCIDENCE_GROUPS = {
    'col1': [1, 5, 9, 13],
    'col2': [2, 6, 10, 14],
    'col3': [3, 7, 11, 15],
    'col4': [4, 8, 12, 16],
}

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

# === Append teensy-2 orphans (events that didn't have time match across teensies) ==============================================
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
has_t1 = merged['_t1_orig_idx'].notna()                                    # boolean Series, one entry per row of merged: True where _t1_orig_idx is not NaN → this row has data from teensy 1
has_t2 = merged['_t2_orig_idx'].notna()                                    # same for teensy 2 → True where this row has df2 data attached (matched or orphan_t2)
t1_pattern_arr  = merged[TRIGGER_BIN_COLS].values                          # slice merged down to the 16 teensy-1 binary columns; .values strips the pandas wrapper → plain 2D NumPy array of shape (N_rows, 16) so we can do fast elementwise comparison next
t2_pattern_arr  = merged[[c + '_t2' for c in TRIGGER_BIN_COLS]].values     # same but build the teensy-2 column names on the fly by appending '_t2' to each name in TRIGGER_BIN_COLS (those were the names after the rename step) → matching (N_rows, 16) array
patterns_agree  = np.all(t1_pattern_arr == t2_pattern_arr, axis=1)         # elementwise == gives a (N_events, 16) boolean array of "does this bit agree between teensies"; np.all(..., axis=1) collapses each row to a single bool that's True only if all 16 bits agree → 1D boolean array of length N_rows

merged['match_status'] = np.select( # adding a new column 'match_status' to merged; np.select chooses values based on whether the conditions in condlist are True for each row
    condlist=[
        has_t1 & has_t2 &  patterns_agree, # if this row has data from both teensies and their trigger patterns agree, then match_status is 'matched'
        has_t1 & has_t2 & ~patterns_agree, # if this row has data from both teensies but their trigger patterns disagree (in other words, at least one of the 16 scintillator triggers doesn't match), then match_status is 'pattern_mismatch'
        has_t1 & ~has_t2, # if this row has data from teensy 1 but no matching data from teensy 2 (no df2 row within tolerance), then match_status is 'orphan_t1'
        ~has_t1 & has_t2, # if this row has data from teensy 2 but no matching data from teensy 1 (this would be the rows we appended from df2 that had no match), then match_status is 'orphan_t2'
    ],
    choicelist=['matched', 'pattern_mismatch', 'orphan_t1', 'orphan_t2'], # the corresponding values to assign to match_status for each condition
    default='unknown',
)

# === Drop redundant teensy-2 binary columns (kept teensy-1's as canonical) ===
# at this point, matches have been validated and time columns have been kept for drift analysis, so the teensy-2 binary cols are no longer needed and just take up space.
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
matched_only = merged[merged['match_status'] == 'matched'] # slicing the dataframe to only include rows where there was a time match and trigger patterns agreed
drift_pieces = []
for nn in range(1, 17): # looping through each of the 16 columns
    fired = matched_only[f'trigger_{nn:02d}_binary'] == 1 # boolean Series that's True where this trigger fired (binary column is 1) among the matched events
    delta = (matched_only.loc[fired, f'trigger_{nn:02d}_signal_time_t1'] - matched_only.loc[fired, f'trigger_{nn:02d}_signal_time_t2']) # for the rows where this trigger fired, compute the difference in signal_time between teensy 1 and teensy 2 → Series of Δt values for this trigger
    drift_pieces.append(delta) # add the column's Δt Series to the list; we'll concatenate them all together into one big Series of Δt values across all triggers
drift = pd.concat(drift_pieces, ignore_index=True).dropna() # concatenating all the Δt lists back to back, with index resetting 

print("\nInter-teensy signal-time drift (t1 - t2) stats:")
print(drift.describe())


# === Cumulative coincidence counts per column ============================
# For each configured column, build a chain of cumulative coincidence event
# counts: e.g. col1_CW_1&2 counts events where triggers 1 AND 2 both fired,
# col1_CW_1&2&3 counts events where 1 AND 2 AND 3 all fired, etc.
# These are RUNNING TOTALS (cumulative sum down the rows), matching the
# spirit of the old datalogger's "Events CW1&2" / "Events CW1&2&3" columns.
for col_name, trigger_nums in COINCIDENCE_GROUPS.items():                  # loop over each column's list of trigger numbers, e.g. col_name='col1', trigger_nums=[1,2,3,4]
    matched_mask = merged['match_status'] == 'matched'                     # COMMENT OUT THIS LINE (and the `& matched_mask` below) to include orphans/pattern-mismatches in the cumulative counts
    running_and = pd.Series(True, index=merged.index)                      # start with all-True boolean Series; we'll AND each successive trigger's binary into it to build up the coincidence condition row by row
    chain_label = ''                                                       # human-readable label suffix, grows as '1' → '1&2' → '1&2&3' → '1&2&3&4'
    for nn in trigger_nums:                                                # walk through the trigger numbers IN ORDER (order in the config list matters here)
        running_and &= (merged[f'trigger_{nn:02d}_binary'] == 1) & matched_mask  # AND this trigger's "fired?" boolean into the running condition, also AND in matched_mask so only fully-matched rows can ever count
        chain_label = f'{nn}' if not chain_label else f'{chain_label}&{nn}' # build up the label: first iteration sets it to e.g. '1', later ones append '&2', '&3', etc.
        if len(chain_label.split('&')) >= 2:                               # only emit a column once we've ANDed at least 2 triggers (a single trigger isn't really "coincidence")
            merged[f'{col_name}_CW_{chain_label}'] = running_and.cumsum()  # store the cumulative count of coincidence events down the rows (.cumsum() on a bool Series counts Trues running total); column name e.g. 'col1_CW_1&2', 'col1_CW_1&2&3', 'col1_CW_1&2&3&4'

print("\nFinal coincidence counts per column:")
for col_name, trigger_nums in COINCIDENCE_GROUPS.items():                  # print the last value of each cumulative column as a sanity check — that's the total event count for that coincidence level over the whole run
    cw_cols = [c for c in merged.columns if c.startswith(f'{col_name}_CW_')]
    for c in cw_cols:
        print(f"  {c}: {int(merged[c].iloc[-1])}")


print("\nFinal coincidence counts per column:")
for col_name, trigger_nums in COINCIDENCE_GROUPS.items():                  # print the last value of each cumulative column as a sanity check — that's the total event count for that coincidence level over the whole run
    cw_cols = [c for c in merged.columns if c.startswith(f'{col_name}_CW_')]
    for c in cw_cols:
        print(f"  {c}: {int(merged[c].iloc[-1])}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(drift, bins=100, color='steelblue', edgecolor='black')
ax.set_xlabel('Δt = signal_time_t1 - signal_time_t2 (per fired trigger, matched events)')
ax.set_ylabel('Count')
ax.set_title('Inter-teensy signal-time drift distribution')
plt.tight_layout()
plt.show()



