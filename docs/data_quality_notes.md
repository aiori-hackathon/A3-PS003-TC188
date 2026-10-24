# Data-quality notes — XTraffic 2023_p01 (January), Channel 0 flow

Last updated: 2026-10-24 (revision 2: training-period duplicate detection). Labels: **verified** = measured on this machine; **interpretation** = our reading of the evidence; **design choice** = decision made for this project.

## 1. Public array row-to-sensor-ID mapping — supported but unconfirmed

- **Verified:** `2023_p01.npy` has 16,972 rows; `node_order.npy` has 16,972 unique IDs; the ID set equals `sensor_meta_feature.csv` `station_id`; `rebuild_aiori_metadata.py` reproduces the handoff's aligned metadata and client counts exactly.
- **Not verified:** that public array row `i` is the sensor `node_order.npy[i]`. No export script, manifest or maintainer statement links them (handoff §8).
- **Consequence:** every `station_id`, `District`, `County` and `proposed_client` in our outputs comes from `node_order[array_index]` and is labelled provisional (`id_mapping_status`). The centralized and IID-like arms do not depend on this mapping. Geographic non-IID conclusions do and must carry the caveat.
- **Weak indirect support (interpretation):** see §2. Copied-series groups never span Districts, which would be unlikely if the mapping scrambled rows. This is circumstantial, not proof.

## 2. Copied (byte-identical) flow series — 1,902 sensors

**Finding (verified).** Hashing each sensor's full January Channel 0 series (SHA-256 of the float32 bytes) shows:

| Measure | Value |
|---|---:|
| Sensors whose series is byte-identical to at least one other sensor | **1,902** (of 14,406 with any observation) |
| Distinct copied series (groups) | 97 |
| Group size | min 2, median 6, mean 19.6, max **449**; 24 groups are pairs |
| Groups identical on **all three** channels (flow, occupancy, speed) | 92 of 97 |
| Members that are fully observed (0 missing steps) | 1,902 of 1,902 |
| Groups spanning more than one District | **0** |
| Groups spanning more than one freeway / direction | 63 / 78 |
| Sensor type of members | HOV 702, Mainline 1,200 |
| Members by District | 3: 65, 4: 225, 6: 9, 7: 1,070, 8: 257, 10: 80, 12: 196 |
| Members by provisional client | LA/Ventura 1,070; Inland 411; Bay Area 225; Coastal 196 |

The largest group is 449 sensors in District 7 (434 HOV + 15 Mainline, on 18 freeways), all with the same series: mean flow 82.74 and no zeros.

**Interpretation (not proven).** Independent loop detectors on different freeways and directions are very unlikely to produce bit-for-bit identical 5-minute series for a month. Most likely these are upstream fills, for example a district-level default or imputed profile, especially for HOV lanes, rather than measurements. This is an inference: an individual excluded sensor is **not** established as invalid, and a group could contain one genuine original. They are also why "fully observed" overstates real coverage: 1,902 of the 10,613 fully observed sensors are copies. The handoff (§5) did not report this.

**Why it matters.**
- Copies in train and test inflate apparent accuracy, because the model sees the "same sensor" many times.
- Under the IID-like partition, copies split across clients would put identical data on several clients, weakening the IID vs non-IID comparison.
- Because copies cluster by District (LA/Ventura holds 56% of them), they would distort the geographic arm in particular.

**Handling (design choice; revision 2, 2026-10-24).**
- `aiori/flow_audit.py` computes `series_sha256`, `duplicate_group_size` and `duplicate_series` on the **training-period** series (steps [0, 6336)), since primary eligibility now uses the training period only. The whole-month equivalents are kept as informational `month_*` columns.
- **Training-period result:** 1,944 sensors in 104 groups, the largest 452.
  - All 1,902 whole-month duplicates are included.
  - 42 more sensors are identical only through 22 January.
  - One group spans Districts 7 and 10. It is a degenerate pair with 4.5% training coverage and all-zero observations, caught first by the `constant` rule. No non-degenerate group spans Districts.
- **Exclusion policy:** **all** members of a copied group are excluded from the primary experiment (`exclusion_reason = duplicate_series`) **as a precaution**, because the original, if any, cannot be identified. Keeping one representative per group is an alternative we have not adopted.
- **Effect:** in revision 1, 3 of the first 40 pilot sensors were copies and were replaced. The current primary pilot has 40 distinct series (verified by `scripts/verify_artifacts.py`).

**Limitations.** Only exact copies are detected. Series that were imputed but are not identical (for example scaled or noised copies, or partially imputed periods) remain undetected. Before scaling beyond the pilot or adding months, rerun the check per month and consider a near-duplicate test (for example correlation ≈ 1 or an identical day-profile).

## 3. Other Channel 0 facts (verified, match handoff §5)

NaN cells 26,459,359 (17.46%); 0 infinities; 3,627,470 zero readings (2.90% of observed, treated as valid); 10,613 fully observed / 3,793 partially observed / 2,566 fully missing / 11 constant sensors.

## 4. Preprocessing conventions that follow from these findings

- The primary pilot has 100% January coverage, so fill and masking never trigger on it. A separate 8-sensor missing-data validation set (99–<100% January coverage, `data/processed/missing_validation/`) exercises them on real gaps:
  - 121 filled inputs: 102 forward-filled, 19 filled with the time-of-day median;
  - 1,452 masked target cells.

  The training-mean fallback and dropping windows with zero observed targets are still exercised only by unit tests.
- Zero flow is never converted to missing; missing targets are never filled (NaN + mask).
