# Official scorer, two pinned copies

Both directories hold the organisers' `normalize.py` and `eval_lib.py` from the leaderboard Space
`TuwaiqAcademy/AISA-ArabicFC-SharedTask-Leaderboard` (Apache-2.0), copied unchanged.

- `submitted_20260623/` is the scorer as of 23 June 2026 (Space commit `3bea3275`), the version the
  development leaderboard and the submitted paper were scored with, together with data release v1.4.
- `final_20260724/` is the scorer at its last change, 24 July 2026 (Space commit `30745ed0`, and the later
  commits touch only the leaderboard interface), the version the blind test and the camera-ready
  figures use, together with data release v1.6 (dataset commit `35338790`).

Differences between the two: date fields (named days, relative terms, month names) are normalised
in the final scorer, bare currency words (ريال, درهم, جنيه, دولار) and a few names are aliased,
`search_quran.search_type` is ignored, and Maghrebi month names are recognised. `eval_lib.py` is identical.
