# TODO

## Goals
Fix the “CSV has more than 4 entries while video shows only 4 cars” issue by correcting duplicate emission / OCR-variant grouping.

## Steps
1. Update `scripts/run_alpr.py` to cluster OCR reads into vehicle passes and create final CSV rows per pass (pick best plate per pass), then fuzzy-merge within each pass. ✅
2. Update `scripts/run_pipeline.py` to ensure storage is **strictly once per `track_id`**: after successful store, mark the track as stored and prevent any further flush/store attempts. ✅
3. Add a regression test to prevent reintroducing duplicate row emission due to OCR string variations. ✅
4. Run `scripts/run_alpr.py` and compare number of unique CSV plate rows to expected 4 for `ALPR.mp4`.
5. Update TODO status during execution.



