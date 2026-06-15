import pytest


def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(
                min(
                    prev[j + 1] + 1,
                    curr[j] + 1,
                    prev[j] + (0 if c1 == c2 else 1),
                )
            )
        prev = curr
    return prev[-1]


def merge_similar(plate_reads, max_edit_distance=2):
    # Local simplified copy mirroring scripts/run_alpr.py merge behavior.
    plates = sorted(plate_reads.keys(), key=lambda p: len(plate_reads[p]), reverse=True)
    merged = {}
    assigned = set()

    for plate in plates:
        if plate in assigned:
            continue
        group_reads = list(plate_reads[plate])
        assigned.add(plate)
        for other in plates:
            if other in assigned:
                continue
            if abs(len(plate) - len(other)) > max_edit_distance:
                continue
            if levenshtein(plate, other) <= max_edit_distance:
                group_reads.extend(plate_reads[other])
                assigned.add(other)
        merged[plate] = group_reads
    return merged


def test_fuzzy_merge_collapses_ocr_variants_into_one_plate():
    # Expected: OCR variants should collapse to a single canonical bucket.
    plate_reads = {
        "DL7CD5017": [(0.90, 10), (0.88, 11)],
        "DL7CDS017": [(0.70, 12)],
        "DL7CD5O17": [(0.65, 13)],
        "DL3CBJ1384": [(0.80, 20)],
        "DL2CAT4762": [(0.85, 30)],
        "HR26CQ6869": [(0.83, 40)],
    }

    merged = merge_similar(plate_reads, max_edit_distance=2)

    # The first three keys should merge into one bucket.
    # We don't assert the exact canonical key; only that total merged buckets is 4.
    assert len(merged) == 4

    # Total reads in merged buckets should equal original reads.
    total_reads = sum(len(v) for v in merged.values())
    assert total_reads == sum(len(v) for v in plate_reads.values())

