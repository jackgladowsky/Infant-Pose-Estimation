"""Offline result checks, optionally rescore saved predictions using external Test100 GT."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "original_flip_on": 93.6071706887983, "head": 93.5472078062278,
    "stage4_head": 93.198637527681, "original_flip_off": 91.9158670222938,
    "BN_buffers_restored": 93.35239311605797, "three_crop_TTA": 93.32060253033816,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescore", action="store_true", help="CPU COCO evaluation of all six saved outputs")
    args = parser.parse_args()
    screen = json.loads((ROOT / "reproducibility/results/small-screen.json").read_text())
    diagnostics = json.loads((ROOT / "reproducibility/results/bounded-diagnostics.json").read_text())
    baseline = json.loads((ROOT / "reproducibility/results/baseline.json").read_text())
    assert abs(baseline["AP_percent"] - EXPECTED["original_flip_on"]) < 1e-10
    rows = {**screen["results"], **diagnostics["results"]}
    for name, expected in EXPECTED.items():
        assert abs(rows[name]["AP"] - expected) < 1e-10, name
        assert abs(rows[name]["metrics_fraction"]["AP"] * 100 - expected) < 1e-10
    for name in ("head", "stage4_head"):
        assert rows[name]["training_steps"] == 600
        assert rows[name]["frozen_parameters_and_buffers_unchanged"]
    assert abs(rows["original_control"]["AP"] - baseline["AP_percent"]) < 1e-10
    assert abs(rows["trained_control"]["AP"] - rows["stage4_head"]["AP"]) < 1e-10
    if args.rescore:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        path = ROOT / "data/syrip/annotations/person_keypoints_validate_infant.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == "8b299a879a95913483498e3f2115a826ac23ebc5262c3efb8d65f63718e3194c"
        gt = COCO(str(path))
        assert len(gt.getImgIds()) == 100
        for name, expected in EXPECTED.items():
            run = "bounded-diagnostics-11556720" if name in diagnostics["results"] else "small-screen-11553889"
            path = ROOT / "output" / run / name / "results/keypoints_validate_infant_results_0.json"
            evaluator = COCOeval(gt, gt.loadRes(str(path)), "keypoints")
            evaluator.params.useSegm = None
            evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
            assert abs(evaluator.stats[0] * 100 - expected) < 1e-9, name
    print("PASS: all six AP results, final-step/freeze controls and repeated baseline controls")
    if args.rescore:
        print("PASS: independent CPU COCO rescoring of all six saved prediction files")


if __name__ == "__main__":
    main()
