"""Train and evaluate the final population and nine-shot models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.perceived_safety_model import cross_validate, load_data, to_jsonable  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the experiment workbook or CSV")
    parser.add_argument("--output", default="results", help="Directory for metrics and predictions")
    parser.add_argument("--save-predictions", action="store_true", help="Write row-level OOF probabilities")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(args.data)
    result = cross_validate(data)

    summary = {
        "data_source": str(Path(args.data).resolve()),
        "conversion": "omega = 1.5 * speed_scale_factor (rad/s)",
        "features": ["omega", "distance", "distance^2", "omega*distance"],
        "n_records": int(len(data)),
        "n_participants": int(data.participant_id.nunique()),
        "n_population_oof": result["n_population_oof"],
        "n_nine_shot_evaluation": result["n_nine_shot_evaluation"],
        "metrics": {
            "population_oof": result["population_oof"],
            "nine_shot_matched_population": result["nine_shot_matched_population"],
            "nine_shot_personalized": result["nine_shot_personalized"],
        },
        "fold_details": result["fold_details"],
    }
    (output / "metrics.json").write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    result["calibration_records"].to_csv(output / "nine_shot_calibration_records.csv", index=False, encoding="utf-8-sig")

    if args.save_predictions:
        frame = result["data_with_source_row"].copy()
        for name, probabilities in (
            ("population", result["population_probabilities"]),
            ("generic_9shot", result["generic_probabilities"]),
            ("personalized_9shot", result["personalized_probabilities"]),
        ):
            for score in range(1, 11):
                frame[f"{name}_p{score}"] = probabilities[:, score - 1]
        frame.to_csv(output / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    print(json.dumps(to_jsonable(summary["metrics"]), indent=2))
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
