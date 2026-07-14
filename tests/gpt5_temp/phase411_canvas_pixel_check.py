#!/usr/bin/env python3
"""Check that Phase411 desktop and mobile WebGL captures are nonblank."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


ROOT = Path(__file__).resolve().parents[2]
SCREENSHOTS = ROOT / "tests/gpt5/result/phase411_finite_operation_preflight/screenshots"


def main() -> None:
    rows = []
    interaction_rows = []
    for path in sorted(SCREENSHOTS.glob("*_canvas.png")):
        image = Image.open(path).convert("RGB")
        stat = ImageStat.Stat(image)
        pixels = list(image.resize((160, 100)).get_flattened_data())
        background = pixels[0]
        non_background_ratio = sum(
            max(abs(pixel[index] - background[index]) for index in range(3)) > 8
            for pixel in pixels
        ) / len(pixels)
        rows.append(
            {
                "name": path.name,
                "width": image.width,
                "height": image.height,
                "rgb_mean": [round(value, 6) for value in stat.mean],
                "rgb_standard_deviation": [round(value, 6) for value in stat.stddev],
                "non_background_ratio": round(non_background_ratio, 6),
                "nonblank": max(stat.stddev) > 5 and non_background_ratio > 0.05,
            }
        )
        interaction_path = path.with_name(
            path.name.removesuffix("_canvas.png") + "_canvas_interaction.png"
        )
        if interaction_path.is_file():
            interaction = Image.open(interaction_path).convert("RGB")
            difference = ImageChops.difference(image, interaction)
            difference_stat = ImageStat.Stat(difference)
            mean_difference = sum(difference_stat.mean) / 3
            interaction_rows.append(
                {
                    "before": path.name,
                    "after": interaction_path.name,
                    "mean_absolute_rgb_difference": round(mean_difference, 6),
                    "camera_interaction_visible": mean_difference > 0.2,
                }
            )
    payload = {
        "phase_id": "Phase411-CanvasPixelCheck",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": (
            len(rows) == 2
            and all(row["nonblank"] for row in rows)
            and len(interaction_rows) == 2
            and all(row["camera_interaction_visible"] for row in interaction_rows)
        ),
        "screenshots": rows,
        "interaction_checks": interaction_rows,
    }
    (SCREENSHOTS / "phase411_canvas_pixel_check.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
