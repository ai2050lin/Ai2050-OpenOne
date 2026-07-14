#!/usr/bin/env python3
"""Check Phase418 interface-history canvases are nonblank and interactive."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


ROOT = Path(__file__).resolve().parents[2]
PREFIX = os.environ.get("ATLAS_PREFIX", "phase418")
SCREENSHOTS = ROOT / os.environ.get(
    "ATLAS_SCREENSHOT_DIR", "tests/gpt5/result/phase418_interface_history_atlas/screenshots"
)
CHECK_PHASE_ID = os.environ.get(
    "ATLAS_PIXEL_PHASE_ID", "Phase418-InterfaceHistoryAtlasCanvasPixelCheck"
)
CHECK_FILENAME = os.environ.get(
    "ATLAS_PIXEL_FILENAME", "phase418_interface_history_canvas_pixel_check.json"
)
EXPECTED_DATASET_COUNT = int(os.environ.get("ATLAS_EXPECTED_DATASET_COUNT", "3"))
REQUIRE_ALL_PAIRWISE_DISTINCT = os.environ.get(
    "ATLAS_REQUIRE_ALL_PAIRWISE_DISTINCT", "1"
) not in {"0", "false", "False"}


def image_metrics(path: Path) -> tuple[dict, Image.Image]:
    image = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(image)
    pixels = list(image.resize((160, 100)).get_flattened_data())
    background = pixels[0]
    ratio = sum(
        max(abs(pixel[channel] - background[channel]) for channel in range(3)) > 8
        for pixel in pixels
    ) / len(pixels)
    return {
        "name": path.name,
        "width": image.width,
        "height": image.height,
        "rgb_standard_deviation": [round(value, 6) for value in stat.stddev],
        "non_background_ratio": round(ratio, 6),
        "nonblank": max(stat.stddev) > 5 and ratio > 0.05,
    }, image


def main() -> None:
    rows = []
    interactions = []
    desktop = []
    for path in sorted(SCREENSHOTS.glob(f"{PREFIX}_*_canvas.png")):
        row, image = image_metrics(path)
        rows.append(row)
        if "desktop_1440x900" in path.name:
            desktop.append((path.name, image.resize((240, 150))))
        after_path = path.with_name(path.name.removesuffix("_canvas.png") + "_canvas_interaction.png")
        after = Image.open(after_path).convert("RGB")
        difference = ImageChops.difference(image, after)
        mean_difference = sum(ImageStat.Stat(difference).mean) / 3
        interactions.append({
            "before": path.name,
            "after": after_path.name,
            "mean_absolute_rgb_difference": round(mean_difference, 6),
            "camera_interaction_visible": mean_difference > 0.2,
        })
    pairwise = []
    for index, (left_name, left) in enumerate(desktop):
        for right_name, right in desktop[index + 1:]:
            difference = ImageChops.difference(left, right)
            mean_difference = sum(ImageStat.Stat(difference).mean) / 3
            pairwise.append({
                "left": left_name,
                "right": right_name,
                "mean_absolute_rgb_difference": round(mean_difference, 6),
                "model_visuals_distinct": mean_difference > 0.5,
            })
    pairwise_gate = (
        all(row["model_visuals_distinct"] for row in pairwise)
        if REQUIRE_ALL_PAIRWISE_DISTINCT
        else any(row["model_visuals_distinct"] for row in pairwise)
    )
    payload = {
        "phase_id": CHECK_PHASE_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": bool(
            len(rows) == EXPECTED_DATASET_COUNT + 1
            and all(row["nonblank"] for row in rows)
            and len(interactions) == EXPECTED_DATASET_COUNT + 1
            and all(row["camera_interaction_visible"] for row in interactions)
            and len(pairwise) == EXPECTED_DATASET_COUNT * (EXPECTED_DATASET_COUNT - 1) // 2
            and pairwise_gate
        ),
        "require_all_pairwise_distinct": REQUIRE_ALL_PAIRWISE_DISTINCT,
        "screenshots": rows,
        "interaction_checks": interactions,
        "desktop_model_pairwise_checks": pairwise,
    }
    output = SCREENSHOTS / CHECK_FILENAME
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
