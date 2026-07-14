#!/usr/bin/env python3
"""Verify Phase416 model canvases are nonblank, interactive, and distinct."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


ROOT = Path(__file__).resolve().parents[2]
SCREENSHOTS = ROOT / "tests/gpt5/result/phase416_formal_world_physical_atlas/screenshots"


def metrics(path: Path) -> tuple[dict, Image.Image]:
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
    for path in sorted(SCREENSHOTS.glob("phase416_*_canvas.png")):
        row, image = metrics(path)
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

    payload = {
        "phase_id": "Phase416-PrefillAtlasCanvasPixelCheck",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": bool(
            len(rows) == 4
            and all(row["nonblank"] for row in rows)
            and len(interactions) == 4
            and all(row["camera_interaction_visible"] for row in interactions)
            and len(pairwise) == 3
            and all(row["model_visuals_distinct"] for row in pairwise)
        ),
        "screenshots": rows,
        "interaction_checks": interactions,
        "desktop_model_pairwise_checks": pairwise,
    }
    output = SCREENSHOTS / "phase416_prefill_atlas_canvas_pixel_check.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
