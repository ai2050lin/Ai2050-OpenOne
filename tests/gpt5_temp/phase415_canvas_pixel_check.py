#!/usr/bin/env python3
"""Check that every Phase415 route produces a nonblank, interactive canvas."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


ROOT = Path(__file__).resolve().parents[2]
SCREENSHOTS = (
    ROOT / "tests/gpt5/result/phase415_multi_route_vis_sources/screenshots"
)


def image_metrics(path: Path) -> tuple[dict, Image.Image]:
    image = Image.open(path).convert("RGB")
    stat = ImageStat.Stat(image)
    pixels = list(image.resize((160, 100)).get_flattened_data())
    background = pixels[0]
    non_background_ratio = sum(
        max(abs(pixel[channel] - background[channel]) for channel in range(3)) > 8
        for pixel in pixels
    ) / len(pixels)
    return (
        {
            "name": path.name,
            "width": image.width,
            "height": image.height,
            "rgb_standard_deviation": [round(value, 6) for value in stat.stddev],
            "non_background_ratio": round(non_background_ratio, 6),
            "nonblank": max(stat.stddev) > 5 and non_background_ratio > 0.05,
        },
        image,
    )


def main() -> None:
    rows = []
    interaction_rows = []
    route_images: list[tuple[str, Image.Image]] = []
    for path in sorted(SCREENSHOTS.glob("phase415_*_canvas.png")):
        row, image = image_metrics(path)
        rows.append(row)
        route_images.append((path.name, image.resize((240, 150))))
        interaction_path = path.with_name(
            path.name.removesuffix("_canvas.png") + "_canvas_interaction.png"
        )
        if interaction_path.is_file():
            interaction = Image.open(interaction_path).convert("RGB")
            difference = ImageChops.difference(image, interaction)
            mean_difference = sum(ImageStat.Stat(difference).mean) / 3
            interaction_rows.append(
                {
                    "before": path.name,
                    "after": interaction_path.name,
                    "mean_absolute_rgb_difference": round(mean_difference, 6),
                    "camera_interaction_visible": mean_difference > 0.2,
                }
            )

    pairwise_rows = []
    desktop_routes = [
        (name, image)
        for name, image in route_images
        if "desktop_1440x900" in name
    ]
    for index, (left_name, left) in enumerate(desktop_routes):
        for right_name, right in desktop_routes[index + 1 :]:
            difference = ImageChops.difference(left, right)
            mean_difference = sum(ImageStat.Stat(difference).mean) / 3
            pairwise_rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "mean_absolute_rgb_difference": round(mean_difference, 6),
                    "route_visuals_distinct": mean_difference > 0.5,
                }
            )

    payload = {
        "phase_id": "Phase415-MultiRouteCanvasPixelCheck",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": bool(
            len(rows) == 5
            and all(row["nonblank"] for row in rows)
            and len(interaction_rows) == 5
            and all(row["camera_interaction_visible"] for row in interaction_rows)
            and len(pairwise_rows) == 6
            and all(row["route_visuals_distinct"] for row in pairwise_rows)
        ),
        "screenshots": rows,
        "interaction_checks": interaction_rows,
        "desktop_route_pairwise_checks": pairwise_rows,
    }
    output = SCREENSHOTS / "phase415_canvas_pixel_check.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
