#!/usr/bin/env python3
"""Check that Phase406 desktop and mobile WebGL captures are nonblank."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[2]
SCREENSHOTS = ROOT / "tests/gpt5/result/phase406_conditioned_sequence_state/screenshots"


def main() -> None:
    rows = []
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
    payload = {
        "phase_id": "Phase406-CanvasPixelCheck",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "valid": len(rows) == 2 and all(row["nonblank"] for row in rows),
        "screenshots": rows,
    }
    (SCREENSHOTS / "phase406_canvas_pixel_check.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
