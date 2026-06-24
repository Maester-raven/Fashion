#!/usr/bin/env python3

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_preview(path, max_size):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image.copy()


def wrapped(text, width=55):
    return "\n".join(textwrap.wrap(text, width=width))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--groups",
        type=Path,
        default=Path(
            "data_manifests/v1/fashionai_visual_dedup/"
            "fashionai_visual_duplicate_groups.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "data_manifests/v1/fashionai_visual_review"
        ),
    )

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(args.groups)

    near_groups = [
        group
        for group in data["groups"]
        if group.get("near_duplicate_edges")
    ]

    font = ImageFont.load_default()
    decisions = {}

    for group in near_groups:
        members = group["members"]
        edge = group["near_duplicate_edges"][0]

        panel_width = 570
        panel_height = 650
        margin = 20

        canvas = Image.new(
            "RGB",
            (
                panel_width * len(members),
                panel_height + 100,
            ),
            "white",
        )

        draw = ImageDraw.Draw(canvas)

        title = (
            f'{group["group_id"]} | '
            f'members={group["member_count"]} | '
            f'pHash={edge["phash_distance"]} | '
            f'dHash={edge["dhash_distance"]} | '
            f'hist={edge["histogram_l1"]:.4f}'
        )

        draw.text(
            (margin, 10),
            title,
            fill="black",
            font=font,
        )

        for index, member in enumerate(members):
            left = index * panel_width + margin

            preview = load_preview(
                member["image_abs_path"],
                (520, 520),
            )

            image_left = (
                left + (520 - preview.width) // 2
            )
            image_top = (
                55 + (520 - preview.height) // 2
            )

            canvas.paste(
                preview,
                (image_left, image_top),
            )

            label = (
                f'Image {index + 1}\n'
                f'{wrapped(member["image_path"])}\n'
                f'Size: {member["width"]}x'
                f'{member["height"]}'
            )

            draw.multiline_text(
                (left, 590),
                label,
                fill="black",
                font=font,
                spacing=4,
            )

        instruction = (
            "Decision: merge = same source image; "
            "separate = merely visually similar"
        )

        draw.text(
            (margin, panel_height + 65),
            instruction,
            fill="red",
            font=font,
        )

        output_path = (
            args.out_dir
            / f'{group["group_id"]}.jpg'
        )

        canvas.save(
            output_path,
            quality=92,
        )

        decisions[group["group_id"]] = {
            "decision": "pending",
            "reason": "",
            "members": [
                member["image_path"]
                for member in members
            ],
            "review_image": str(output_path),
        }

    write_json(
        args.out_dir / "review_decisions.json",
        {
            "allowed_decisions": [
                "merge",
                "separate",
            ],
            "instructions": {
                "merge":
                    "同一张原始商品图，仅压缩、缩放、轻微裁边或重新编码。",
                "separate":
                    "不同图片，只是服饰、构图、姿势或背景相似。",
            },
            "groups": decisions,
        },
    )

    print(
        json.dumps(
            {
                "near_groups": len(near_groups),
                "review_images_written":
                    len(decisions),
                "output_dir":
                    str(args.out_dir.resolve()),
                "decision_file":
                    str(
                        (
                            args.out_dir
                            / "review_decisions.json"
                        ).resolve()
                    ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
