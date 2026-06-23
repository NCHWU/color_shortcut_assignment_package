#!/usr/bin/env python3
"""
Generate a simple synthetic control dataset for testing color shortcut learning.

True task:
    Classify the shape: circle vs square.

Shortcut variable:
    Color. In the shortcut train split, most squares are red and most circles
    are blue. In the counterfactual split, this color-label correlation is
    reversed while the true label remains shape.

Example:
    python generate_color_shortcut_dataset.py --output color_shortcut_shapes --overwrite

Output structure:
    color_shortcut_shapes/
      metadata.json
      labels.csv
      examples_grid.png
      train_shortcut/
        circle/*.png
        square/*.png
      test_iid/
        circle/*.png
        square/*.png
      test_counterfactual/
        circle/*.png
        square/*.png
      test_balanced_control/
        circle/*.png
        square/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

Shape = str
ColorName = str
RGB = Tuple[int, int, int]

SHAPES: Tuple[Shape, Shape] = ("circle", "square")
LABELS: Dict[Shape, int] = {"circle": 0, "square": 1}
BASE_COLORS: Dict[ColorName, RGB] = {
    "red": (220, 45, 45),
    "blue": (45, 90, 220),
}

SPLITS = {
    "train_shortcut": "same_as_training",
    "test_iid": "same_as_training",
    "test_counterfactual": "reversed",
    "test_balanced_control": "balanced",
}


def parse_rgb(value: str) -> RGB:
    """Parse a color string like '245,245,245'."""
    try:
        parts = tuple(int(x.strip()) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("RGB values must be integers, e.g. 245,245,245") from exc
    if len(parts) != 3 or any(x < 0 or x > 255 for x in parts):
        raise argparse.ArgumentTypeError("RGB values must contain three numbers between 0 and 255")
    return parts  # type: ignore[return-value]


def clamp(value: int) -> int:
    return max(0, min(255, value))


def jitter_rgb(rng: random.Random, color: RGB, jitter: int) -> RGB:
    """Slightly vary the shade while keeping the named color recognizable."""
    if jitter <= 0:
        return color
    return tuple(clamp(channel + rng.randint(-jitter, jitter)) for channel in color)  # type: ignore[return-value]


def dominant_color_for(split_name: str, shape: Shape) -> ColorName | None:
    """
    Return the majority color for a shape in a split.

    Training shortcut rule:
        square -> red
        circle -> blue

    Counterfactual rule reverses this relation.
    Balanced split has no dominant color.
    """
    split_type = SPLITS[split_name]
    if split_type == "balanced":
        return None

    training_rule = {"square": "red", "circle": "blue"}
    if split_type == "same_as_training":
        return training_rule[shape]
    if split_type == "reversed":
        return "blue" if training_rule[shape] == "red" else "red"

    raise ValueError(f"Unknown split type: {split_type}")


def build_color_sequence(
    rng: random.Random,
    split_name: str,
    shape: Shape,
    n: int,
    shortcut_probability: float,
) -> List[ColorName]:
    """Create an exact color list for one shape in one split."""
    dominant = dominant_color_for(split_name, shape)

    if dominant is None:
        colors = ["red"] * (n // 2) + ["blue"] * (n - n // 2)
    else:
        majority_count = int(round(n * shortcut_probability))
        minority_count = n - majority_count
        minority = "blue" if dominant == "red" else "red"
        colors = [dominant] * majority_count + [minority] * minority_count

    rng.shuffle(colors)
    return colors


def random_center_for_object(
    rng: random.Random,
    image_size: int,
    object_size: int,
    shape: Shape,
    rotation_degrees: float,
) -> Tuple[float, float]:
    """Sample a center point so the object stays fully inside the image."""
    half = object_size / 2.0
    if shape == "square":
        # A rotated square needs a larger safety margin.
        margin = int(math.ceil(half * math.sqrt(2))) + 2
    else:
        margin = int(math.ceil(half)) + 2

    cx = rng.uniform(margin, image_size - margin)
    cy = rng.uniform(margin, image_size - margin)
    return cx, cy


def draw_shape_image(
    rng: random.Random,
    image_size: int,
    background: RGB,
    shape: Shape,
    color_rgb: RGB,
    min_object_size: int,
    max_object_size: int,
) -> Tuple[Image.Image, Dict[str, float]]:
    """Draw one image and return it together with its generation parameters."""
    object_size = rng.randint(min_object_size, max_object_size)

    # Squares are rotated slightly so that the dataset is less pixel-perfect.
    # Circles are rotation-invariant, so the value is kept at 0 for metadata clarity.
    rotation_degrees = rng.uniform(-35.0, 35.0) if shape == "square" else 0.0
    cx, cy = random_center_for_object(rng, image_size, object_size, shape, rotation_degrees)

    image = Image.new("RGB", (image_size, image_size), background)
    draw = ImageDraw.Draw(image)

    if shape == "circle":
        radius = object_size / 2.0
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=color_rgb,
        )
    elif shape == "square":
        half = object_size / 2.0
        theta = math.radians(rotation_degrees)
        base_points = [(-half, -half), (half, -half), (half, half), (-half, half)]
        rotated_points = []
        for x, y in base_points:
            xr = x * math.cos(theta) - y * math.sin(theta)
            yr = x * math.sin(theta) + y * math.cos(theta)
            rotated_points.append((cx + xr, cy + yr))
        draw.polygon(rotated_points, fill=color_rgb)
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    params = {
        "object_size": object_size,
        "center_x": round(cx, 3),
        "center_y": round(cy, 3),
        "rotation_degrees": round(rotation_degrees, 3),
    }
    return image, params


def create_example_grid(output_dir: Path, rows: Sequence[Dict[str, object]], image_size: int) -> None:
    """Create a small visual grid of examples for the blog post."""
    selected: List[Dict[str, object]] = []

    # Prefer examples from all four splits, with both shapes represented when possible.
    for split in SPLITS:
        for shape in SHAPES:
            match = next(
                (row for row in rows if row["split"] == split and row["shape"] == shape),
                None,
            )
            if match is not None:
                selected.append(match)

    if not selected:
        return

    font = ImageFont.load_default()
    cell_w = image_size + 18
    cell_h = image_size + 38
    cols = 4
    rows_n = math.ceil(len(selected) / cols)
    grid = Image.new("RGB", (cols * cell_w, rows_n * cell_h), (255, 255, 255))
    drawer = ImageDraw.Draw(grid)

    for idx, row in enumerate(selected):
        x0 = (idx % cols) * cell_w
        y0 = (idx // cols) * cell_h
        img = Image.open(output_dir / str(row["filepath"])).convert("RGB")
        grid.paste(img, (x0 + 9, y0 + 5))
        label = f"{row['split'].replace('test_', '')}\n{row['shape']}, {row['color_name']}"
        drawer.multiline_text((x0 + 9, y0 + image_size + 8), label, fill=(0, 0, 0), font=font, spacing=1)

    grid.save(output_dir / "examples_grid.png")


def write_csv(output_dir: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "split",
        "filepath",
        "label",
        "shape",
        "color_name",
        "color_rgb",
        "is_training_shortcut_aligned",
        "object_size",
        "center_x",
        "center_y",
        "rotation_degrees",
    ]
    with (output_dir / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    split_sizes: Dict[str, int],
) -> None:
    metadata = {
        "dataset_name": "color_shortcut_shapes",
        "task": "Classify shape: circle=0, square=1",
        "shortcut_variable": "color",
        "training_shortcut_rule": "square is usually red; circle is usually blue",
        "splits": {
            "train_shortcut": {
                "description": "Training split with a strong color-label shortcut.",
                "color_rule": f"P(red|square)={args.shortcut_probability}, P(blue|circle)={args.shortcut_probability}",
                "size": split_sizes["train_shortcut"],
            },
            "test_iid": {
                "description": "IID test split with the same shortcut as training.",
                "color_rule": f"P(red|square)={args.shortcut_probability}, P(blue|circle)={args.shortcut_probability}",
                "size": split_sizes["test_iid"],
            },
            "test_counterfactual": {
                "description": "Counterfactual test split where the color-label shortcut is reversed.",
                "color_rule": f"P(blue|square)={args.shortcut_probability}, P(red|circle)={args.shortcut_probability}",
                "size": split_sizes["test_counterfactual"],
            },
            "test_balanced_control": {
                "description": "Control test split where color is independent of shape.",
                "color_rule": "P(red|shape)=0.5 and P(blue|shape)=0.5 for both shapes",
                "size": split_sizes["test_balanced_control"],
            },
        },
        "image_generation": {
            "image_size": args.image_size,
            "background_rgb": args.background,
            "min_object_size": args.min_object_size,
            "max_object_size": args.max_object_size,
            "color_jitter": args.color_jitter,
            "square_rotation_range_degrees": [-35, 35],
            "seed": args.seed,
        },
        "citation_motivation": [
            {
                "title": "Shortcut Learning in Deep Neural Networks",
                "authors": "Geirhos et al.",
                "year": 2020,
                "url": "https://arxiv.org/abs/2004.07780",
            },
            {
                "title": "Invariant Risk Minimization",
                "authors": "Arjovsky et al.",
                "year": 2019,
                "url": "https://arxiv.org/abs/1907.02893",
            },
        ],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def validate_args(args: argparse.Namespace) -> None:
    split_sizes = [args.train_size, args.iid_test_size, args.counterfactual_test_size, args.balanced_test_size]
    if any(size <= 0 for size in split_sizes):
        raise ValueError("All split sizes must be positive.")
    if any(size % 2 != 0 for size in split_sizes):
        raise ValueError("All split sizes must be even so circle and square classes are balanced exactly.")
    if not (0.5 <= args.shortcut_probability <= 1.0):
        raise ValueError("shortcut_probability must be between 0.5 and 1.0.")
    if args.min_object_size <= 0 or args.max_object_size <= 0:
        raise ValueError("Object sizes must be positive.")
    if args.min_object_size > args.max_object_size:
        raise ValueError("min_object_size cannot be larger than max_object_size.")
    max_safe_size = int(args.image_size * 0.62)
    if args.max_object_size > max_safe_size:
        raise ValueError(
            f"max_object_size={args.max_object_size} is too large for image_size={args.image_size}. "
            f"Use at most {max_safe_size}."
        )


def generate_dataset(args: argparse.Namespace) -> None:
    validate_args(args)
    rng = random.Random(args.seed)
    output_dir: Path = args.output

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --overwrite to replace it.")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, object]] = []
    split_sizes = {
        "train_shortcut": args.train_size,
        "test_iid": args.iid_test_size,
        "test_counterfactual": args.counterfactual_test_size,
        "test_balanced_control": args.balanced_test_size,
    }

    image_id = 0
    for split_name, total_size in split_sizes.items():
        per_shape = total_size // 2
        for shape in SHAPES:
            split_shape_dir = output_dir / split_name / shape
            split_shape_dir.mkdir(parents=True, exist_ok=True)
            color_sequence = build_color_sequence(
                rng=rng,
                split_name=split_name,
                shape=shape,
                n=per_shape,
                shortcut_probability=args.shortcut_probability,
            )

            for idx, color_name in enumerate(color_sequence):
                color_rgb = jitter_rgb(rng, BASE_COLORS[color_name], args.color_jitter)
                image, params = draw_shape_image(
                    rng=rng,
                    image_size=args.image_size,
                    background=args.background,
                    shape=shape,
                    color_rgb=color_rgb,
                    min_object_size=args.min_object_size,
                    max_object_size=args.max_object_size,
                )

                filename = f"{split_name}_{shape}_{idx:05d}_{color_name}.png"
                relative_path = Path(split_name) / shape / filename
                image.save(output_dir / relative_path)

                is_training_shortcut_aligned = (
                    (shape == "square" and color_name == "red")
                    or (shape == "circle" and color_name == "blue")
                )

                row = {
                    "split": split_name,
                    "filepath": str(relative_path),
                    "label": LABELS[shape],
                    "shape": shape,
                    "color_name": color_name,
                    "color_rgb": color_rgb,
                    "is_training_shortcut_aligned": int(is_training_shortcut_aligned),
                    **params,
                }
                all_rows.append(row)
                image_id += 1

    write_csv(output_dir, all_rows)
    write_metadata(output_dir, args, split_sizes)
    create_example_grid(output_dir, all_rows, args.image_size)

    print(f"Created dataset: {output_dir.resolve()}")
    print(f"Images: {len(all_rows)}")
    print(f"Metadata: {output_dir / 'metadata.json'}")
    print(f"Labels: {output_dir / 'labels.csv'}")
    print(f"Example grid: {output_dir / 'examples_grid.png'}")


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic color-shortcut shape dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=Path("color_shortcut_shapes"), help="Output dataset directory.")
    parser.add_argument("--overwrite", action="store_true", help="Delete the output directory if it already exists.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    parser.add_argument("--image-size", type=int, default=64, help="Width and height of generated square images.")
    parser.add_argument("--background", type=parse_rgb, default=(245, 245, 245), help="Background RGB color as R,G,B.")
    parser.add_argument("--min-object-size", type=int, default=18, help="Minimum circle diameter / square side length.")
    parser.add_argument("--max-object-size", type=int, default=30, help="Maximum circle diameter / square side length.")
    parser.add_argument("--color-jitter", type=int, default=18, help="Random +/- jitter applied to each RGB channel.")

    parser.add_argument("--shortcut-probability", type=float, default=0.95, help="Majority color probability in shortcut splits.")
    parser.add_argument("--train-size", type=int, default=2000, help="Number of images in train_shortcut.")
    parser.add_argument("--iid-test-size", type=int, default=500, help="Number of images in test_iid.")
    parser.add_argument("--counterfactual-test-size", type=int, default=500, help="Number of images in test_counterfactual.")
    parser.add_argument("--balanced-test-size", type=int, default=500, help="Number of images in test_balanced_control.")
    return parser


def main() -> None:
    parser = make_arg_parser()
    args = parser.parse_args()
    generate_dataset(args)


if __name__ == "__main__":
    main()
