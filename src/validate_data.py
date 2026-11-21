from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

EXPECTED_SPLITS = {"train": 1034, "validation": 133, "test": 128}
EXPECTED_CLASSES = {"angular_leaf_spot", "bean_rust", "healthy"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_dataset(raw_root: Path) -> dict:
    """Validate the raw dataset and return a deterministic quality report.

    Required Core checks:
      1. required splits exist;
      2. exactly the expected classes exist in every split;
      3. expected image counts are present;
      4. every image can be fully decoded by Pillow (`Image.load()` or equivalent);
      5. no split or class is empty.

    TODO: implement the checks and return a JSON-serializable report.
    The report must not contain timestamps because it is a DVC output.
    """
    raw_root = Path(raw_root)
    splits_report: dict[str, dict]= {}
    total_images = 0

    # Sorted order everywhere so the same files always give the same report.
    for split in sorted(EXPECTED_SPLITS):
        split_dir = raw_root / split

        # Check 1: the split exist:
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")

        # Check 2: exactly the expected classes exist in every split:
        found_classes = {p.name for p in split_dir.iterdir() if p.is_dir()}
        if found_classes != EXPECTED_CLASSES:
            missing = sorted(EXPECTED_CLASSES - found_classes)
            extra = sorted(found_classes - EXPECTED_CLASSES)
            raise RuntimeError(
                f"Split {split} has the wrong classes"
                f"(missing: {missing}, unexpected: {extra})"
            )

        class_counts: dict[str,int] = {}
        for class_name in sorted(EXPECTED_CLASSES):
            images = sorted(path for path in (split_dir/class_name).rglob("*")
                            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)

            # Check 5 : no class folder is empty
            if not images:
                raise RuntimeError(f"Class {class_name} in split '{split} is empty")

            # Check 4: Decode every pixel not just the file header
            for image_path in images:
                try: 
                    with Image.open(image_path) as image:
                        image.load()
                except OSError as error:
                    raise OSError(f"Cannot read image {image_path}: {error}") from error
            class_counts[class_name] = len(images)

        # Check 3 : the split exactly the expected number of image
        split_total = sum(class_counts.values())
        if split_total != EXPECTED_SPLITS[split]:
            raise RuntimeError(
                f"Split '{split}' has {split_total} images, "
                f"expected {EXPECTED_SPLITS[split]}"
            )

        splits_report[split] = {"images": split_total, "class_counts": class_counts}
        total_images += split_total

    return {
        "status": "PASS",
        "classes": sorted(EXPECTED_CLASSES),
        "splits": splits_report,
        "total_images": total_images
    }

        


def main() -> None:
    report = validate_dataset(Path("data/raw"))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
