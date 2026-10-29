"""Create all benchmark directory scaffolding with .gitkeep files.
Run once from the project root: python scripts/create_directories.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REGIONS = [
    "lacus_mortis",
    "posidonius",
    "montes_apenninus",
    "aristoteles",
    "endymion",
    "mare_crisium",
]

INTER_SUBS = ["optical", "dem", "thermal", "registration"]
BENCH_SUBS = ["tiles", "metadata"]


def main():
    created = 0
    for region in REGIONS:
        for sub in INTER_SUBS:
            d = ROOT / "data" / "intermediate" / region / sub
            d.mkdir(parents=True, exist_ok=True)
            gk = d / ".gitkeep"
            if not gk.exists():
                gk.touch()
                created += 1

        for sub in BENCH_SUBS:
            d = ROOT / "data" / "benchmark" / region / sub
            d.mkdir(parents=True, exist_ok=True)
            gk = d / ".gitkeep"
            if not gk.exists():
                gk.touch()
                created += 1

        for sub in ["validation"]:
            d = ROOT / "outputs" / region / sub
            d.mkdir(parents=True, exist_ok=True)
            gk = d / ".gitkeep"
            if not gk.exists():
                gk.touch()
                created += 1

    # Also ensure the global benchmark root exists
    manifest_dir = ROOT / "data" / "benchmark"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Done. Created {created} .gitkeep files across {len(REGIONS)} regions.")


if __name__ == "__main__":
    main()
