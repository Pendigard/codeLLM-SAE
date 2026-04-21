import argparse
import io
import json
from pathlib import Path
import zipfile

import requests


ZIP_URL = "https://github.com/reddy-lab-code-research/MuST-CoST/raw/refs/heads/main/CoST_data.zip"
SPLITS = ("train", "test", "val")
CODE_FORMATS = {
    "C": ".c",
    "C++": ".cpp",
    "C#": ".cs",
    "Java": ".java",
    "Javascript": ".js",
    "PHP": ".php",
    "Python": ".py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Telecharge MuST-CoST et genere les snippets filtres."
    )
    parser.add_argument(
        "--extract-dir",
        default=".",
        help="Dossier dans lequel extraire l'archive MuST-CoST.",
    )
    parser.add_argument(
        "--output-dir",
        default="code_snippets",
        help="Dossier de sortie pour les snippets JSON filtres.",
    )
    return parser.parse_args()


def download_and_extract(zip_url: str, extract_dir: Path) -> None:
    response = requests.get(zip_url, timeout=300)
    response.raise_for_status()

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content), "r") as archive:
        archive.extractall(extract_dir)


def normalize_snippet_id(raw_id: str) -> str:
    parts = raw_id.split("-")
    return f"{parts[0]}-{parts[-1]}"


def collect_common_snippet_ids(snippet_data_dir: Path) -> set[str]:
    all_indices: list[list[str]] = []

    for language_pair_dir in snippet_data_dir.iterdir():
        if not language_pair_dir.is_dir():
            continue

        lang1, lang2 = language_pair_dir.name.split("-")
        for language in (lang1, lang2):
            if language not in CODE_FORMATS:
                continue

            pair_indices: list[str] = []
            extension = CODE_FORMATS[language]

            for split in SPLITS:
                map_file = language_pair_dir / f"{split}-{language}-map.jsonl"
                code_file = language_pair_dir / f"{split}-{language_pair_dir.name}-tok{extension}"

                if not map_file.exists() or not code_file.exists():
                    continue

                with map_file.open("r", encoding="utf-8") as handle:
                    pair_indices.extend(normalize_snippet_id(line.strip()) for line in handle if line.strip())

            if pair_indices:
                all_indices.append(pair_indices)

    if not all_indices:
        return set()

    common_ids = set(all_indices[0])
    for indices in all_indices[1:]:
        common_ids &= set(indices)
    return common_ids


def write_filtered_snippets(map_data_dir: Path, output_dir: Path, common_ids: set[str]) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for file_path in map_data_dir.glob("*mapping-tok.jsonl"):
        language = file_path.name.split("-")[0]
        language_output_dir = output_dir / language
        language_output_dir.mkdir(parents=True, exist_ok=True)

        with file_path.open("r", encoding="utf-8") as infile:
            for line in infile:
                entry = json.loads(line)
                base_id = normalize_snippet_id(entry["idx"])

                if base_id not in common_ids:
                    continue

                entry.pop("bpe", None)
                entry.pop("comment_bpe", None)
                entry.pop("desc_bpe", None)

                output_file = language_output_dir / f"{base_id}.json"
                with output_file.open("w", encoding="utf-8") as outfile:
                    json.dump(entry, outfile)
                    outfile.write("\n")
                written += 1

    return written


def main() -> None:
    args = parse_args()
    extract_dir = Path(args.extract_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    download_and_extract(ZIP_URL, extract_dir)

    data_root = extract_dir / "CoST_data_release" / "processed_data"
    snippet_data_dir = data_root / "snippet_data"
    map_data_dir = data_root / "map_data"

    common_ids = collect_common_snippet_ids(snippet_data_dir)
    written = write_filtered_snippets(map_data_dir, output_dir, common_ids)

    print(f"Archive extraite dans : {extract_dir}")
    print(f"{written} snippets filtres ecrits dans : {output_dir}")


if __name__ == "__main__":
    main()
