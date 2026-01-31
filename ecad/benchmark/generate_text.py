# generate_text.py
"""
Use saved prompt "embeddings" (typically tokenized input_ids / attention_mask in .pt files)
to generate text for one or many schedule JSONs.

This mirrors ecad/scripts/generate_images.py, but targets TextGenerator subclasses.

Expected directory layout:
  - input_dir: contains .pt files produced by TextGenerator.encode_and_save_prompts()
  - schedule_dir: contains .json schedules (or nested directories of schedules)
  - output_dir: will contain a subdir per schedule, mirroring schedule_dir structure if schedule_dir is nested

Each prompt's generations are saved as .txt (or whatever your TextGenerator.save_text uses).
"""

import argparse
from pathlib import Path

from ecad.image_generators.text_generator import TextGenerator
from ecad.image_generators.load_text_generator import (
    TextGeneratorRegistry,
    get_text_generator_type,
)


def _count_text_files(output_dir: Path) -> int:
    # Count common text artifacts. Adjust if you use jsonl, etc.
    patterns = ["*.txt", "*.jsonl", "*.json"]
    total = 0
    for p in patterns:
        total += len(list(output_dir.glob(p)))
    return total


def generate_for_schedule(
    text_generator_type: type[TextGenerator],
    schedule_file: Path,
    embedding_dir: Path,
    output_dir: Path,
    batch_size: int,
    num_generations_per_prompt: int,
    seed: int,
    seed_step: int,
    regen_not_n_texts: int | None = None,
    dont_include_seed: bool = False,
    max_new_tokens: int | None = None,
):
    """
    Generate text for a single schedule JSON.

    Skips if output exists unless regen_not_n_texts is set and count != expected.
    """
    if output_dir.exists() and ((found := _count_text_files(output_dir)) > 0):
        if regen_not_n_texts is not None and found != regen_not_n_texts:
            print(
                f"Regenerating text for schedule {schedule_file.stem}; "
                f"expected {regen_not_n_texts} outputs, found {found}."
            )
            for pat in ["*.txt", "*.jsonl", "*.json"]:
                for f in output_dir.glob(pat):
                    f.unlink()
        else:
            print(
                f"Skipping schedule {schedule_file.stem} because output directory already exists. "
                f"Expected {regen_not_n_texts if regen_not_n_texts is not None else 'any number'}, "
                f"found {found} files."
            )
            return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n\n\nGenerating text for schedule: {schedule_file.stem}.\n")

    try:
        text_generator = text_generator_type(  # type: ignore
            start_seed=seed,
            seed_step=seed_step,
            schedule_path=schedule_file,
        )
    except Exception as e:
        raise ValueError(
            "Error creating text generator; check init args for the chosen generator.",
            text_generator_type,
        ) from e

    # The TextGenerator base we discussed provides generate_from_saved_prompts(...)
    # that calls generate_text(...) underneath and writes outputs to disk.
    text_generator.generate_from_saved_prompts(
        embedding_dir,
        output_dir,
        batch_size=batch_size,
        generations_per_prompt=num_generations_per_prompt,
        include_seed_in_name=(not dont_include_seed),
        max_new_tokens=max_new_tokens,
    )


def generate_all_schedules(
    text_generator_type: type[TextGenerator],
    schedule_dir: Path,
    embedding_dir: Path,
    output_dir: Path,
    batch_size: int,
    num_generations_per_prompt: int,
    seed: int,
    seed_step: int,
    regen_not_n_texts: int | None = None,
    dont_include_seed: bool = False,
    max_new_tokens: int | None = None,
):
    """
    Recurse over schedule_dir:
      - if a single .json schedule file, generate into output_dir/<schedule_stem>
      - if schedule_dir contains only directories, mirror structure into output_dir
      - otherwise, glob **/*.json
    """
    if schedule_dir.is_file() and schedule_dir.suffix == ".json":
        generate_for_schedule(
            text_generator_type,
            schedule_dir,
            embedding_dir,
            output_dir / schedule_dir.stem,
            batch_size,
            num_generations_per_prompt,
            seed,
            seed_step,
            regen_not_n_texts,
            dont_include_seed,
            max_new_tokens,
        )
        return

    if schedule_dir.is_dir() and all((subdir.is_dir() for subdir in schedule_dir.iterdir())):
        output_subdir = output_dir / schedule_dir.name
        for subdir in schedule_dir.iterdir():
            if subdir.is_dir():
                generate_all_schedules(
                    text_generator_type,
                    subdir,
                    embedding_dir,
                    output_subdir,
                    batch_size,
                    num_generations_per_prompt,
                    seed,
                    seed_step,
                    regen_not_n_texts,
                    dont_include_seed,
                    max_new_tokens,
                )
        return

    for schedule_file in schedule_dir.glob("**/*.json"):
        generate_for_schedule(
            text_generator_type,
            schedule_file,
            embedding_dir,
            output_dir / schedule_file.stem,
            batch_size,
            num_generations_per_prompt,
            seed,
            seed_step,
            regen_not_n_texts,
            dont_include_seed,
            max_new_tokens,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Use saved prompt embeddings (usually tokenized .pt files) to generate text.\n"
            "A subdirectory under the output directory will be created for each JSON schedule found "
            "in the schedule directory."
        )
    )
    parser.add_argument(
        "--image-generator",
        type=str,
        required=True,
        help="The name of the text generator to use.",
        choices=list(TextGeneratorRegistry.registry.keys()),
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Path to directory containing saved prompt embeddings in <name>.pt format.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save generated text to. A subdirectory is created per schedule.",
    )
    parser.add_argument(
        "-d",
        "--schedule-dir",
        type=Path,
        required=True,
        help="Directory (or a single .json file) containing schedules in <schedule-name>.json format.",
    )
    parser.add_argument(
        "-n",
        "--num-images-per-prompt",
        type=int,
        default=10,
        help="Number of generations to produce per prompt.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size while generating text.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed to start each prompt's generation with.",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=1,
        help="Increment seed by this amount for each subsequent generation of the same prompt.",
    )
    parser.add_argument(
        "--regen-if-not-n-images",
        type=int,
        required=False,
        help="Force re-generating outputs for schedules whose directories do not contain exactly n output files.",
    )
    parser.add_argument(
        "--dont-include-seed",
        action="store_true",
        help="Do not include the seed in generated filenames.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        required=False,
        help="Override max_new_tokens passed into the generator (if supported).",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory {args.input_dir} not found.")

    if not args.schedule_dir.exists():
        raise FileNotFoundError(f"Schedule directory/file {args.schedule_dir} not found.")

    if not args.output_dir.exists():
        print(f"Output directory {args.output_dir} not found. Creating.")
        args.output_dir.mkdir(parents=True, exist_ok=True)

    text_generator_type = get_text_generator_type(args.image_generator)
    
    print('\nUsing', text_generator_type,'\n')

    generate_all_schedules(
        text_generator_type,
        args.schedule_dir,
        args.input_dir,
        args.output_dir,
        args.batch_size,
        args.num_images_per_prompt,
        args.seed,
        args.seed_step,
        args.regen_if_not_n_images,
        args.dont_include_seed,
        args.max_new_tokens,
    )

    print("Done.")


if __name__ == "__main__":
    main()
