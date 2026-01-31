import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

import torch
import torch.nn.functional as F 
import pandas as pd

from ecad.benchmark.generate_embeddings import read_benchmark_prompts

DEFAULT_BENCHMARK_DIR = Path(__file__).parent / "../../results/benchmark/"
DEFAULT_BENCHMARK_PROMPTS = DEFAULT_BENCHMARK_DIR / "benchmark-prompts.json"
DEFAULT_IMAGE_DIR = DEFAULT_BENCHMARK_DIR / "full-benchmark" / "images"

# Pattern to extract prompt id and image seed from filenames.
FILENAME_PATTERN = re.compile(
    r".*__prompt_id:(?P<prompt_id>.+?)__.*?__image_seed:(?P<image_seed>\d+)"
)
FILENAME_PATTERN_PARTI = re.compile(
    r"(?P<prompt_num>\d+)__prompt_seed:(?P<prompt_seed>.+?)__image_seed:(?P<image_seed>\d+)"
)
FILENAME_PATTERN_TOCA = re.compile(r"(?P<prompt_num>\d+)__.*")
FILENAME_PATTERN_TOCA_SEEDED = re.compile(
    r"(?P<prompt_num>\d+)__.*?image_seed:(?P<image_seed>\d+)"
)


# GSM8K extraction 

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

# def load_image_reward_model() -> RM.ImageReward:
#     print("Loading ImageReward model.")
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     model: RM.ImageReward = RM.load("ImageReward-v1.0", device=device)
#     model = torch.compile(model, fullgraph=True)  # type: ignore

#     print("Model loaded.")
#     return model


def save_to_file(input_path: Path, output_subpath: Path, data: dict[str, Any]):
    output_path = input_path / output_subpath
    # Ensure parent directories exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    i = 1
    while output_path.exists():
        print(f"File {output_path} already exists.")
        output_path = output_path.parent / (
            output_path.stem + f"({i})" + output_path.suffix
        )
        i += 1

    print(f"Saving score data to {output_path}.")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def fname_to_info(input_dir: Path) -> dict[str, dict[str, str]]:
    print(f"Reading from {input_dir}.\nUsing pattern {FILENAME_PATTERN}.\n")
    fname_to_prompt_id = {}
    for file in input_dir.glob("*.txt"):
        name = file.stem
        match = FILENAME_PATTERN.match(name)
        if match is None:
            print(f"Invalid filename: {name}")
            continue
        prompt_id = match.group("prompt_id")
        dlm_seed = match.group("dlm_seed")
        fname_to_prompt_id[str(file.resolve())] = {
            "prompt_id": prompt_id,
            "dlm_seed": dlm_seed,
        }
    return fname_to_prompt_id


@torch.inference_mode()
def score_dir(
    input_dir: Path,
    tokenizer: AutoTokenizer,
    text_scoring_mode: str,
) -> dict[str, dict[int, float]]:
    if text_scoring_mode == "ce_loss":
        return score_dir_text_ce_loss(input_dir, 
                                      tokenizer=tokenizer)

    raise ValueError(f"Got invalid image naming mode: {text_scoring_mode}")


@torch.inference_mode()
def score_dir_text_ce_loss(
    input_dir: Path,
    tokenizer: Any,
    
) -> dict[str, dict[int, float]]: # want to return nll between gold standard answer after #### in GSM8k vs. model output. 
    infos = fname_to_info(input_dir)
    score_by_prompt_id = defaultdict(dict)

    for fname, info in infos.items():
        question_id = info["question_id"]  # type: ignore
        dlm_seed = info["dlm_seed"]  # type: ignore

        with open(fname, "r") as f:
            answer_text = f.read()
            gt_answer = extract_answer(answer_text)

        gold_tokens = tokenizer(gt_answer, return_tensors="pt")
        gold_answer = gold_tokens.input_ids.squeeze()

        predicted_text = extract_answer(fname) # extract logits
        predicted_tokens = tokenizer(predicted_text, return_tensors="pt")
        predicted_answer = predicted_tokens.input_ids.squeeze()
        # convert saved predicted text to logits

        score = -F.cross_entropy(predicted_answer, gold_answer).item()
        score_by_prompt_id[question_id][dlm_seed] = score

    return score_by_prompt_id


def avg_per_prompt(
    score_by_prompt_id: dict[str, dict[int, float]]
) -> dict[str, float]:
    return {
        prompt_id: sum(info.values()) / len(info)
        for prompt_id, info in score_by_prompt_id.items()
    }


def avg_overall(score_by_prompt_id: dict[str, dict[int, float]]) -> float:
    total = sum(sum(info.values()) for info in score_by_prompt_id.values())
    num = sum(len(info) for info in score_by_prompt_id.values())
    print(
        f"Summed score across all images: {total}, number of images scored: {num}"
    )
    if num == 0:
        print("ERROR: No scores found.")
        return 0
    return total / num


def score_dirs_recursive(
    input_dir: Path,
    output_subpath: Path,
    tokenizer: AutoTokenizer,
    text_scoring_mode: str,
    delete_after: bool = False,
    exactly_n_images: int | None = None,
    rescore_existing: bool = False,
) -> None:
    if not input_dir.is_dir():
        return

    num_texts = len(list(input_dir.glob("*.txt")))
    if num_texts > 0:
        if exactly_n_images is not None and num_texts != exactly_n_images:
            print(
                f"ERROR: Directory {input_dir} has wrong number of texts. Expected {exactly_n_images}, found {num_texts}."
            )
        elif not rescore_existing and (input_dir / output_subpath).exists():
            print(f"Skipping {input_dir} because it already has a score file.")
        else:
            score_by_prompt_id = score_dir(
                input_dir,
                tokenizer,
                text_scoring_mode,
            )
            avg_by_prompt = avg_per_prompt(score_by_prompt_id)
            total_score = avg_overall(score_by_prompt_id)
            full_data = {
                "total_score": total_score,
                "avg_by_prompt": avg_by_prompt,
                "score_by_prompt_id": score_by_prompt_id,
            }
            save_to_file(input_dir, output_subpath, full_data)
            print(f"Total Score Data for {input_dir}: {total_score}")
            if delete_after:
                print(f"Deleting images in {input_dir}.")
                for file in input_dir.glob("*.png"):
                    file.unlink()

    for sub_dir in input_dir.iterdir():
        score_dirs_recursive(
            sub_dir,
            output_subpath,
            tokenizer,
            text_scoring_mode,
            delete_after,
            exactly_n_images,
        )


def get_score_text_argparser() -> argparse.ArgumentParser:
    """
    Returns an ArgumentParser preloaded with score-images–specific arguments.
    This parser can be used standalone or merged into a multi-job parser.
    """
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument(
        "--text-dir",
        type=Path,
        help="Directory containing texts to be scored.",
    )
    parser.add_argument(
        "--output-subpath",
        "-o",
        type=Path,
        default=Path("scores.json"),
        help="Filename or subpath for the generated score JSON file.",
    )
    parser.add_argument(
        "--delete-after",
        action="store_true",
        help="Delete the texts within the directory after scoring.",
    )
    parser.add_argument(
        "--exactly-n-texts",
        "-n",
        type=int,
        help="Only score directories with exactly n texts.",
    )
    parser.add_argument(
        "--rescore-existing",
        action="store_true",
        help="Rescore directories that already have a score file.",
    )
    parser.add_argument(
        "--text-scoring-mode",
        choices=["image_reward", "parti", "toca"],
        default="image_reward",
        help="The pattern to use to match image filenames.",
    )
    parser.add_argument(
        "--file-mode",
        choices=["json", "text", "txt", "tsv"],
        help="JSON is the Image Reward prompts mode. TSV must have a column named 'Prompt', and have a header line. Text format is a simple text file with a prompt on each line.",
    )
    return parser


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score benchmark texts using the ImageReward model."
    )
    score_parser = get_score_text_argparser()
    # Merge score-specific arguments into the main parser.
    for action in score_parser._actions:
        parser._add_action(action)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        "Dream-org/Dream-v0-Instruct-7B",
        trust_remote_code=True
    )
        
    score_dirs_recursive(
        args.text_dir,
        args.output_subpath,
        tokenizer, # model replaced with tokenizer
        args.text_scoring_mode,
        args.delete_after,
        args.exactly_n_texts,
        args.rescore_existing,
    )
    print("Done.")


if __name__ == "__main__":
    main()
