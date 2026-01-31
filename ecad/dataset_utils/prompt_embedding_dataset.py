from __future__ import annotations

from typing import Generic, Any, Dict, Iterable, Iterator, List, Optional, Tuple, Literal
import torch
import json
import gc
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from ecad.types import PromptEmbeddingType

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase


class PromptEmbeddingDataset(Dataset, Generic[PromptEmbeddingType]):
    def __init__(self, embedding_dir):
        """
        Initializes the dataset by loading all embedding and attention mask tensors.

        Args:
            embedding_dir (str or Path): Directory containing the .pt files with embeddings.
        """
        self.embedding_dir = Path(embedding_dir)

        self.filenames = list(self.embedding_dir.glob("**/*.pt"))

    def _get_relative_path(self, idx):
        filepath = self.filenames[idx]
        rel_path = filepath.relative_to(self.embedding_dir)
        # of the form foo/../bar/baz.pt. want foo/.../bar/
        return str(rel_path.parent)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.filenames)

    def __getitem__(self, idx) -> dict[str, torch.Tensor | str]:
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing prompt embeddings, attention masks, and their negative counterparts.
        """

        loaded_dict = torch.load(
            self.filenames[idx], weights_only=True, map_location="cpu"
        )

        return_dict = {
            "name": self.filenames[idx].stem,
            "relative_path": self._get_relative_path(idx),
        }
        for key, value in loaded_dict.items():
            if value is None:
                continue
            elif isinstance(value, torch.Tensor):
                return_dict[key] = value.squeeze()
            else:
                return_dict[key] = value

        return return_dict


@dataclass
class GSM8KTokenizedExample:
    """One tokenized prompt loaded from a .pt file."""
    name: str
    split: str
    path: Path
    input_ids: torch.LongTensor          # [L]
    attention_mask: Optional[torch.Tensor] = None  # [L] (long/bool)


class GSM8KTokenizedPtDataset(Dataset):
    """
    Dataset for pre-tokenized GSM8K prompts saved as individual .pt files.

    Expected directory layout:
      <root>/
        train/
          *.pt
        test/
          *.pt

    Each .pt should contain a dict like:
      {
        "input_ids": LongTensor [L],
        "attention_mask": LongTensor|BoolTensor [L],
        "name": str (optional),
        "prompt": str (optional)
      }

    This dataset returns *per-example* tensors (unbatched). Use the collate_fn below to pad
    into [B, L] for model.diffusion_generate(inputs=input_ids, attention_mask=...).
    """

    def __init__(
        self,
        root_dir: Path,
        split: Literal["train", "test"] = "train",
        pattern: str = "*.pt",
        strict: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        self.pattern = pattern
        self.strict = strict

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self._files: List[Path] = sorted(self.split_dir.glob(self.pattern))
        if not self._files:
            raise FileNotFoundError(f"No .pt files found in {self.split_dir} matching {self.pattern}")

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fpath = self._files[idx]
        obj = torch.load(fpath, map_location="cpu")

        if not isinstance(obj, dict):
            raise ValueError(f"{fpath} did not contain a dict; got {type(obj)}")

        if "input_ids" not in obj:
            raise KeyError(f"{fpath} missing 'input_ids'")

        input_ids = obj["input_ids"]
        if not torch.is_tensor(input_ids):
            raise TypeError(f"{fpath} 'input_ids' is not a tensor; got {type(input_ids)}")
        if input_ids.dtype != torch.long:
            # diffusion_generate expects token ids; enforce LongTensor
            input_ids = input_ids.to(torch.long)

        attention_mask = obj.get("attention_mask", None)
        if attention_mask is not None:
            if not torch.is_tensor(attention_mask):
                raise TypeError(f"{fpath} 'attention_mask' is not a tensor; got {type(attention_mask)}")
            # keep dtype; downstream may cast to bool/float as needed

        name = obj.get("name", fpath.stem)

        # Optional consistency checks
        if self.strict and attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"{fpath} attention_mask shape {tuple(attention_mask.shape)} "
                f"!= input_ids shape {tuple(input_ids.shape)}"
            )

        return {
            "name": name,
            "split": self.split,
            "path": str(fpath),
            "input_ids": input_ids,                 # [L]
            "attention_mask": attention_mask,       # [L] or None
        }


def pad_tokenized_batch(
    batch: List[Dict[str, Any]],
    pad_token_id: int,
    attention_mask_dtype: torch.dtype = torch.long,
) -> Dict[str, Any]:
    """
    Collate function for GSM8KTokenizedPtDataset.

    Pads variable-length [L] -> [B, Lmax] and produces attention_mask if missing.

    Returns a dict suitable for:
      model.diffusion_generate(inputs=batch["input_ids"], attention_mask=batch["attention_mask"], ...)

    Output:
      - name: List[str]
      - path: List[str]
      - input_ids: LongTensor [B, Lmax]
      - attention_mask: Tensor [B, Lmax] (dtype = attention_mask_dtype)
    """
    names = [b["name"] for b in batch]
    paths = [b["path"] for b in batch]

    seqs: List[torch.Tensor] = [b["input_ids"] for b in batch]
    masks: List[Optional[torch.Tensor]] = [b.get("attention_mask", None) for b in batch]

    lengths = [int(s.numel()) for s in seqs]
    if any(s.dim() != 1 for s in seqs):
        raise ValueError("Expected each 'input_ids' to be 1D [L].")

    Lmax = max(lengths) if lengths else 0
    B = len(batch)

    input_ids = torch.full((B, Lmax), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, Lmax), dtype=attention_mask_dtype)

    for i, (ids, m, L) in enumerate(zip(seqs, masks, lengths)):
        input_ids[i, :L] = ids
        if m is None:
            # default: 1 where tokens exist
            attention_mask[i, :L] = 1
        else:
            if m.dim() != 1:
                raise ValueError("Expected each 'attention_mask' to be 1D [L].")
            if m.numel() != L:
                raise ValueError(f"attention_mask length {m.numel()} != input_ids length {L}")
            # store as requested dtype
            attention_mask[i, :L] = m.to(attention_mask_dtype)

    return {
        "name": names,
        "path": paths,
        "input_ids": input_ids,               # [B, Lmax]
        "attention_mask": attention_mask,     # [B, Lmax]
    }
