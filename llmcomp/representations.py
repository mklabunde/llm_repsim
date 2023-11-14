import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import datasets
import torch


def extract_info(arrow_directory: Union[str, Path]) -> Dict[str, Any]:
    with open(Path(arrow_directory).joinpath("dataset_info.json"), "r") as f:
        info = json.load(f)

    regex = (
        r"Logged outputs \(hidden representations\) of the language model "
        r"(?P<model>.*) on the (?P<dataset>.*) dataset.\n    "
        r"Model information: (?P<model_info>.*)"
    )
    match = re.search(regex, info["description"], re.MULTILINE)
    if match:
        info = match.groupdict()
        if "validation" in str(arrow_directory):
            split = "validation"
            info["split"] = split
    info["model"] = convert_to_path_compatible(info["model"]).replace("_", "-")
    info["dataset"] = convert_to_path_compatible(info["dataset"]).replace("_", "-")

    return info


def convert_to_path_compatible(s: str) -> str:
    return s.replace("\\", "-").replace("/", "-")


def load_arrow_files(directory: Union[str, Path]) -> List[torch.Tensor]:
    directory = Path(directory)
    tensors = []
    for path in list(sorted(directory.glob("data*.arrow"))):
        tensors.extend(load_arrow_file(path))
    return tensors


def load_arrow_file(path: Union[str, Path]) -> List[torch.Tensor]:
    path = str(path)
    reps = datasets.Dataset.from_file(path)
    all_reps = []
    for output_emb, prompt_tokens in zip(
        reps["output_embeddings"], reps["prompt_tokens"]
    ):
        # We do not use the full output_embedding as it contains representations of
        # generated tokens.
        rep = torch.tensor(output_emb[: len(prompt_tokens)])
        assert len(rep) == len(prompt_tokens)
        all_reps.append(rep)
    return all_reps


def final_token_representation(reps: List[torch.Tensor]) -> torch.Tensor:
    # A tandard rep from huggingface are of the shape (1, n_token, dim)
    # A rep from LLM-Eval has (n_token, dim) -> unsqueeze to retain the token dimension
    return torch.cat(
        [rep[:, -1, :] if (rep.ndim == 3) else rep[-1, :].unsqueeze(0) for rep in reps],
        dim=0,
    )
