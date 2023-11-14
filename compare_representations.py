import abc
import itertools
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from tqdm import tqdm

import llmcomp.measures
import llmcomp.measures.utils
import llmcomp.representations

log = logging.getLogger(__name__)


class Strategy(abc.ABC):
    strat_id = ""

    def __init__(self, baseline_score: bool = True, baseline_perms: int = 10) -> None:
        self.baseline_score = baseline_score
        self.baseline_perms = baseline_perms
        super().__init__()

    @abc.abstractmethod
    def __call__(
        self,
        rep1: List[Tensor],
        rep2: List[Tensor],
        sim_func: Callable[[Tensor, Tensor], float],
        **kwds: Any,
    ) -> Any:
        pass


class FinalTokenStrategy(Strategy):
    strat_id = "final_token"

    def __init__(self, baseline_score: bool = False, baseline_perms: int = 10) -> None:
        super().__init__(baseline_score, baseline_perms)

    def __call__(
        self,
        rep1: List[Tensor],
        rep2: List[Tensor],
        sim_func: Callable[[Tensor, Tensor], float],
        **kwds: Any,
    ) -> Dict[str, List[Any]]:
        rep1_adapted = llmcomp.representations.final_token_representation(rep1)
        rep2_adapted = llmcomp.representations.final_token_representation(rep2)
        score = sim_func(rep1_adapted, rep2_adapted)

        score = score if isinstance(score, float) else score["score"]

        randomized_score = None
        res = {"baseline_scores": None}
        if self.baseline_score:
            res = llmcomp.measures.utils.sim_random_baseline(
                rep1_adapted, rep2_adapted, sim_func, n_permutations=self.baseline_perms
            )
            randomized_score = np.mean(res["baseline_scores"].mean())

        return {
            "score": [score],
            "baseline_score": [randomized_score],
            "baseline_scores_full": [res["baseline_scores"]],
        }


def pair_should_be_compared(
    info1: Dict[str, str],
    info2: Dict[str, str],
    pair_results_path: Path,
    recompute: bool,
) -> bool:
    if (
        info1["dataset"] != info2["dataset"]
        or info1["split"] != info2["split"]
        or info1["model"] == info2["model"]
    ):
        return False

    if pair_results_path.exists() and not recompute:
        log.info(f"Skipping comparison due to existing results at {pair_results_path}")
        return False

    return True


def compare_pair(
    dir1: Path,
    dir2: Path,
    strategy: Strategy,
    measures: List[Callable],
    modelname1: str,
    modelname2: str,
    datasetname: str,
    splitname: str,
    pair_results_path: Optional[Path] = None,
):
    rep1 = llmcomp.representations.load_arrow_files(dir1)
    rep2 = llmcomp.representations.load_arrow_files(dir2)
    results_pair = defaultdict(list)

    log.info(f"Comparing with strategy {strategy.strat_id}")
    for sim_func in measures:
        start = time.perf_counter()
        results = strategy(rep1, rep2, sim_func)
        extend_dict_of_lists(results_pair, results)
        # TODO: can we specify these keys in a more modifiable way?
        n_times_to_add = len(results_pair["score"]) - len(results_pair["model1"])
        results_pair["model1"].extend([modelname1] * n_times_to_add)
        results_pair["model2"].extend([modelname2] * n_times_to_add)
        results_pair["dataset"].extend([datasetname] * n_times_to_add)
        results_pair["split"].extend([splitname] * n_times_to_add)

        if hasattr(sim_func, "__name__"):
            measure_name = sim_func.__name__
        elif hasattr(sim_func, "func"):
            measure_name = sim_func.func.__name__
        else:
            measure_name = str(sim_func)

        results_pair["measure"].extend([measure_name] * n_times_to_add)
        results_pair["strategy"].extend([strategy.strat_id] * n_times_to_add)

        log.info(
            f"{measure_name} completed in {time.perf_counter() - start:.1f} seconds"  # noqa: E501
        )
    pd.DataFrame.from_dict(results_pair).to_parquet(pair_results_path)
    return results_pair


def extend_dict_of_lists(
    to_extend: Dict[Any, List[Any]], to_add: Dict[Any, List[Any]]
) -> None:
    for key, value in to_add.items():
        if key in to_extend:
            to_extend[key].extend(value)
        else:
            to_extend[key] = value


def filter_combinations(
    combinations: List[Tuple[Path, Path]],
    must_contain_all: List[str],
    must_contain_any: List[str],
    must_not_contain: List[str],
    one_must_contain: List[str],
) -> List[Tuple[Path, Path]]:
    if must_contain_all:
        combinations = [
            c
            for c in combinations
            if all(s in str(c[0]) and s in str(c[1]) for s in must_contain_all)
        ]

    if must_contain_any:
        combinations = [
            c
            for c in combinations
            if any(s in str(c[0]) for s in must_contain_any)
            and any(s in str(c[1]) for s in must_contain_any)
        ]

    if must_not_contain:
        combinations = [
            c
            for c in combinations
            if not any(s in str(c[0]) for s in must_not_contain)
            and not any(s in str(c[1]) for s in must_not_contain)
        ]

    if one_must_contain:
        combinations = [
            c
            for c in combinations
            if all(s in str(c[0]) or s in str(c[1]) for s in one_must_contain)
        ]

    return combinations


@hydra.main(config_path="config", config_name="compare", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print("Working directory : {}".format(os.getcwd()))

    strategy = hydra.utils.instantiate(cfg.strategy)
    MEASURES = hydra.utils.instantiate(cfg.measures)

    results = {}

    basedir = Path(cfg.storage.root_dir, cfg.storage.reps_subdir)
    representation_dirs = list(sorted(basedir.glob("*")))
    results_path = Path(cfg.storage.root_dir, cfg.storage.results_subdir)
    results_path.mkdir(exist_ok=True)

    combinations = list(itertools.combinations(representation_dirs, 2))
    combinations = filter_combinations(
        combinations,
        cfg.filter.must_contain_all,
        cfg.filter.must_contain_any,
        cfg.filter.must_not_contain,
        cfg.filter.one_must_contain,
    )

    progress_bar = tqdm(total=len(combinations), desc="Model pairs")
    for dir1, dir2 in combinations:
        info1 = llmcomp.representations.extract_info(dir1)
        info2 = llmcomp.representations.extract_info(dir2)
        partial_results_path = Path(
            results_path,
            f"similarity_{info1['model']}_{info2['model']}_{info1['dataset']}_{info1['split']}.parquet",  # noqa: E501
        )

        if not pair_should_be_compared(
            info1, info2, partial_results_path, cfg.recompute
        ):
            progress_bar.update(1)
            continue
        log.info(
            f"Comparing {info1['model']} and {info2['model']} on {info1['dataset']}"
        )

        results_pair = compare_pair(
            dir1,
            dir2,
            strategy,
            MEASURES,
            info1["model"],
            info2["model"],
            info1["dataset"],
            info1["split"],
            pair_results_path=partial_results_path,
        )
        extend_dict_of_lists(results, results_pair)

        progress_bar.update(1)

    results_path.mkdir(exist_ok=True)
    pd.DataFrame.from_dict(results).to_parquet(Path(results_path, "similarity.parquet"))


if __name__ == "__main__":
    main()  # type:ignore
