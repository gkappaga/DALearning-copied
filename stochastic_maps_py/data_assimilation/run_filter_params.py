from __future__ import annotations

from itertools import product
from typing import Any, Callable, Dict, List

from stochastic_maps_py.data_assimilation.post_process import post_process
from stochastic_maps_py.data_assimilation.seq_assimilation import seq_assimilation
from stochastic_maps_py.models import DataAssimilationModel


FilterFactory = Callable[[DataAssimilationModel, Dict[str, Any]], Any]


def run_filter_params(
    model: DataAssimilationModel,
    factory: FilterFactory,
    tune_params: Dict[str, List[Any]],
    fixed_params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    keys = list(tune_params.keys())
    values = [tune_params[key] for key in keys]

    results = []
    for combination in product(*values):
        options = {key: value for key, value in zip(keys, combination)}
        options.update(fixed_params)
        algorithm = factory(model, options)
        seq_result = seq_assimilation(model, algorithm.sample_posterior)
        summary = post_process(model, seq_result, options)
        results.append(summary)
    return results


__all__ = ["run_filter_params"]
