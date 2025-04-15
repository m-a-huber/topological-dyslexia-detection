import pickle
from pathlib import Path
from typing import Callable

import gudhi.representations as gdrep  # type: ignore
import numpy as np
import numpy.typing as npt
from sklearn.base import clone  # type: ignore
from tqdm import tqdm  # type: ignore

from scripts.time_series_homology import TimeSeriesHomology


def compute_persistences(
    time_series_dir: Path,
    out_dir: Path,
    homology_coeff_field: int = 11,
    min_persistence: float = 0.0,
    overwrite: bool = False
) -> None:
    tsh = TimeSeriesHomology(
        homology_coeff_field=homology_coeff_field,
        min_persistence=min_persistence,
    )
    for time_series_path in tqdm(
        sorted(time_series_dir.glob("*.npy")),
        desc="Computing persistences"
    ):
        id = time_series_path.stem.split("_")[-1]
        out_file = out_dir / f"persistences_{id}.pkl"
        if not out_file.is_file() or overwrite:
            out_file.parent.mkdir(exist_ok=True, parents=True)
            time_series = np.load(time_series_path)
            dgm_ordinary_x, dgm_relative_x, dgm_essential_x = clone(
                tsh
            ).fit_transform(time_series[:, 0])
            dgm_ordinary_y, dgm_relative_y, dgm_essential_y = clone(
                tsh
            ).fit_transform(time_series[:, 1])
            with open(out_file, "wb") as f_out:
                pickle.dump([
                    dgm_ordinary_x,
                    dgm_relative_x,
                    dgm_essential_x,
                    dgm_ordinary_y,
                    dgm_relative_y,
                    dgm_essential_y,
                ], f_out)
    return


def make_persistence_images(
    persistences_dir: Path,
    out_dir: Path,
    bandwidth: float = 1.0,
    weight: Callable[[npt.NDArray], float] = lambda pt: 1,
    resolution: list[int] = [20, 20],
    overwrite: bool = False
) -> None:
    persistence_imager = gdrep.PersistenceImage(
        bandwidth=bandwidth,
        resolution=resolution,
        weight=weight,
    )
    for persistences_path in tqdm(
        sorted(persistences_dir.glob("*.pkl")),
        desc="Creating persistence images"
    ):
        id = persistences_path.stem.split("_")[-1]
        out_file = out_dir / f"persistence_images_{id}.npy"
        if not out_file.is_file() or overwrite:
            out_file.parent.mkdir(exist_ok=True, parents=True)
            with open(persistences_path, "rb") as f_in:
                persistences = pickle.load(f_in)
            assert len(persistences) == 6
            persistences_to_transform = [
                np.sort(dim, axis=1)  # ensure all dgms are above diagonal
                for persistence in persistences
                for dim in persistence
                if len(dim) > 0  # ignore empty persistence dgms
            ]
            assert len(persistences_to_transform) == 6
            persistence_images = persistence_imager.fit_transform(
                persistences_to_transform
            )
            assert persistence_images.shape == (6, np.prod(resolution))
            np.save(out_file, persistence_images)
    return
