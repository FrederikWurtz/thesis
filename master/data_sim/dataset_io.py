"""Dataset IO helpers: listing .npz files, reading metadata and splitting.
"""

import os
from pathlib import Path
import numpy as np
from typing import List, Tuple


def list_npz_files(directory: str) -> List[str]:
    p = Path(directory)
    if not p.exists():
        return []
    return sorted([str(x) for x in p.glob('*.npz')])

def list_pt_files(directory: str) -> List[str]:
    p = Path(directory)
    if not p.exists():
        return []
    return sorted([str(x) for x in p.glob('*.pt')])
    
def read_npz_metadata(path: str) -> dict:
    data = np.load(path)
    md = {
        'dem_shape': tuple(data['dem'].shape) if 'dem' in data else None,
        'n_images': int(data['data'].shape[0]) if 'data' in data else None,
    }
    data.close()
    return md


def dataset_statistics(directory: str) -> dict:
    files = list_npz_files(directory)
    stats = {'n_files': len(files), 'dem_shapes': {}, 'n_images': {}}
    for f in files:
        md = read_npz_metadata(f)
        stats['dem_shapes'].setdefault(str(md['dem_shape']), 0)
        stats['dem_shapes'][str(md['dem_shape'])] += 1
        stats['n_images'].setdefault(str(md['n_images']), 0)
        stats['n_images'][str(md['n_images'])] += 1
    return stats


__all__ = ['list_npz_files', 'read_npz_metadata', 'dataset_statistics']
