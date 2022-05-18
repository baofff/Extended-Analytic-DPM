import os
from typing import List
import ml_collections
import argparse
from core.diffusion.schedule import NamedSchedule
from core.diffusion.sde import VPSDE


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_schedule(schedule):
    if schedule.startswith('linear') or schedule.startswith('cosine'):
        typ, N = schedule.split('_')
        N = int(N)
        return NamedSchedule(typ, N)
    elif schedule.startswith('vpsde'):
        typ, N = schedule.split('_')
        N = int(N)
        return VPSDE().get_schedule(N)
    else:
        raise NotImplementedError


def parse_sde(sde):
    if sde == 'vpsde':
        return VPSDE()
    else:
        raise NotImplementedError


def sub_dict(dct: dict, *keys):
    return {key: dct[key] for key in keys if key in dct}


def dict2str(dct):
    pairs = []
    for key, val in dct.items():
        pairs.append("{}_{}".format(key, val))
    return "_".join(pairs)


def create_sample_config(get_config_fn, workspace, ckpt: str, hparams: dict, keys: List[str], description=None):
    description = description or dict2str({key: hparams[key] for key in keys if key in hparams})  # a description of the hparams
    path = os.path.join(workspace, f'evaluate/evaluator/sample2dir/{ckpt}/{description}')
    config = get_config_fn(path=path, task='sample2dir', **hparams)
    config.workspace = workspace
    config.backup_root = os.path.join(workspace, f'evaluate/evaluator/sample2dir/{ckpt}/reproducibility/{description}')
    config.interact = interact = ml_collections.ConfigDict()
    interact.fname_log = os.path.join(workspace, f'evaluate/evaluator/sample2dir/{ckpt}/{description}.log')
    return config


def create_nll_config(get_config_fn, workspace, ckpt: str, hparams: dict, keys: List[str], description=None):
    description = description or dict2str({key: hparams[key] for key in keys if key in hparams})
    fname = os.path.join(workspace, f'evaluate/evaluator/nll/{ckpt}/{description}.pth')
    config = get_config_fn(fname=fname, task='nll', **hparams)
    config.workspace = workspace
    config.backup_root = os.path.join(workspace, f'evaluate/evaluator/nll/{ckpt}/reproducibility/{description}')
    config.interact = interact = ml_collections.ConfigDict()
    interact.fname_log = os.path.join(workspace, f'evaluate/evaluator/nll/{ckpt}/{description}.log')
    return config
