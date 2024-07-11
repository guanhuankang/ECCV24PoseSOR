import os, copy
import itertools
from datetime import timedelta

from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch
)

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils import comm, logger
from detectron2.evaluation import (
    verify_results
)
from detectron2.data import build_detection_train_loader, build_detection_test_loader

from configs import add_custom_config
from dataset import register_sor_dataset, sor_dataset_mapper_train, sor_dataset_mapper_test
from evaluation import SOREvaluator
from network import *

import torch
import numpy as np
import random

def set_seed(seed):
    """
    Set random seed for detectron2 and other modules.
    Args:
        seed (int): The seed value to use.
    """
    # Set seed for torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set seed for numpy
    np.random.seed(seed)

    # Set seed for random
    random.seed(seed)

    # Set seed for detectron2
    from detectron2.utils.env import seed_all_rng
    seed_all_rng(seed)

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return SOREvaluator(cfg, dataset_name)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params = []
        memo = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                        "relative_position_bias_table" in module_param_name
                        or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def __init__(self, params, defaults):
                    super().__init__(params, defaults)

                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

                def zero_grad(self, set_to_none: bool = True):
                    super().zero_grad(set_to_none)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = torch.optim.SGD(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_full_model_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=lambda x: sor_dataset_mapper_train(x, cfg=cfg))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=lambda x: sor_dataset_mapper_test(x, cfg=cfg))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_custom_config(cfg, num_gpus=args.num_gpus)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.IMS_PER_BATCH = cfg.SOLVER.IMS_PER_GPU * cfg.SOLVER.NUM_GPUS
    cfg.freeze()

    default_setup(cfg, args)
    logger.setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="toy")
    return cfg


def main(args):
    set_seed(2024)

    cfg = setup(args)
    ## register sor dataset before starts training
    register_sor_dataset(cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model=model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

def readCfgFromArgs(args, key, default=None):
    args_dict = dict((args.opts[i], args.opts[i + 1]) for i in range(len(args.opts)) if (int(i) & 1) == 0)
    return args_dict.get(key, default)

def hardSetArgs(args, key, value):
    args_dict = dict((args.opts[i], args.opts[i + 1]) for i in range(len(args.opts)) if (int(i) & 1) == 0)
    args_dict[key] = value  ## hardSet
    opts = []
    for k, v in args_dict.items():
        opts += [k, v]
    args.opts = opts
    return args

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.num_gpus = int(readCfgFromArgs(args, "SOLVER.NUM_GPUS", torch.cuda.device_count()))
    # timeout = int(readCfgFromArgs(args, "SOLVER.TIMEOUT", 59)); hardSetArgs(args, "SOLVER.TIMEOUT", timeout)

    # print("Available GPUs:", args.num_gpus, "Timeout:", timeout)
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
        # timeout=timedelta(minutes=timeout)
    )