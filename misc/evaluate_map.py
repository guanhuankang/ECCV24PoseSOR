import os, sys
sys.path.append("..")

from PIL import Image
import numpy as np
import tqdm, pandas

from detectron2.engine import (
    default_argument_parser,
    default_setup,
    launch
)

from detectron2.config import get_cfg
from detectron2.utils import comm, logger
from detectron2.data import DatasetCatalog
from configs import add_custom_config
from dataset import register_sor_dataset, sor_dataset_mapper_test
from evaluation import SOREvaluator

def setup(args):
    cfg = get_cfg()
    add_custom_config(cfg, num_gpus=args.num_gpus)
    cfg.merge_from_file(args.config_file)

    root = {
        "WORK": cfg.DATASETS.ENV.WORK,
        "GROUP4090": cfg.DATASETS.ENV.GROUP4090,
        "BURGUNDY": cfg.DATASETS.ENV.BURGUNDY,
        "HTGC": cfg.DATASETS.ENV.HTGC,
        "GROUP3090": cfg.DATASETS.ENV.GROUP3090
    }.get(dict(os.environ).get("ENVNAME").upper(), cfg.DATASETS.ROOT)
    cfg.DATASETS.ROOT = root
    print("ROOT:", root)

    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # logger.setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="toy")
    register_sor_dataset(cfg)
    return cfg

def main(args):
    cfg = setup(args)
    dataset_name = cfg.DATASETS.TEST[0]
    groundtruth = DatasetCatalog.get(dataset_name)

    sor_evaluator = SOREvaluator(cfg, dataset_name)
    name_paths = {dataset_name: cfg.MODEL.WEIGHTS}
    for name, path in name_paths.items():
        for gt_zip in tqdm.tqdm(groundtruth):
            gt = sor_dataset_mapper_test(gt_zip, cfg)

            pred = np.array(Image.open(os.path.join(path, gt["image_name"]+".png")).convert("L"))
            unis = np.unique(pred)[::-1][0:-1]
            instances = [(pred == u) * 1.0 for u in unis]

            sor_evaluator.process([gt], [{"masks": instances}])

        results = sor_evaluator.evaluate()
        print(results)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(cfg.OUTPUT_DIR, f"dataset_{dataset_name}_method_{name}.csv"), "w") as f:
            f.write(str(results))

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,)
    )