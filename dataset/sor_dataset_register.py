import json
import os

from detectron2.data import DatasetCatalog


def loadJson(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data


def prepare_sor_dataset_list_of_dict(dataset_id, split, root="datasets"):
    path_to_ds = os.path.join(root, dataset_id, "{}_{}.json".format(dataset_id, split))
    print("Path to {}: {}".format(dataset_id, path_to_ds), flush=True)

    dataset = loadJson(path_to_ds)
    image_path = os.path.join(root, dataset_id, "images", split)

    image_list = dataset["images"]
    iid_to_index = dict((x["id"], i) for i, x in enumerate(image_list))

    for img in image_list:
        img["file_name"] = os.path.join(image_path, img["file_name"])
        img["annotations"] = []

    for anno in dataset["annotations"]:
        index = iid_to_index[anno["image_id"]]
        image_list[index]["annotations"].append(anno)

    print("#Length of SOR dataset [{}]:{}".format(dataset_id, len(image_list)), flush=True)
    return image_list

def register_sor_dataset(cfg):
    DatasetCatalog.register("assr_train", lambda s="train": prepare_sor_dataset_list_of_dict(dataset_id="assr", split=s,
                                                                                             root=cfg.DATASETS.ROOT))
    DatasetCatalog.register("assr_val", lambda s="val": prepare_sor_dataset_list_of_dict(dataset_id="assr", split=s,
                                                                                         root=cfg.DATASETS.ROOT))
    DatasetCatalog.register("assr_test", lambda s="test": prepare_sor_dataset_list_of_dict(dataset_id="assr", split=s,
                                                                                           root=cfg.DATASETS.ROOT))

    DatasetCatalog.register("irsr_train", lambda s="train": prepare_sor_dataset_list_of_dict(dataset_id="irsr", split=s,
                                                                                             root=cfg.DATASETS.ROOT))
    DatasetCatalog.register("irsr_test", lambda s="test": prepare_sor_dataset_list_of_dict(dataset_id="irsr", split=s,
                                                                                           root=cfg.DATASETS.ROOT))
