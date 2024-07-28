# PoseSOR: Human Pose Can Guide Our Attention

We release our PoseSOR dataset, which includes all annotations (instance masks, saliency ranks and human pose annotations), for both SOR benchmarks (ASSR and IRSR). The images of both benchmarks can be downloaded from MS-COCO-2017 or their original codebase (ASSR and IRSR repo).



We organize our dataset in COCO format. The structure of the dataset is as follows:

```shell
DATASET_ROOT
├── assr
│       ├── assr_test.json
│       ├── assr_train.json
│       ├── assr_val.json
│       └── images
│           ├── test
│           │   └── COCO_val2014_000000000192.jpg
│           ├── train
│           │   └── COCO_train2014_000000000009.jpg
│           └── val
│               └── COCO_val2014_000000000164.jpg
├── irsr
│       ├── images
│       │       ├── test
│       │       │   └── COCO_val2014_000000000192.jpg
│       │       └── train
│       │           └── COCO_train2014_000000000110.jpg
│       ├── irsr_test.json
│       └── irsr_train.json
└── readme.md

```

