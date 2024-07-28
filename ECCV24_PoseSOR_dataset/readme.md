# PoseSOR - Dataset
> PoseSOR : Human Pose Can Guide Our Attention

Please download our full dataset via [link](https://drive.google.com/drive/folders/1m1Xtr2N6CLMH2J3HviOQioHELQY1hjY8?usp=sharing). 

We organize our dataset in **COCO format**. The structure of the dataset is as follows:

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
Note that our code repo includes the annotations only. Images can be downloaded from the above full dataset link or from [MSCOCO 2017 split](https://cocodataset.org/#home). We also include a link to our annotations~[[dataset-without-images]](https://drive.google.com/drive/folders/1L0dagM2-UtuZX4lb6dSe1O5mdcDyd64d?usp=sharing), which is much smaller in size.


## Annotation Files
All annotations are stored in json format following MSCOCO style. 

`images` field contains the basic information about each image:
```json
"images": [
    {
        "license": 3,
        "file_name": "COCO_val2014_000000000192.jpg",
        "coco_url": "http://images.cocodataset.org/train2017/000000000192.jpg",
        "height": 480,
        "width": 640,
        "date_captured": "2013-11-22 22:14:30",
        "flickr_url": "http://farm7.staticflickr.com/6016/5960058194_1dfae5d508_z.jpg",
        "id": 192
    },
]
```

`annotations` field contains the keypoints, instance mask, saliency ranking, class_id, etc.:
```json
"annotations": [
    {
        "image_id": 192,
        "id": 545132,
        "category_id": 1,
        "class_id": 1,
        "segmentation": [[365, 392, 350, 391, 348, 406, 345, 415, 331, 414, 324, 400, 324, 386, 310, 332, 297, 309, 290, 278, 280, 266, 268, 252, 270, 236, 292, 209, 304, 201, 311, 194, 316, 184, 324, 179, 338, 182, 345, 187, 347, 197, 360, 202, 370, 209, 377, 221, 379, 231, 379, 243, 377, 256, 377, 268, 376, 276, 372, 279, 360, 278, 358, 292, 358, 307, 358, 322, 357, 332, 357, 343, 355, 352, 352, 363, 358, 370, 368, 373, 374, 377]],
        "bbox": [268.11, 179.46, 111.35, 235.68],
        "keypoints": [327, 216, 2, 331, 211, 2, 322, 211, 2, 337, 206, 2, 314, 206, 2, 352, 215, 2, 292, 226, 2, 365, 236, 2, 278, 251, 2, 367, 260, 2, 296, 274, 2, 340, 283, 2, 312, 288, 2, 344, 335, 2, 330, 339, 2, 347, 376, 2, 338, 402, 2],
        "num_keypoints": 17,
        "visiting_order": 1,
        "iscrowd": 0,
        "area": 10000
    }
]
```

