python train_net.py --config-file configs/swinL.yaml --eval-only INPUT.TEST_IMAGE_SIZE 768 MODEL.WEIGHTS assets/model_cards/posesor_assr_768x768/checkpoint.pth OUTPUT_DIR assets/model_cards/posesor_assr_768x768/inference

## train
# python train_net.py --config-file configs/swinL.yaml INPUT.TRAIN_IMAGE_SIZE 768 INPUT.TEST_IMAGE_SIZE 768 MODEL.WEIGHTS assets/pretrained/swin_large_patch4_window12_384_22k.pth OUTPUT_DIR assets/model_cards/posesor_assr_768x768/training SOLVER.IMS_PER_GPU 8 SOLVER.NUM_GPUS 4 