rm -rf output/*.png
rm -rf output/*.jpg

WEIGHTS=../assets/model_cards/posesor_irsr_768x768/checkpoint.pth
CUDA_LAUNCH_BLOCKING=1 python demo.py --config-file ../configs/swinL.yaml --input examples/*.jpg --output output --opts MODEL.WEIGHTS $WEIGHTS INPUT.TEST_IMAGE_SIZE 768 MODEL.META_ARCHITECTURE PoseSOR