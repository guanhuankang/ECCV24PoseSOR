# conda create -n hpsor python=3.9 -y
# conda activate hpsor

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

pip install opencv-python==4.7.0.72
pip install albumentations==1.3.0
pip install pandas==2.0.1
pip install scipy==1.9.1
pip install timm==0.9.0

pip uninstall Pillow -y
pip install Pillow==9.3.0
pip uninstall setuptools -y
pip install setuptools==59.5.0
