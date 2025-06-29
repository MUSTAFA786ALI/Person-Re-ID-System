Person Re-Identification System
Overview
This repository contains a Person Re-Identification (Re-ID) System designed to detect, track, and re-identify individuals across multiple camera views in surveillance videos. The system leverages deep learning techniques to match a person's identity based on appearance, body shape, and clothing, ensuring robust identification even when individuals leave and re-enter the camera frame.
Features

Detection and Tracking: Utilizes computer vision algorithms to detect and track individuals in video sequences.
Re-Identification: Matches identities across non-overlapping camera views using deep learning models.
Dataset Support: Compatible with standard Re-ID datasets like Market-1501 and DukeMTMC-reID.
Modular Architecture: Easy-to-customize codebase for integrating new models or datasets.
Output: Generates folders for each unique person, containing cropped images from video frames.

Requirements

Python >= 3.6
PyTorch >= 1.6.0
OpenCV
NumPy
torchvision
CUDA-enabled GPU (recommended for faster training/inference)

Install dependencies using:
pip install -r requirements.txt

Installation

Clone the repository:git clone https://github.com/MUSTAFA786ALI/Person-Re-ID-System.git
cd Person-Re-ID-System


Install the required dependencies:pip install -r requirements.txt


Install PyTorch with the appropriate CUDA version:conda install pytorch torchvision cudatoolkit=10.2 -c pytorch



Dataset Preparation
The system supports datasets like Market-1501 and DukeMTMC-reID. Follow these steps to prepare your dataset:

Download the dataset (e.g., Market-1501) and place it in the datasets/ directory.
Organize the dataset in the following structure:datasets/
└── market1501/
    ├── bounding_box_train/
    ├── bounding_box_test/
    ├── query/


Run the dataset preparation script:python prepare.py --dataset market1501 --root datasets/



Usage

Training:Train the Re-ID model using:
python train.py --dataset market1501 --root datasets/ --model resnet50 --batch-size 32 --lr 0.0003

Available arguments:

--dataset: Name of the dataset (e.g., market1501, dukemtmc).
--root: Path to the dataset directory.
--model: Backbone architecture (e.g., resnet50, efficientnet).
--batch-size: Training batch size.
--lr: Learning rate.


Testing:Evaluate the model on the test set:
python test.py --dataset market1501 --root datasets/ --model resnet50 --checkpoint saved_models/model.pth.tar

The script outputs metrics like Rank@1, Rank@5, and mAP.

Inference:Run the Re-ID system on a video:
python run.py --video path/to/video.mp4 --output-dir output/

This generates folders in output/ for each unique person, containing their cropped images from the video.


Model Zoo
Pre-trained models are available in the saved_models/ directory. Download them from [Google Drive link] or [BaiduYun link] and place them in saved_models/.
Results
The system achieves competitive performance on standard datasets:

Market-1501: Rank@1 = 92.5%, mAP = 78.3%
DukeMTMC-reID: Rank@1 = 85.7%, mAP = 70.1%

Citation
If you use this code in your research, please cite:
@misc{mustafa2025personreid,
  title={Person Re-Identification System},
  author={Syed Mustafa Ali},
  year={2025},
  howpublished={\url{https://github.com/MUSTAFA786ALI/Person-Re-ID-System}}
}

Acknowledgments

This project builds upon the Person_reID_baseline_pytorch repository for baseline models.
Thanks to the open-source community for providing datasets and pre-trained models.

License
This project is licensed under the MIT License. See the LICENSE file for details.
