## Analysis and optimization of photoplethysmography imaging methods for non-contact measurement of heart variability parameters

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

Deep learning neuronal networks based on remote photoplethysmography. Extracting the pulse signal from video using machine learning with a view to heart rate variability parameters.
Source code of the master thesis titles: "Analysis and optimization of photoplethysmography imaging methods for non-contact measurement of heart variability parameters"

## Cite as

Sarah Quehl. (2022, April 11). Analysis and optimization of photoplethysmography imaging methods for non-contact measurement of heart variability parameters


## Basis Paper
The code is based on the following paper:
#### [Xin Liu](https://homes.cs.washington.edu/~xliu0/), [Josh Fromm](https://www.linkedin.com/in/josh-fromm-2a4a2258/), [Shwetak Patel](https://ubicomplab.cs.washington.edu/members/), [Daniel McDuff](https://www.microsoft.com/en-us/research/people/damcduff/), “Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement”, NeurIPS 2020, Oral Presentation (105 out of 9454 submissions)

## Training

`python code/train.py --exp_name test --exp_name [e.g., test] --data_dir [DATASET_PATH] --temporal [e.g., MMTS_CAN]`

examples:
python code/train.py --exp_name testCohFace2 --data_dir E:/Databases --temporal CAN_3D --nb_task 4

python code/train.py --exp_name test1 --data_dir /mnt/share/StudiShare/sarah/Databases/ --temporal TS_CAN --database_name MIX

python code/train.py --exp_name test1 --data_dir /mnt/share/StudiShare/sarah/Databases/ --temporal TS_CAN


## Inference

`python code/predict_vitals.py --video_path [VIDEO_PATH]`

The default video sampling rate is 30Hz.

#### Note

During the inference, the program will generate a sample pre-processed frame. Please ensure it is in portrait orientation. If not, you can comment out line 30 (rotation) in the `inference_preprocess.py`.


## Requirements


Tensorflow 2.0+
tested with Tensorflow-gpu=2.3

`conda create -n tf-gpu tensorflow-gpu cudatoolkit=10.1` -- this command takes care of both CUDA and TF environments.

`pip install opencv-python scipy numpy matplotlib heartpy scikit-learn`

If`pip install opencv-python` does not work, I found these commands always work on my mac.

```
conda install -c menpo opencv -y
pip install opencv-python
```

## Contact

Please post your technical questions regarding this repo via Github Issues.
