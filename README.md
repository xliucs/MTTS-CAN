# Analysis and optimization of photoplethysmography imaging methods for non-contact measurement of heart variability parameters

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

Deep learning neuronal networks based on remote photoplethysmography. Extracting the pulse signal from video using machine learning with a view to heart rate variability parameters.
Source code of the master thesis titles: "Analysis and optimization of photoplethysmography imaging methods for non-contact measurement of heart variability parameters"

## Cite as

Sarah Quehl. (2022, April 11). Analysis and Optimization of Photoplethysmography Imaging Methods for Non-Contact Measurement of Heart Variability Parameters

## Abstract 
Heart rate variability is an important physiological parameter for health and refers to the natural
variation of the time between two heartbeats. Heart rate variability describes the adaptability of an
organism to external and internal factors and can be measured with common measuring devices, like
electrocardiogram or photoplethysmogram. Today, this is even possible with the smartphone via apps.
Photoplethysmography Imaging as a non-contact method is a further development of state-of-the-art
photoplethysmography for recording cardiac activity by detecting minimal pulse-induced fluctuations
on the skin with a RGB camera. Most Photoplethysmography Imaging methods focus on heart rate
measurement and do not consider heart rate variability. In recent years, many new approaches based
on signal filtering or neural networks have been presented. However, the accuracy required for medical
purposes, especially with regard to heart rate variability, has not yet been achieved and represents a
major challenge.
This thesis compares current Photoplethysmography Imaging methods based on neural networks. For
this purpose, four basis methods are implemented and tested for functionality. Based on these findings,
two new networks were developed, the PTS-CAN and the PPTS-CAN. These are based on multi-objective
optimization and add one and two additional outputs to the neural network, respectively. The additional
output of the PTS-CAN outputs a binary signal that has a value of one at peaks. For this output two new
loss functions were developed, which have the goal to reduce the temporal error of the peaks. For this
purpose, two new loss functions named ownGauss and the TE were developed, the last one allows an
interpretation of the error in seconds. Both manipulate the ground truth to generate a loss, to reward
the peaks that are close to the real peak and to punish peaks that are further away. A further output
was added to the first model, which outputs various variable parameters and is evaluated by the mean
absolute percentage error loss function. All used models are trained on the same database and are
compared. In addition, there is a comparison with the first developed methods on the subject of vital
parameters extraction from video. A final comparison shows an improvement in HR and HRV parameter
calculation with the new methods. The heart rate calculation can be improved by about 20%. In the field
of HRV parameters, an improvement of 5,7% can be achieved for the parameter SDNN, for example.
In a cross-validation, improvements are achieved over the baseline methods and there is also a slight
improvement over the basis models. For the parameters in the frequency domain, the improvements are
a bit less clear than in the time domain, since the frequency analysis is more challenging here.
A project was generated, which can be used as a basis for further experiments with further approaches
and loss functions. The integration of further network architectures as well as loss functions is easily
possible.

## Preprocessing
It is recommended to save the important information of each video into a hdf5-file using the `prepare_databases.py` script. Here pixel data, ground truth and various parameters are integrated.

## Training

`python code/train.py --exp_name test --exp_name [e.g., test] --data_dir [DATASET_PATH] --temporal [e.g., MMTS_CAN]`

examples:

python code/train.py --exp_name test1 --data_dir /mnt/share/StudiShare/sarah/Databases/ --temporal TS_CAN --database_name MIX2


#### Issues:

In PPTS_CAN, the frame rate used is derived from the video length used (which results from the data sets). This must still be passed in generalized form in the layers.

## Inference

`python code/predict_vitals_oneVideo.py --video_path [VIDEO_PATH] --save_dir [SAVE_PATH] --trained_model [CHECKPOINT_PATH]
        --model_name [e.g., TS_CAN, PTS_CAN, PPTS_CAN] --parameter [e.g., "bpm, sdnn, pnn50, lfhf"]`

## Path dependencies in the following scripts
final_evaluation.py

model_evaluation.py

pre_process.py

predict_vitals_comparison.py

predict_vitals_new.py

predict_vitals_oneVideo.py

predict.vitals.py

layer_output.py


In the current scripts, the data has been divided into the folders 1)Training and 2)Validation.

## evaluation_iPhys.py
Script for evaluating the prediction of the iPhys models (GreenChannel, POH, CHROM) with the same procedure and products as in the finalEvaluation.py script. 

### Requirements:
Predictions of the models, saved as a .txt file with the names: `*GC.txt`, `*ICA_POH.txt`, `*CHROM.txt` 

They are located in the same folder as the ground truth files.


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


## Basis Paper
The code is based on the following paper:
#### [Xin Liu](https://homes.cs.washington.edu/~xliu0/), [Josh Fromm](https://www.linkedin.com/in/josh-fromm-2a4a2258/), [Shwetak Patel](https://ubicomplab.cs.washington.edu/members/), [Daniel McDuff](https://www.microsoft.com/en-us/research/people/damcduff/), “Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement”, NeurIPS 2020, Oral Presentation (105 out of 9454 submissions)´

## Contact

Please post your technical questions regarding this repo via Github Issues.
