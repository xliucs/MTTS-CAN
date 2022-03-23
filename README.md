## MTTS-CAN: Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)


## Paper

#### [Xin Liu](https://homes.cs.washington.edu/~xliu0/), [Josh Fromm](https://www.linkedin.com/in/josh-fromm-2a4a2258/), [Shwetak Patel](https://ubicomplab.cs.washington.edu/members/), [Daniel McDuff](https://www.microsoft.com/en-us/research/people/damcduff/), “Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement”, NeurIPS 2020, Oral Presentation (105 out of 9454 submissions) 

#### Link: <https://papers.nips.cc/paper/2020/file/e1228be46de6a0234ac22ded31417bc7-Paper.pdf>

## New Results (Trained only on PURE, and Tested on UBFC)

|  (BPM) | MAE | MAPE | RMSE |Pearson Coef. |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| TS-CAN  | 1.47  | 1.56%  | 2.31  | 0.99  |


## New Pre-Trained Model (Updated Nov 2021)

Working in pregress. Check back later! 

#### Abstract

Telehealth and remote health monitoring have become increasingly important during the SARS-CoV-2 pandemic and it is widely expected that this will have a lasting impact on healthcare practices. These tools can help reduce the risk of exposing patients and medical staff to infection, make healthcare services more accessible, and allow providers to see more patients. However, objective measurement of vital signs is challenging without direct contact with a patient. We present a video-based and on-device optical cardiopulmonary vital sign measurement approach. It leverages a novel multi-task temporal shift convolutional attention network (MTTS-CAN) and enables real-time cardiovascular and respiratory measurements on mobile platforms. We evaluate our system on an ARM CPU and achieve state-of-the-art accuracy while running at over 150 frames per second which enables real-time applications. Systematic experimentation on large benchmark datasets reveals that our approach leads to substantial (20\%-50\%) reductions in error and generalizes well across datasets.



## Waveform Samples

### Pulse

![pulse_waveform](./pulse_waveform.png)


### Respiration 

![resp_waveform](./resp_waveform.png)


## Citation 

``` bash
@article{liu2020multi,
  title={Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement},
  author={Liu, Xin and Fromm, Josh and Patel, Shwetak and McDuff, Daniel},
  journal={arXiv preprint arXiv:2006.03790},
  year={2020}
}
```

## Demo

**Try out our live demo via link [here](https://vitals.cs.washington.edu/).**

Our demo code: https://github.com/ubicomplab/rppg-web


## TVM

If you want to use TVM, pleaea follow [this tutorial](https://tvm.apache.org/docs/) to set it up. Then, you will need to replace the code in `incubator-tvm/python/tvm/relay/frontend/keras.py` with our `code/tvm-ops-mtts-can.py`. We implemented required tensor operations for attention, tensor shift module used in our models. 

## Training 

`python code/train.py --exp_name test --exp_name [e.g., test] --data_dir [DATASET_PATH] --temporal [e.g., MMTS_CAN]`

## Inference 

`python code/predict_vitals.py --video_path [VIDEO_PATH]`

The default video sampling rate is 30Hz. 

#### Note

During the inference, the program will generate a sample pre-processed frame. Please ensure it is in portrait orientation. If not, you can comment out line 30 (rotation) in the `inference_preprocess.py`. 


## Requirements


Tensorflow 2.2-2.4


`conda create -n tf-gpu tensorflow-gpu cudatoolkit=10.1` -- this command takes care of both CUDA and TF environments. 

`pip install opencv-python scipy numpy matplotlib`

If`pip install opencv-python` does not work, I found these commands always work on my mac. 

```
conda install -c menpo opencv -y
pip install opencv-python
```




## Contact

Please post your technical questions regarding this repo via Github Issues. 







