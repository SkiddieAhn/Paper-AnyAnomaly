# AnyAnomaly: Zero-Shot Customizable Video Anomaly Detection with LVLM 
[![arXiv](https://img.shields.io/badge/Arxiv-2311.08046-b31b1b.svg?logo=arXiv)](https://google.com)
[![Colab1](https://img.shields.io/badge/⚡-Colab%20Totorial%201-yellow.svg)](https://colab.research.google.com/drive/1vDU6j2c9YwVEhuvBbUHx5GjorwKKI6sX?usp=sharing)
[![Colab2](https://img.shields.io/badge/⚡-Colab%20Tutorial%202-green.svg)](https://colab.research.google.com/drive/1xnXjvlUlB8DgbTVGRrwvuLRz2rFAbdQ5?usp=sharing)


This repository is the ```official open-source``` of [AnyAnomaly: Zero-Shot Customizable Video Anomaly Detection with LVLM](https://google.com) by Sunghyun Ahn*, Youngwan Jo*, Kijung Lee, Sein Kwon, Inpyo Hong and Sanghyun Park. ```(*equally contributed)```

## Description
Video Anomaly Detection (VAD) is a critical task in video analysis and surveillance within computer vision. However, existing VAD models rely on learned normal patterns, making them difficult to apply across diverse environments. As a result, users must retrain models or develop separate AI models for new environments, which requires expertise in machine learning, high-performance hardware, and extensive data collection, limiting the practical usability of VAD. **To address these challenges, this study proposes Customizable Video Anomaly Detection (C-VAD) and the AnyAnomaly model. C-VAD considers user-defined text as an abnormal event and detects frames containing the specified event in a video.** We implement C-VAD effectively using a Context-aware VQA approach without fine-tuning a Large Vision Language Model (LVLM). To validate the effectiveness of the proposed model, we constructed a C-VAD dataset and demonstrated the superiority of AnyAnomaly. Furthermore, despite adopting a zero-shot approach, our method achieves competitive performance on VAD benchmarks.<br/><br/>
<img width="850" alt="fig-1" src="https://github.com/user-attachments/assets/938d46a3-56cc-4cd4-8900-84bdfbd64b98">  

## Context-aware VQA
Comparison of the proposed model with the baseline. Both models perform C-VAD, but the baseline operates with frame-level VQA, whereas the proposed model employs a segment-level Context-Aware VQA.
**Context-Aware VQA is a method that performs VQA by utilizing additional contexts that describe an image.** To enhance the object analysis and action understanding capabilities of LVLM, we propose Position Context and Temporal Context.
- **Position Context Tutorial: [[Google Colab](https://colab.research.google.com/drive/1vDU6j2c9YwVEhuvBbUHx5GjorwKKI6sX?usp=sharing)]**
- **Temporal Context Tutorial: [[Google Colab](https://colab.research.google.com/drive/1xnXjvlUlB8DgbTVGRrwvuLRz2rFAbdQ5?usp=sharing)]**<br/>
<img width="850" alt="fig-2" src="https://github.com/user-attachments/assets/f0fbef97-693e-4e58-bbc2-d02ee4b89274">  

## Results
Table 1 and Table 2 present **the evaluation results on the C-VAD datasets (C-ShT, C-Ave).** The proposed model achieved performance improvements of **9.88% and 13.65%** over the baseline on the C-ShT and C-Ave datasets, respectively. Specifically, it showed improvements of **14.34% and 8.2%** in the action class, and **3.25% and 21.98%** in the appearance class.<br/><br/>
<img width="850" alt="fig-3" src="https://github.com/user-attachments/assets/1118a4ad-cb1d-47db-be72-8602d8e8a6a2">  
<img width="850" alt="fig-3" src="https://github.com/user-attachments/assets/c16bc817-8538-4503-bf38-85007c72a39a">  

## Qualitative Evaluation 
- **Anomaly Detection in diverse scenarios**
  
|         Text              |Demo  |
|:--------------:|:-----------:|
| **Jumping-Falling<br/>-Pickup** |![c5](https://github.com/user-attachments/assets/9110c14b-0999-45ca-9bd0-9319ceed883c)|
| **Bicycle-<br/>Running** |![c6-2](https://github.com/user-attachments/assets/d62c0052-ee7b-452c-b3ce-a8fe9e5fcd5f)|
| **Bicycle-<br/>Stroller** |![c7](https://github.com/user-attachments/assets/f3cc1bbd-ea8e-4473-a84c-4b692e74ddb4)|


- **Anomaly Detection in complex scenarios**

|         Text              |Demo  |
|:--------------:|:-----------:|
| **Driving outside<br/> lane** |![c4](https://github.com/user-attachments/assets/4996b407-0ca1-4232-85d6-1f5efbad4ab5)|
| **People and car<br/> accident** |![c1](https://github.com/user-attachments/assets/2c5faa09-5141-471c-83ef-cee4c4574f5f)|
| **Jaywalking** |![c2](https://github.com/user-attachments/assets/bc9b6789-0d71-4712-97b9-1007efaed273)|
| **Walking<br/> drunk** |![c3](https://github.com/user-attachments/assets/eb450801-b700-419d-a920-2799553c2452)|

## Datasets
- We processed the Shanghai Tech Campus (ShT) and CUHK Avenue (Ave) datasets to create the labels for the C-ShT and C-Ave datasets. These labels can be found in the ```ground_truth``` folder. **To test the C-ShT and C-Ave datasets, you need to first download the ShT and Ave datasets and store them in the directory corresponding to** ```'data_root'```.
- You can specify the dataset's path by editing ```'data_root'``` in ```config.py```.
  
|     CUHK Avenue    | Shnaghai Tech.    |
|:------------------------:|:-----------:|
|[Official Site](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)|[Official Site](https://svip-lab.github.io/dataset/campus_dataset.html)


## 1. Requirements and Installation For Chat-UniVi
- ```Chat-UniVi```: [[GitHub]](https://github.com/PKU-YuanGroup/Chat-UniVi)
- weights: Chat-UniVi 7B [[Huggingface]](https://huggingface.co/Chat-UniVi/Chat-UniVi/tree/main), Chat-UniVi 13B [[Huggingface]](https://huggingface.co/Chat-UniVi/Chat-UniVi-13B/tree/main)
- Install required packages:
```bash
git clone https://github.com/PKU-YuanGroup/Chat-UniVi
cd Chat-UniVi
conda create -n chatunivi python=3.10 -y
conda activate chatunivi
pip install --upgrade pip
pip install -e .
pip install numpy==1.24.3

# Download the Model (Chat-UniVi 7B)
mkdir weights
cd weights
sudo apt-get install git-lfs
git lfs install
git lfs clone https://huggingface.co/Chat-UniVi/Chat-UniVi

# Download extra packages
cd ../../
pip install -r requirements.txt
```


## Command
- ```C-Ave type```: [too_close, bicycle, throwing, running, dancing]
- ```C-ShT type```: [car, bicycle, fighting, throwing, hand_truck, running, skateboarding, falling, jumping, loitering, motorcycle]
- ```C-Ave type (multiple)```: [throwing-too_close, running-throwing]
- ```C-ShT type (multiple)```: [stroller-running, stroller-loitering, stroller-bicycle, skateboarding-bicycle, running-skateboarding, running-jumping, running-bicycle, jumping-falling-pickup, car-bicycle]
```Shell
# Baseline model (Chat-UniVi) → C-ShT
python -u vad_chatunivi.py --dataset=shtech --type=falling
# proposed model (AnyAomaly) → C-ShT
python -u vad_proposed_chatunivi.py --dataset=shtech --type=falling
# proposed model (AnyAnomaly) → C-ShT, diverse anomaly scenarios
python -u vad_proposed_chatunivi.py --dataset=shtech --multiple=True --type=jumping-falling-pickup
```

## 2. Requirements and Installation For MiniCPM-V
- ```MiniCPM-V```: [[GitHub]](https://github.com/OpenBMB/MiniCPM-V.git)
- Install required packages:
```bash
git clone https://github.com/OpenBMB/MiniCPM-V.git
cd MiniCPM-V
conda create -n MiniCPM-V python=3.10 -y
conda activate MiniCPM-V
pip install -r requirements.txt

# Download extra packages
cd ../
pip install -r requirements.txt
```

## Command
```Shell
# Baseline model (MiniCPM-V) → C-ShT
python -u vad_MiniCPM.py --dataset=shtech --type=falling 
# proposed model (AnyAomaly) → C-ShT
python -u vad_proposed_MiniCPM.py --dataset=shtech --type=falling 
# proposed model (AnyAnomaly) → C-ShT, diverse anomaly scenarios
python -u vad_proposed_MiniCPM.py --dataset=shtech --multiple=True --type=jumping-falling-pickup
```

## Contact
Should you have any question, please create an issue on this repository or contact me at skd@yonsei.ac.kr.
