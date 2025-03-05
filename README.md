# AnyAnomaly: Zero-Shot Customizable Video Anomaly Detection with LVLM 
[![arXiv](https://img.shields.io/badge/Arxiv-2311.08046-b31b1b.svg?logo=arXiv)](https://google.com)
[![Colab1](https://img.shields.io/badge/⚡-Colab%20Totorial%201-yellow.svg)](https://colab.research.google.com/drive/1vDU6j2c9YwVEhuvBbUHx5GjorwKKI6sX?usp=sharing)
[![Colab2](https://img.shields.io/badge/⚡-Colab%20Tutorial%202-green.svg)](https://colab.research.google.com/drive/1xnXjvlUlB8DgbTVGRrwvuLRz2rFAbdQ5?usp=sharing)

This repository is the ```official open-source``` of [AnyAnomaly: Zero-Shot Customizable Video Anomaly Detection with LVLM](https://google.com) by Sunghyun Ahn*, Youngwan Jo*, Kijung Lee, Sein Kwon, Inpyo Hong and Sanghyun Park. ```(*equally contributed)```

## Description
Video anomaly detection (VAD) is crucial for video anal ysis and surveillance in computer vision. However, exist ing VAD models rely on learned normal patterns, which makes them difficult to apply to diverse environments. Con sequently, users should retrain models or develop separate AI models for new environments, which requires expertise in machine learning, high-performance hardware, and ex tensive data collection, limiting the practical usability of VAD. **To address these challenges, this study proposes customizable video anomaly detection (C-VAD) technique and introduces the C-VAD model, AnyAnomaly. C-VAD con siders user-defined text as an abnormal event and detects frames containing a specified event in a video.** We effectively implemented AnyAnomaly using a context-aware VQA ap proach without fine-tuning the large vision language model. To validate the effectiveness of the proposed model, we construct C-VAD datasets and demonstrate the superiority of AnyAnomaly. Furthermore, despite adopting a zero-shot approach, our method achieved competitive performance on the VAD benchmarks.<br/><br/>
<img width="850" alt="fig-1" src="https://github.com/user-attachments/assets/16807b72-e4b6-4a84-8550-7642f2fdf9ad">  

## Context-aware VQA
Comparison of the proposed model with the baseline. Both models perform C-VAD, but the baseline operates with frame-level VQA, whereas the proposed model employs a segment-level Context-Aware VQA.
**Context-Aware VQA is a method that performs VQA by utilizing additional contexts that describe an image.** To enhance the object analysis and action understanding capabilities of LVLM, we propose Position Context and Temporal Context.
- **Position Context Tutorial: [[Google Colab](https://colab.research.google.com/drive/1vDU6j2c9YwVEhuvBbUHx5GjorwKKI6sX?usp=sharing)]**
- **Temporal Context Tutorial: [[Google Colab](https://colab.research.google.com/drive/1xnXjvlUlB8DgbTVGRrwvuLRz2rFAbdQ5?usp=sharing)]**<br/>
<img width="850" alt="fig-2" src="https://github.com/user-attachments/assets/bf814ab4-0363-469f-8814-7dea2ad10d52">  

## Results
Table 1 and Table 2 present **the evaluation results on the C-VAD datasets (C-ShT, C-Ave).** The proposed model achieved performance improvements of **9.88% and 13.65%** over the baseline on the C-ShT and C-Ave datasets, respectively. Specifically, it showed improvements of **14.34% and 8.2%** in the action class, and **3.25% and 21.98%** in the appearance class.<br/><br/>
<img width="850" alt="fig-3" src="https://github.com/user-attachments/assets/e86a8596-063c-4b17-a029-e5e917fda88f">  
<img width="850" alt="fig-3" src="https://github.com/user-attachments/assets/d1b36414-e699-4fdb-9b29-b1ab7160afe4">  

## Qualitative Evaluation 
- **Anomaly Detection in diverse scenarios**
  
|         Text              |Demo  |
|:--------------:|:-----------:|
| **Jumping-Falling<br/>-Pickup** |![c5-2](https://github.com/user-attachments/assets/55cb83cd-3e89-4309-8bac-efb4808eb57d)|
| **Bicycle-<br/>Running** |![c6-2](https://github.com/user-attachments/assets/8b1348bc-cc86-4895-abbb-f11882586d76)|
| **Bicycle-<br/>Stroller** |![c7](https://github.com/user-attachments/assets/ead0a1bd-bce1-49b8-96f0-595773ad760c)|


- **Anomaly Detection in complex scenarios**

|         Text              |Demo  |
|:--------------:|:-----------:|
| **Driving outside<br/> lane** |![c4](https://github.com/user-attachments/assets/69dabe3b-518f-474d-96e7-7c9965caf5fe)|
| **People and car<br/> accident** |![c1](https://github.com/user-attachments/assets/3c20c39b-d0da-44f4-94cf-10a8beff2cbc)|
| **Jaywalking** |![c2](https://github.com/user-attachments/assets/4a2eda4c-1519-47ee-9984-39a25a42bbcc)|
| **Walking<br/> drunk** |![c3](https://github.com/user-attachments/assets/662c2c92-1a0b-485c-aea2-40a9ccba052d)|


## Datasets
- We processed the Shanghai Tech Campus (ShT) and CUHK Avenue (Ave) datasets to create the labels for the C-ShT and C-Ave datasets. These labels can be found in the ```ground_truth``` folder. **To test the C-ShT and C-Ave datasets, you need to first download the ShT and Ave datasets and store them in the directory corresponding to** ```'data_root'```.
- You can specify the dataset's path by editing ```'data_root'``` in ```config.py```.
  
|     CUHK Avenue    | Shnaghai Tech.    |
|:------------------------:|:-----------:|
|[Official Site](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)|[Official Site](https://svip-lab.github.io/dataset/campus_dataset.html)


## 1. Requirements and Installation For Chat-UniVi
- **Once the datasets and the Chat-UniVi model are ready, you can move the provided ```tutorial files``` to the main directory and run them directly!**
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
