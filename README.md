# AnyAnomaly: Zero-Shot Customizable Video Anomaly Detection with LVLM (WACV 2026)
[![arXiv](https://img.shields.io/badge/arXiv-2503.04504-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2503.04504)
[![hf](https://img.shields.io/badge/🤗-Hugging%20Face-blue.svg)](https://huggingface.co/papers/2503.04504)
[![Colab1](https://img.shields.io/badge/⚡-Colab%20Totorial%201-yellow.svg)](https://colab.research.google.com/drive/1vDU6j2c9YwVEhuvBbUHx5GjorwKKI6sX?usp=sharing)
[![Colab2](https://img.shields.io/badge/⚡-Colab%20Tutorial%202-green.svg)](https://colab.research.google.com/drive/1xnXjvlUlB8DgbTVGRrwvuLRz2rFAbdQ5?usp=sharing)

This repository is the ```official open-source``` of [AnyAnomaly: Zero-Shot Customizable Video Anomaly Detection with LVLM](https://arxiv.org/pdf/2503.04504) by Sunghyun Ahn*, Youngwan Jo*, Kijung Lee, Sein Kwon, Inpyo Hong and Sanghyun Park. ```(*equally contributed)```

## 📣 News
* **[2025/11/09]** Our paper has been accepted to **WACV 2026**!
* **[2025/09/20]** The **Qwen2.5-VL utilizing vLLM** code has been released!
* **[2025/05/10]** Our **codes and tutorials** are released!

## Description
Video anomaly detection (VAD) is crucial for video analysis and surveillance in computer vision. However, existing VAD models rely on learned normal patterns, which makes them difficult to apply to diverse environments. Consequently, users should retrain models or develop separate AI models for new environments, which requires expertise in machine learning, high-performance hardware, and extensive data collection, limiting the practical usability of VAD. **To address these challenges, this study proposes customizable video anomaly detection (C-VAD) technique and the AnyAnomaly model. C-VAD considers user-defined text as an abnormal event and detects frames containing a specified event in a video.** We effectively implemented AnyAnomaly using a context-aware visual question answering without fine-tuning the large vision language model. To validate the effectiveness of the proposed model, we constructed C-VAD datasets and demonstrated the superiority of AnyAnomaly. Furthermore, our approach showed competitive performance on VAD benchmark datasets, achieving state-of-the-art results on the UBnormal dataset and outperforming other methods in generalization across all datasets.<br/><br/>
<img width="850" alt="fig-1" src="https://github.com/user-attachments/assets/db865457-ffe5-41db-a424-51aebcc7ab4b"> 

## Context-aware VQA
Comparison of the proposed model with the baseline. Both models perform C-VAD, but the baseline operates with frame-level VQA, whereas the proposed model employs a segment-level Context-Aware VQA.
**Context-Aware VQA is a method that performs VQA by utilizing additional contexts that describe an image.** To enhance the object analysis and action understanding capabilities of LVLM, we propose Position Context and Temporal Context.
- **Position Context Tutorial: [[Google Colab](https://colab.research.google.com/drive/1vDU6j2c9YwVEhuvBbUHx5GjorwKKI6sX?usp=sharing)]**
- **Temporal Context Tutorial: [[Google Colab](https://colab.research.google.com/drive/1xnXjvlUlB8DgbTVGRrwvuLRz2rFAbdQ5?usp=sharing)]**<br/>
<img width="850" alt="fig-2" src="https://github.com/user-attachments/assets/aa318443-9e12-4500-bdb6-2601adb23d0a">  

## Results
Table 1 and Table 2 present **the evaluation results on the C-VAD datasets (C-ShT, C-Ave).** The proposed model achieved performance improvements of **9.88% and 13.65%** over the baseline on the C-ShT and C-Ave datasets, respectively. Specifically, it showed improvements of **14.34% and 8.2%** in the action class, and **3.25% and 21.98%** in the appearance class.<br/><br/>
<img width="850" alt="fig-3" src="https://github.com/user-attachments/assets/136cd177-b84f-4d23-af52-2528afebeee2">  
<img width="850" alt="fig-3" src="https://github.com/user-attachments/assets/bc69c74b-adf5-472b-b6f3-e7149a8a2d71">  

## Qualitative Evaluation 
- **Anomaly Detection in Diverse scenarios**
  
|         Text              |Demo  |
|:--------------:|:-----------:|
| **Jumping-Falling<br/>-Pickup** |![c5-2](https://github.com/user-attachments/assets/8c7e9814-da21-4879-b085-b77530bd8592)|
| **Bicycle-<br/>Running** |![c6-2](https://github.com/user-attachments/assets/c855fba6-c823-4fff-8c99-44e573c5c0bd)|
| **Bicycle-<br/>Stroller** |![c7](https://github.com/user-attachments/assets/c5df5b82-e938-42f7-8ee7-dbde517198ec)|


- **Anomaly Detection in Complex scenarios**

|         Text              |Demo  |
|:--------------:|:-----------:|
| **Driving outside<br/> lane** |![c4](https://github.com/user-attachments/assets/fd3bb6c7-16b5-4d54-bea0-ae002d7e34a9)|
| **People and car<br/> accident** |![c1](https://github.com/user-attachments/assets/511738d1-f38e-46f6-ba38-535ecb3f2a42)|
| **Jaywalking** |![c2](https://github.com/user-attachments/assets/203f92c0-923d-4dfd-bdfa-4e44bbd112ed)|
| **Walking<br/> drunk** |![c3](https://github.com/user-attachments/assets/f22126b0-6d5b-4c0c-a12a-0747d11bc01a)|

## Datasets
- We processed the Shanghai Tech Campus (ShT) and CUHK Avenue (Ave) datasets to create the labels for the C-ShT and C-Ave datasets. These labels can be found in the ```ground_truth``` folder. **To test the C-ShT and C-Ave datasets, you need to first download the ShT and Ave datasets and store them in the directory corresponding to** ```'data_root'```.
- You can specify the dataset's path by editing ```'data_root'``` in ```config.py```.
  
|     CUHK Avenue    | Shnaghai Tech.    | Quick Download |
|:------------------------:|:-----------:|:-----------:|
|[Official Site](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)|[Official Site](https://svip-lab.github.io/dataset/campus_dataset.html)|[GitHub Page](https://github.com/SkiddieAhn/Paper-VideoPatchCore/blob/main/DATA_README.md)


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


### Command
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

### Command
```Shell
# Baseline model (MiniCPM-V) → C-ShT
python -u vad_MiniCPM.py --dataset=shtech --type=falling 
# proposed model (AnyAomaly) → C-ShT
python -u vad_proposed_MiniCPM.py --dataset=shtech --type=falling 
# proposed model (AnyAnomaly) → C-ShT, diverse anomaly scenarios
python -u vad_proposed_MiniCPM.py --dataset=shtech --multiple=True --type=jumping-falling-pickup
```

## 3. Requirements and Installation For Qwen2.5-VL (vLLM)
- ```Qwen2.5-VL```: [[GitHub]](https://github.com/QwenLM/Qwen2.5-VL) [[vLLM]](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html)
- weights: Qwen2.5-VL 3B [[Huggingface]](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), Qwen2.5-VL 7B [[Huggingface]](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- Install required packages:
```bash
pip install -U vllm --torch-backend auto
pip install git+https://github.com/huggingface/transformers accelerate
pip install flash-attn
pip install qwen-vl-utils[decord]==0.0.8
pip install -r requirements.txt
```

### Command
```Shell
# proposed model (AnyAomaly) → C-ShT
python -u vad_proposed_Qwen_vllm.py --dataset=shtech --type=falling 
# proposed model (AnyAnomaly) → C-ShT, diverse anomaly scenarios
python -u vad_proposed_Qwen_vllm.py --dataset=shtech --multiple=True --type=jumping-falling-pickup
```

## Citation
If you use our work, please consider citing:  
```Shell
@article{ahn2025anyanomaly,
  title={AnyAnomaly: Zero-Shot Customizable Video Anomaly Detection with LVLM},
  author={Ahn, Sunghyun and Jo, Youngwan and Lee, Kijung and Kwon, Sein and Hong, Inpyo and Park, Sanghyun},
  journal={arXiv preprint arXiv:2503.04504},
  year={2025}
}
```

## Contact
Should you have any question, please create an issue on this repository or contact me at skd@yonsei.ac.kr.
