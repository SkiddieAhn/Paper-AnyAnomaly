# AnyAnomaly
AnyAnomaly: A Zero-shot Approach for Customizable Video Anomaly Detection using LVLM
  
## Customizable VAD
![image](https://github.com/user-attachments/assets/12201aec-c562-4884-941d-591318ef5da2)

## AnyAnomaly Model
- ```Position Context tutorial code```: [[Colab]](https://colab.research.google.com/drive/1_BRBkodZeIJLbGqs5r4AO76QqZBeQ5WP)    
- ```Temporal Context tutorial code```: [[Colab]](https://colab.research.google.com/drive/1Am4d2yMRypMnmvrb11QWco70at9paEb9#scrollTo=3QGYNpk90Vvq)
  
![image](https://github.com/user-attachments/assets/f621d667-6079-41ce-8401-3441b9d4b8da)


## 1. Requirements and Installation For ChatUniVi
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

# Download the Model (Chat-UniVi 7B)
mkdir weights
cd weights
sudo apt-get install git-lfs
git lfs install
git lfs clone https://huggingface.co/Chat-UniVi/Chat-UniVi

# Download extra packages
cd ../../
pip install -r requirements.txt
pip install numpy==1.24.3
```


## Command
- ```avenue type```: [too_close, bicycle, throwing, running, dancing]
- ```shtech type```: [car, bicycle, fighting, throwing, hand_truck, running, skateboarding, falling, jumping, loitering, motorcycle]
```Shell
# Baseline model (Chat-UniVi) -> Customizable-STC
python -u vad_chatunivi.py --dataset=shtech --type=falling
# proposed model (AnyAomaly) -> Customizable-STC
python -u vad_proposed_chatunivi.py --dataset=shtech --type=falling 
# proposed model (AnyAomaly) -> STC
python -u ovad_proposed_chatunivi.py --dataset=shtech
```

## 2. Requirements and Installation For MiniCPM
- ```MiniCPM```: [[GitHub]](https://github.com/OpenBMB/MiniCPM-V.git)
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
- ```avenue type```: [too_close, bicycle, throwing, running, dancing]
- ```shtech type```: [car, bicycle, fighting, throwing, hand_truck, running, skateboarding, falling, jumping, loitering, motorcycle]
- ```model path```: MiniCPM-V-2_6, MiniCPM-V-2_6-int4, MiniCPM-Llama3-V-2_5, MiniCPM-Llama3-V-2_5-int4, MiniCPM-V-2, MiniCPM-V
```Shell
# Baseline model (MiniCPM) -> Customizable-STC
python -u vad_MiniCPM.py --dataset=shtech --type=falling --model_path=MiniCPM-Llama3-V-2_5
# proposed model (AnyAomaly) -> Customizable-STC
python -u vad_proposed_MiniCPM.py --dataset=shtech --type=falling --model_path=MiniCPM-Llama3-V-2_5
# proposed model (AnyAomaly) -> STC
python -u ovad_proposed_MiniCPM.py --dataset=shtech
```


## 3. Requirements and Installation For MiniGPT-4
- ```MiniGPT-4```: [[GitHub]](https://github.com/Vision-CAIR/MiniGPT-4.git)
- Install required packages:
```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigptv

# Download extra packages
cd ../
pip install -r requirements.txt
```

## Preparing LVLM checkpoints
- ```MiniGPT-4```: [[GitHub]](https://github.com/Vision-CAIR/MiniGPT-4.git)
- LLM weights: Llama 2 Chat 7B, Vicuna V0 13B, Vicuna V0 7B
- Model checkpoints; MiniGPT-4 (Vicuna 13B), MiniGPT-4 (Vicuna 7B), MiniGPT-4 (LLaMA-2 Chat 7B)
- set the path on the config file for LLM weights and model checkpoints
    - LLM weights
        - LLaMA-2: MiniGPT-4/minigpt4/configs/models/minigpt4_llama2.yaml Line 15
        - Vicuna: MiniGPT-4/minigpt4/configs/models/minigpt4_vicuna0.yaml Line 18
    - Model Checkpoints
        - LLaMA-2: MiniGPT-4/eval_configs/minigpt4_llama2_eval.yaml Line 8
        - Vicuna: MiniGPT-4/eval_configs/minigpt4_eval.yaml Line 8

## Command
- ```avenue type```: [too_close, bicycle, throwing, running, dancing]
- ```shtech type```: [car, bicycle, fighting, throwing, hand_truck, running, skateboarding, falling, jumping, loitering, motorcycle]
- ```model path```: minigpt4_llama2_eval.yaml, minigpt4_eval.yaml
```Shell
# Baseline model (MiniGPT-4) -> Customizable-STC
python -u vad_minigpt.py --dataset=shtech --type=falling --model_path=MiniGPT-4/eval_configs/minigpt4_llama2_eval.yaml
# proposed model (AnyAomaly) -> Customizable-STC
python -u vad_proposed_minigpt.py --dataset=shtech --type=falling --model_path=MiniGPT-4/eval_configs/minigpt4_llama2_eval.yaml
# proposed model (AnyAomaly) -> STC
python -u ovad_proposed_minigpt.py --dataset=shtech
```


## 4. Requirements and Installation For LLaVA-pp
- ```LLaVA-pp```: [[GitHub]](https://github.com/mbzuai-oryx/LLaVA-pp)
- Install required packages:
```bash
git clone https://github.com/mbzuai-oryx/LLaVA-pp.git
cd LLaVA-pp
git submodule update --init --recursive

# for LLaMA-3-V
cp LLaMA-3-V/train.py LLaVA/llava/train/train.py
cp LLaMA-3-V/conversation.py LLaVA/llava/conversation.py
cp LLaMA-3-V/builder.py LLaVA/llava/model/builder.py
cp LLaMA-3-V/llava_llama.py LLaVA/llava/model/language_model/llava_llama.py

# Create enviroments
conda create -n llava
conda activate llava
cd LLaVA
pip install --upgrade pip
pip install -e .
pip install git+https://github.com/huggingface/transformers@a98c41798cf6ed99e1ff17e3792d6e06a2ff2ff3
pip install --upgrade transformers

# Download the Model (LLaVA-Meta=Llama-3-8B-Instruct-FT)
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/MBZUAI/LLaVA-Meta-Llama-3-8B-Instruct-FT

# Download extra packages 
cd ../../
pip install -r requirements.txt
```

## Command
- ```avenue type```: [too_close, bicycle, throwing, running, dancing]
- ```shtech type```: [car, bicycle, fighting, throwing, hand_truck, running, skateboarding, falling, jumping, loitering, motorcycle]
- ```model path```: LLaVA-Meta-Llama-3-8B-Instruct-FT, ...
```Shell
# Baseline model (LLaVA-pp) -> Customizable-STC
python -u vad_llavapp.py --dataset=shtech --type=falling --model_path=LLaVA-pp/LLaVA/LLaVA-Meta-Llama-3-8B-Instruct-FT
# proposed model (AnyAomaly) -> Customizable-STC
python -u vad_proposed_llavapp.py --dataset=shtech --type=falling --model_path=LLaVA-pp/LLaVA/LLaVA-Meta-Llama-3-8B-Instruct-FT
# proposed model (AnyAomaly) -> STC
python -u ovad_proposed_llavapp.py --dataset=shtech
```

