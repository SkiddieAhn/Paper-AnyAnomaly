# AnyAnomaly
AnyAnomaly: A Zero-shot Approach for Customizable Video Anomaly Detection using LVLM
  
## Customizable VAD
![image](https://github.com/user-attachments/assets/12201aec-c562-4884-941d-591318ef5da2)

## AnyAnomaly Model
- ```Position Context tutorial code```: [[Colab]](https://colab.research.google.com/drive/1_BRBkodZeIJLbGqs5r4AO76QqZBeQ5WP)    
- ```Temporal Context tutorial code```: [[Colab]](https://colab.research.google.com/drive/1Am4d2yMRypMnmvrb11QWco70at9paEb9#scrollTo=3QGYNpk90Vvq)
  
![image](https://github.com/user-attachments/assets/f621d667-6079-41ce-8401-3441b9d4b8da)

## After setting the enviroments, Install this packages
```bash
pip install h5py
pip install fastprogress
pip install scikit-learn
pip install openai-clip
```

## Requirements and Installation For ChatUniVi
- ```Chat-UniVi```: [[GitHub]](https://github.com/PKU-YuanGroup/Chat-UniVi)
- Python >= 3.10
- Install required packages:
```bash
git clone https://github.com/PKU-YuanGroup/Chat-UniVi
cd Chat-UniVi
conda create -n lvlm python=3.10 -y
conda activate lvlm
pip install --upgrade pip
pip install -e .
```

## Command
- ```avenue type```: [too_close, bicycle, throwing, running, dancing]
- ```shtech type```: [car, bicycle, fighting, throwing, hand_truck, running, skateboarding, falling, jumping, loitering, motorcycle]
```Shell
# Baseline model (Chat-UniVi)
python -u vad_chatunivi.py --dataset=shtech --type=falling
# proposed model (AnyAomaly)
python -u vad_proposed_chatunivi.py --dataset=shtech --type=falling 
```

## Requirements and Installation For MiniCPM
- ```MiniCPM```: [[GitHub]](https://github.com/OpenBMB/MiniCPM-V.git)
- Install required packages:
```bash
git clone https://github.com/OpenBMB/MiniCPM-V.git
cd MiniCPM-V
conda create -n MiniCPM-V python=3.10 -y
conda activate MiniCPM-V
pip install -r requirements.txt
```

## Command
- ```avenue type```: [too_close, bicycle, throwing, running, dancing]
- ```shtech type```: [car, bicycle, fighting, throwing, hand_truck, running, skateboarding, falling, jumping, loitering, motorcycle]
- ```model path```: MiniCPM-V-2_6, MiniCPM-V-2_6-int4, MiniCPM-Llama3-V-2_5, MiniCPM-Llama3-V-2_5-int4, MiniCPM-V-2, MiniCPM-V
```Shell
# Baseline model (Chat-UniVi)
python -u vad_MiniCPM.py --dataset=shtech --type=falling --model_path=MiniCPM-Llama3-V-2_5
# proposed model (AnyAomaly)
python -u vad_proposed_MiniCPM.py --dataset=shtech --type=falling --model_path=MiniCPM-Llama3-V-2_5
```


## Requirements and Installation For MiniGPT-4
- ```MiniGPT-4```: [[GitHub]](https://github.com/Vision-CAIR/MiniGPT-4.git)
- Install required packages:
```bash
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigptv
```

## Following the MiniGPT github for prepare pretrained LLM weights and model checkpoints
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
# Baseline model (Chat-UniVi)
python -u vad_MiniCPM.py --dataset=shtech --type=falling --model_path=MiniGPT-4/eval_configs/minigpt4_llama2_eval.yaml
# proposed model (AnyAomaly)
python -u vad_proposed_MiniCPM.py --dataset=shtech --type=falling --model_path=MiniGPT-4/eval_configs/minigpt4_llama2_eval.yaml
```