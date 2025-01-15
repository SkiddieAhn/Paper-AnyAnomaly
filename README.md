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
- ```model name```: MiniCPM-V-2_6, MiniCPM-V-2_6-int4, MiniCPM-Llama3-V-2_5, MiniCPM-Llama3-V-2_5-int4, MiniCPM-V-2, MiniCPM-V
```Shell
# Baseline model (Chat-UniVi)
python -u vad_MiniCPM.py --dataset=shtech --type=falling --model_name=MiniCPM-Llama3-V-2_5
# proposed model (AnyAomaly)
python -u vad_proposed_MiniCPM.py --dataset=shtech --type=falling --model_name=MiniCPM-Llama3-V-2_5
```