# AnyAnomaly
AnyAnomaly: A Zero-shot Approach for Customizable Video Anomaly Detection using LVLM
  
## Customizable VAD
- AnyAnomaly: A Zero-shot Approach for Customizable Video Anomaly Detection using LVLM
![image](https://github.com/user-attachments/assets/12201aec-c562-4884-941d-591318ef5da2)

## AnyAnomaly Model
- ```Position Context tutorial code```: [[Link]](https://colab.research.google.com/drive/1_BRBkodZeIJLbGqs5r4AO76QqZBeQ5WP)    
- ```Temporal Context tutorial code```: [[Link]](https://colab.research.google.com/drive/1Am4d2yMRypMnmvrb11QWco70at9paEb9#scrollTo=3QGYNpk90Vvq)
  
![image](https://github.com/user-attachments/assets/f621d667-6079-41ce-8401-3441b9d4b8da)

## Command
- ```avenue type```: [too_close, bicycle, throwing, running, dancing]
- ```shtech type```: [car, bicycle, fighting, throwing, hand_truck, running, skateboarding, falling, jumping, loitering, motorcycle]
```Shell
# Baseline model (Chat-UniVi)
python -u vad_org.py --dataset=shtech --type=falling 
# proposed model (AnyAomaly)
python -u vad_proposed.py --dataset=shtech --type=falling 
```
