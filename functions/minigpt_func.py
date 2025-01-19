import torch
import os
import sys
sys.path.append(os.path.join('MiniGPT-4'))
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np

from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2, CONV_VISION_Vicuna0
from minigpt4.common.config import Config
        

def make_instruction(prompt_type, keyword, temporal_context=False):
    # simple
    if prompt_type == 0:
        instruction = (
            f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1. "
            f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
            f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
            f"- **Response**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**."
        )

    # complex (+ consideration)
    elif prompt_type == 1:
        instruction = (
            f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1. "
            f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
            f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
            f"- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.\n"
            f"- **Response**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**."
        )
    
    if temporal_context == False:
        return instruction
    
    # insturction for temporal context
    else:
        tc_instruction = (
            f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1."
            f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
            f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
            f"- **Context**: The given image represents a sequence (row 1 column 1 → row 1 column 2 → row 2 column 1 -> row 2 column 2) illustrating temporal progression.\n" 
            f"- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.\n"    
            f"- **Response**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**."
        )
        return instruction, tc_instruction
    

def load_lvlm(cfg):
    parser = eval_parser()
    args = parser.parse_args(["--cfg-path", cfg.model_path])
    model, vis_processor = init_model(args)
    model.eval()
    return model, vis_processor


def lvlm_test(model, vis_processor, qs, image_path, image=None):
    if image is None:
        image = Image.open(image_path)
    
    image = image.convert('RGB')
    image = vis_processor(image)
    
    conv_temp = CONV_VISION_Vicuna0.copy()
    conv_temp.system = "Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions."
    
    text = prepare_texts([qs], conv_temp)
    
    answer = model.generate(torch.from_numpy(np.expand_dims(image,axis=0)), text, max_new_tokens=50, do_sample=False)
    
    return answer