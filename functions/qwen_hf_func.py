import torch
import os
import sys
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)

from qwen_vl_utils import process_vision_info



def make_instruction(cfg, keyword):
    instruction = f"""\
- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1. 
A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. 
For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.
- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.
- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**.   
"""
    tc_instruction = f"""\
- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1. 
A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. 
For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.
- **Context**: The given image represents a sequence (row 1 column 1 → row 1 column 2 → row 2 column 1 -> row 2 column 2) illustrating temporal progression.
- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.
- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**.   
"""
    return instruction, tc_instruction


def load_lvlm(model_path):

    processor = AutoProcessor.from_pretrained(model_path)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    generation_config = {
        "temperature": 0.1,
        "top_p": 0.001,
        "repetition_penalty": 1.05,
        "max_new_tokens": 50,
        "do_sample": True,
    }

    return model, processor, generation_config


def qwen_make_messages(image, instruction):
    messages = [
        {
            "role": "system", 
            "content": "You are a vision anomaly detector."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text", 
                    "text": instruction
                },
            ],
        }
    ]
    return messages


def lvlm_test(model, processor, generation_config, message_list):

    llm_outputs = []

    device = next(model.parameters()).device

    for message in message_list:

        text = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(message)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(device)

        generated_ids = model.generate(
            **inputs,
            **generation_config
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(
                inputs.input_ids,
                generated_ids
            )
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        llm_outputs.append(output_text)

    return llm_outputs