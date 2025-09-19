import torch
import os
import sys
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
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
    model = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": 10, "video": 10}
    )

    processor = AutoProcessor.from_pretrained(model_path)

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=50,
        stop_token_ids=[],
    )
    return model, processor, sampling_params


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


def lvlm_test(model, processor, sampling_params, message_list):
    llm_inputs = []
    for message in message_list:
        prompt = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, _, _ = process_vision_info(message, return_video_kwargs=True)

        mm_data = {}
        mm_data["image"] = image_inputs

        llm_input = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
        llm_inputs.append(llm_input)

    outputs = model.generate(llm_inputs, sampling_params=sampling_params, use_tqdm=False)

    llm_outputs = []
    for output in outputs:
        generated_text = output.outputs[0].text
        llm_outputs.append(generated_text)

    return llm_outputs