import torch
import os
import sys
sys.path.append(os.path.join('Chat-UniVi'))
from ChatUniVi.constants import *
from ChatUniVi.conversation import conv_templates, SeparatorStyle
from ChatUniVi.model.builder import load_pretrained_model
from ChatUniVi.utils import disable_torch_init
from ChatUniVi.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image


def make_instruction(cfg, keyword, temporal_context=False):
    # simple
    if cfg.prompt_type == 0:
        instruction = (
            f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1. "
            f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
            f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
            f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, **without any additional text or explanation**."
        )

    # complex (+ consideration)
    elif cfg.prompt_type == 1:
        instruction = (
            f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1. "
            f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
            f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
            f"- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.\n"
            f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, **without any additional text or explanation**."
        )

    # complex (+ reasoning)
    elif cfg.prompt_type == 2:
        instruction = (
            f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1. "
            f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
            f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
            f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**."
        )

    # complex (+ reasoning, consideration)
    elif cfg.prompt_type == 3:
        instruction = (
            f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1. "
            f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
            f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
            f"- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.\n"
            f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**."
        )
    
    if temporal_context == False:
        return instruction
    
    # insturction for temporal context
    else:
        # simple
        if cfg.prompt_type == 0:
            tc_instruction = (
                f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1."
                f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
                f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
                f"- **Context**: The given image represents a sequence (row 1 column 1 → row 1 column 2 → row 2 column 1 -> row 2 column 2) illustrating temporal progression.\n" 
                f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, **without any additional text or explanation**."
            )

        # complex (+ consideration)
        elif cfg.prompt_type == 1:
            tc_instruction = (
                f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1."
                f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
                f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
                f"- **Context**: The given image represents a sequence (row 1 column 1 → row 1 column 2 → row 2 column 1 -> row 2 column 2) illustrating temporal progression.\n" 
                f"- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.\n"
                f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, **without any additional text or explanation**."
            )

        # complex (+ reasoning)
        elif cfg.prompt_type == 2:
            tc_instruction = (
                f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1."
                f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
                f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
                f"- **Context**: The given image represents a sequence (row 1 column 1 → row 1 column 2 → row 2 column 1 -> row 2 column 2) illustrating temporal progression.\n" 
                f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**."
            )

        # complex (+ reasoning, consideration)
        elif cfg.prompt_type == 3:
            tc_instruction = (
                f"- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1."
                f"A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all. "
                f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.\n"
                f"- **Context**: The given image represents a sequence (row 1 column 1 → row 1 column 2 → row 2 column 1 -> row 2 column 2) illustrating temporal progression.\n" 
                f"- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.\n"    
                f"- **{cfg.out_prompt}**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**."
            )

        return instruction, tc_instruction


def load_lvlm(model_path):
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = "ChatUniVi"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    return tokenizer, model, image_processor

    

def lvlm_test(tokenizer, model, image_processor, qs, image_path, image=None):
    # Sampling Parameter
    conv_mode = "simple"
    temperature = 0.2
    top_p = None
    num_beams = 1

    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    if image is None:
        image = Image.open(image_path)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            do_sample=False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=128,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs