import torch
import os
import sys
sys.path.append(os.path.join('LLaVA-pp/LLaVA'))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
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
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    return tokenizer, model, image_processor, context_len
    

def lvlm_test(tokenizer, model, image_processor, qs, image_path, image=None):

    conv_mode = "llama3"
    temperature = 0.2
    top_p = 0.7
    num_beams = 1
    max_new_tokens = 512
    
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    if image is None:
        image = Image.open(image_path).convert("RGB")
    
    image = [image]
    image_size = [x.size for x in image]
    
    images_tensor = process_images(
        image,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_size,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=False,
        )
    
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    outputs = outputs.replace("<|end|>", "").strip()

    return outputs