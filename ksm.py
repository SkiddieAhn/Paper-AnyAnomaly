import glob
import re
import numpy as np
import torch
import clip
from PIL import Image


def extract_numbers(file_name):
    numbers = re.findall(r'(\d+)', file_name)
    return tuple(map(int, numbers))


def key_frame_selection(clip_path, anomaly_text, model, preprocess, device):
    images = [preprocess(Image.open(img_path)).unsqueeze(0).to(device) for img_path in clip_path]
    images = torch.cat(images)
    texts = clip.tokenize([anomaly_text for _ in range(1)]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images).float()
        text_features = model.encode_text(texts).float()

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (text_features @ image_features.T).cpu().numpy() # (1, clip_length)

        # key frame selection
        max_idx = np.argmax(similarity)
        max_path = clip_path[max_idx]
    return max_path


def wa_key_frame_selection(clip_path, wa_clip_path, anomaly_text, model, preprocess, device):
    images = [preprocess(Image.open(img_path)).unsqueeze(0).to(device) for img_path in clip_path]
    images = torch.cat(images)
    texts = clip.tokenize([anomaly_text for _ in range(1)]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images).float()
        text_features = model.encode_text(texts).float()

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (text_features @ image_features.T).cpu().numpy() # (1, clip_length)

        # key frame selection
        max_idx = np.argmax(similarity)
        max_path = clip_path[max_idx]
        wa_max_path = wa_clip_path[max_idx]

    return max_path, wa_max_path


def key_frame_selection_four_idx(clip_length, clip_path, anomaly_text, model, preprocess, device):
    images = [preprocess(Image.open(img_path)).unsqueeze(0).to(device) for img_path in clip_path]
    images = torch.cat(images)
    texts = clip.tokenize([anomaly_text for _ in range(1)]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(images).float()
        text_features = model.encode_text(texts).float()

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (text_features @ image_features.T).cpu().numpy() # (1, clip_length)

        # key frames selection
        max_idx = np.argmax(similarity)
        group_len = clip_length // 4
        divide_output = max_idx % group_len

        first_idx = divide_output
        second_idx = group_len+divide_output
        third_idx = group_len*2+divide_output
        fourth_idx = group_len*3+divide_output

    return max_idx, first_idx, second_idx, third_idx, fourth_idx


