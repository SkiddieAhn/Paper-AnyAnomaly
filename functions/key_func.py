import re
import numpy as np
import torch
import clip
from PIL import Image
import random


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

#---------------------------------------------------------------------------------------------------------------#

class KFS:
    def __init__(self, select_num, clip_length, model, preprocess, device):
        self.select_num = select_num
        self.clip_length = clip_length
        self.model = model
        self.preprocess = preprocess
        self.device = device


    def call_function(self, clip_path, anomaly_text):
        if self.select_num == 1:
            return self.key_frame_selection_random()
        elif self.select_num == 2:
            return self.key_frame_selection_clip(clip_path, anomaly_text)
        elif self.select_num == 3:
            return self.key_frame_selection_grouping_clip(clip_path, anomaly_text)
        else:
            return self.key_frame_selection_clip_grouping(clip_path, anomaly_text)
        

    def key_frame_selection_random(self):
        indice = sorted(random.sample(range(self.clip_length), 4))
        max_idx = indice[0]
        first_idx = indice[0]
        second_idx = indice[1]
        third_idx = indice[2]
        fourth_idx = indice[3]
        return max_idx, first_idx, second_idx, third_idx, fourth_idx


    def key_frame_selection_clip(self, clip_path, anomaly_text):
        images = [self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device) for img_path in clip_path]
        images = torch.cat(images)
        texts = clip.tokenize([anomaly_text for _ in range(1)]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(images).float()
            text_features = self.model.encode_text(texts).float()

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (text_features @ image_features.T).cpu().numpy() # (1, clip_length)

            # key frames selection
            top_indices = np.argsort(similarity[0])[::-1][:4]
            top_indices_sorted = sorted(top_indices)

            max_idx = top_indices[0]
            first_idx = top_indices_sorted[0]
            second_idx = top_indices_sorted[1]
            third_idx = top_indices_sorted[2]
            fourth_idx = top_indices_sorted[3]
        return max_idx, first_idx, second_idx, third_idx, fourth_idx


    def key_frame_selection_grouping_clip(self, clip_path, anomaly_text):
        images = [self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device) for img_path in clip_path]
        images = torch.cat(images)
        texts = clip.tokenize([anomaly_text for _ in range(1)]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(images).float()
            text_features = self.model.encode_text(texts).float()

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (text_features @ image_features.T).cpu().numpy() # (1, clip_length)

            # key frames selection
            max_idx = np.argmax(similarity)
            group_len = self.clip_length // 4

            first_group = similarity[0][0:group_len]
            second_group = similarity[0][group_len:group_len*2]
            third_group = similarity[0][group_len*2:group_len*3]
            fourth_group = similarity[0][group_len*3:group_len*4]

            first_idx = np.argmax(first_group)
            second_idx = group_len+np.argmax(second_group)
            third_idx = group_len*2+np.argmax(third_group)
            fourth_idx = group_len*3+np.argmax(fourth_group)
        return max_idx, first_idx, second_idx, third_idx, fourth_idx


    def key_frame_selection_clip_grouping(self, clip_path, anomaly_text):
        images = [self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device) for img_path in clip_path]
        images = torch.cat(images)
        texts = clip.tokenize([anomaly_text for _ in range(1)]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(images).float()
            text_features = self.model.encode_text(texts).float()

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (text_features @ image_features.T).cpu().numpy() # (1, clip_length)

            # key frames selection
            max_idx = np.argmax(similarity)
            group_len = self.clip_length // 4
            divide_output = max_idx % group_len

            first_idx = divide_output
            second_idx = group_len+divide_output
            third_idx = group_len*2+divide_output
            fourth_idx = group_len*3+divide_output
        return max_idx, first_idx, second_idx, third_idx, fourth_idx