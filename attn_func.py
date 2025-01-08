import torch
import torch.nn.functional as F
import cv2
import numpy as np
import cv2
from utils import transform2pil


def split_one_image_with_unfold(image_path, kernel_size=(80, 80), stride_size=None):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (240, 240)).astype('float32')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
    image = (image / 255)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()  # (H, W, C) -> (C, H, W) -> (1, C, H, W)
    
    if stride_size == None:
        stride_size = kernel_size

    org_patches = F.unfold(image, kernel_size=(kernel_size[0], kernel_size[1]), stride=(stride_size[0], stride_size[1]))
    patches = org_patches.permute(0, 2, 1).reshape(-1, 3, kernel_size[0], kernel_size[1])
    patches = F.interpolate(patches, size=(224, 224), mode='bilinear')
    return patches


def patch_similarity(patches, text_embedding, model, device, class_adaption=False, type_id=None):
    with torch.no_grad():        
        patches = patches.to(device)
        image_embedding = model.encode_image(patches).float()
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        similarity = (text_embedding @ image_embedding.T).cpu() # (1, patch_length) or (class_length, patch_length)

    if class_adaption:
        similarity = similarity.softmax(dim=0)
        return similarity[type_id]
    else:
        return similarity
    

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)
    

def attention(img_path, sim):
    img = cv2.cvtColor(cv2.resize(cv2.imread(img_path), (224, 224)), cv2.COLOR_BGR2RGB)
    mask = normalize(sim)
    mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)
    attn = (img * mask).astype(np.uint8)
    return attn


def winclip_attention(img_path, text_embedding, clip_model, device, class_adaption=False, type_id=None):
    patches_img = split_one_image_with_unfold(img_path, kernel_size=(240, 240)) # 1x1
    patches_lge = split_one_image_with_unfold(img_path, kernel_size=(80, 80)) # 3x3
    patches_mid = split_one_image_with_unfold(img_path, kernel_size=(48, 48)) # 5x5

    sim_img = patch_similarity(patches_img, text_embedding, clip_model, device, class_adaption, type_id).view(1, 1, 1, 1)
    sim_lge = patch_similarity(patches_lge, text_embedding, clip_model, device, class_adaption, type_id).view(1, 1, 3, 3)
    sim_mid = patch_similarity(patches_mid, text_embedding, clip_model, device, class_adaption, type_id).view(1, 1, 5, 5)

    usim_img = F.interpolate(sim_img, size=(224, 224), mode='bilinear').squeeze(0)
    usim_lge = F.interpolate(sim_lge, size=(224, 224), mode='bilinear').squeeze(0)
    usim_mid = F.interpolate(sim_mid, size=(224, 224), mode='bilinear').squeeze(0)
    usim_total = ((usim_img + usim_lge + usim_mid) / 3).squeeze(0).numpy()
    
    attentioned = attention(img_path, usim_total)
    output_image = transform2pil(attentioned, False)
    return output_image