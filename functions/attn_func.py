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


def total_fusion(data1, data2, data3):
    inputs = [data1, data2, data3]
    valid_data = [x for x in inputs if x is not None]
    avg_data = sum(valid_data) / len(valid_data)
    out_data = avg_data.squeeze(0).numpy()
    return out_data


def winclip_attention(cfg, img_path, text_embedding, clip_model, device, class_adaption=False, type_id=None):
    usim_sml = usim_mid = usim_lge = None

    if cfg.sml_scale:
        patches_sml = split_one_image_with_unfold(img_path, kernel_size=cfg.sml_size, stride_size=cfg.sml_size_stride) 
        sim_sml = patch_similarity(patches_sml, text_embedding, clip_model, device, class_adaption, type_id).view(1, 1, cfg.sml_patch_num[0], cfg.sml_patch_num[1])
        usim_sml = F.interpolate(sim_sml, size=(224, 224), mode='bilinear').squeeze(0)

    if cfg.mid_scale:
        patches_mid = split_one_image_with_unfold(img_path, kernel_size=cfg.mid_size, stride_size=cfg.mid_size_stride) 
        sim_mid = patch_similarity(patches_mid, text_embedding, clip_model, device, class_adaption, type_id).view(1, 1, cfg.mid_patch_num[0], cfg.mid_patch_num[1])
        usim_mid = F.interpolate(sim_mid, size=(224, 224), mode='bilinear').squeeze(0)

    if cfg.lge_scale:
        patches_lge = split_one_image_with_unfold(img_path, kernel_size=cfg.lge_size, stride_size=cfg.lge_size_stride) 
        sim_lge = patch_similarity(patches_lge, text_embedding, clip_model, device, class_adaption, type_id).view(1, 1, cfg.lge_patch_num[0], cfg.lge_patch_num[1])
        usim_lge = F.interpolate(sim_lge, size=(224, 224), mode='bilinear').squeeze(0)

    usim_total = total_fusion(usim_sml, usim_mid, usim_lge)    
    attentioned = attention(img_path, usim_total)
    output_image = transform2pil(attentioned, False)
    return output_image