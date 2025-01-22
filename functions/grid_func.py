import torch
import numpy as np
import torch.nn.functional as F
import cv2
import clip
from utils import transform2pil


def split_images_with_unfold(image_paths, kernel_size=(80, 80), stride_size=None):
    all_patches = []
    
    for image_path in image_paths:
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
        all_patches.append(patches)
    
    grouped_patches = []
    num_patches = org_patches.shape[2]

    for i in range(num_patches):
        grouped_patches.append(torch.stack([patch_set[i] for patch_set in all_patches]))
    return grouped_patches


def patch_selection(gpatches, text, model, device):
    texts = clip.tokenize([text for _ in range(1)]).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(texts).float()
        max_arr = []

        for gpatch in gpatches:
            gpatch = gpatch.to(device)
            image_features = model.encode_image(gpatch).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (text_features @ image_features.T).cpu().numpy() # (1, clip_length)
            max_arr.append(np.max(similarity))

        # key frames selection
        max_arr = np.array(max_arr)
        max_idx = np.argmax(max_arr)
    return max_idx


def grid_image_generation(gpatches, idx):
    gpatch = gpatches[idx]
    grid_image = np.zeros((3, 224, 224), dtype=np.float32)
    grid_image[:, 0:112, 0:112] = F.interpolate(gpatch[0].unsqueeze(0), size=(112, 112), mode='bilinear').numpy()
    grid_image[:, 0:112, 112:224] = F.interpolate(gpatch[1].unsqueeze(0), size=(112, 112), mode='bilinear').numpy()
    grid_image[:, 112:224, 0:112] = F.interpolate(gpatch[2].unsqueeze(0), size=(112, 112), mode='bilinear').numpy()
    grid_image[:, 112:224, 112:224] = F.interpolate(gpatch[3].unsqueeze(0), size=(112, 112), mode='bilinear').numpy()
    return grid_image


def grid_generation(cfg, image_paths, keyword, clip_model, device):
    gpatches = []
        
    if cfg.sml_scale:
        gpatches_sml = split_images_with_unfold(image_paths, kernel_size=cfg.sml_size, stride_size=cfg.sml_size_stride) 
        gpatches += [gpatch for gpatch in gpatches_sml]

    if cfg.mid_scale:
        gpatches_mid = split_images_with_unfold(image_paths, kernel_size=cfg.mid_size, stride_size=cfg.mid_size_stride) 
        gpatches += [gpatch for gpatch in gpatches_mid]

    if cfg.lge_scale:
        gpatches_lge = split_images_with_unfold(image_paths, kernel_size=cfg.lge_size, stride_size=cfg.lge_size_stride) 
        gpatches += [gpatch for gpatch in gpatches_lge]

    max_patch_idx = patch_selection(gpatches, keyword, clip_model, device)
    grid_image = grid_image_generation(gpatches, max_patch_idx)
    output_image = transform2pil(grid_image)
    return output_image
