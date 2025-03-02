import matplotlib.pyplot as plt
import numpy as np
import json
import random
import torch
import os
import argparse
import re
import textwrap
from PIL import Image


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_long_string(long_string, width=70):
    wrapped_string = textwrap.fill(long_string, width)
    print(wrapped_string)


def print_prompt(check, prompt, prompt2=None):
    if check: 
        print('==========================================')
        # print_long_string(prompt)
        print(prompt)
        if prompt2 != None:
            print('--------------------------------------------------')
            print(prompt2)
            print('--------------------------------------------------')
        print('==========================================')
    return False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def make_results_folders(cfg):
    if not os.path.exists(f'results/{cfg.dataset_name}'):
        os.mkdir(f'results/{cfg.dataset_name}')
    if not os.path.exists(f'results/{cfg.dataset_name}/{cfg.type}'):
        os.mkdir(f'results/{cfg.dataset_name}/{cfg.type}')
    if not os.path.exists(f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}'):
        os.mkdir(f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}')
    if not os.path.exists(f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}/videos'):
        os.mkdir(f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}/videos')


def extract_score(text):
    match = re.search(r'(\d+(\.\d+)?)', text)
    if match:
        return float(match.group(0))
    else:
        return 0.0
    

def generate_output(text):
    result = {
        "score": extract_score(text),
        "reason": text
    }
    return result


def find_except_arr(cfg):
    except_arr = np.array([])
    with open(cfg.m_json_path, "r", encoding='utf-8') as json_file:
        data = json.load(json_file)
        for key in data:
            except_arr = np.concatenate((except_arr, data[key]))
        except_arr.sort()
    return except_arr


def min_max_normalize(arr, eps=1e-8):
    min_val = np.min(arr)
    max_val = np.max(arr)
    denominator = max_val - min_val + eps  # Avoid division by zero
    normalized_arr = (arr - min_val) / denominator
    return normalized_arr


def save_two_graph(answers_idx, scores1, scores2, auc1, auc2, file_path, x='Frame', y='Anomaly Score', lb1='max', lb2='text'):
    length = len(scores1)
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.plot([num for num in range(length)],[score for score in scores1], label=f'{lb1}: AUC={auc1:.3f}%', color='black') # plotting
    plt.plot([num for num in range(length)],[score for score in scores2], label=f'{lb2}: AUC={auc2:.3f}%', color='blue') # plotting
    plt.bar(answers_idx, 1, width=1, color='#BBCBE7',alpha=0.5, label='Ground-truth') # check answer
    plt.xlabel(x, fontsize=12)
    plt.ylabel(y, fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig(file_path)


def save_score_auc_graph(answers_idx, scores, auc, file_path, x='Frame', y='Anomaly Score'):
    length = len(scores)
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.ylim([0.0, 1.0])
    plt.plot([num for num in range(length)],[score for score in scores], label=f'Predicted: AUC={auc:.3f}') # plotting
    plt.bar(answers_idx, max(scores), width=1, color='#BBCBE7',alpha=0.5, label='Ground-truth') # check answer
    plt.xlabel(x, fontsize=14)
    plt.ylabel(y, fontsize=14)
    plt.legend(fontsize=14)
    plt.savefig(file_path)


def save_score_graph(answers_idx, scores, file_path, x='Frame', y='Anomaly Score'):
    length = len(scores)
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.plot([num for num in range(length)],[score for score in scores]) # plotting
    plt.bar(answers_idx, 1.0, width=1, color='#BBCBE7',alpha=0.5) # check answer
    plt.xlabel(x)
    plt.ylabel(y)
    plt.ylim(0, 1)
    plt.legend(fontsize=12)
    plt.savefig(file_path)


def load_names_paths(cfg):
    # load multiple video paths
    if cfg.multiple:
        with open(cfg.m_json_path, "r", encoding='utf-8') as json_file:
            data = json.load(json_file)
            video_names = data[cfg.type]
            video_paths = [os.path.join(cfg.test_data_path, name) for name in video_names]

    # load all video paths
    else:
        except_arr = find_except_arr(cfg)
        video_names = os.listdir(cfg.test_data_path)
        video_names.sort()
        video_names = [name for name in video_names if not name in except_arr]
        video_paths = [os.path.join(cfg.test_data_path, name) for name in video_names if not name in except_arr]

    return video_names, video_paths


def load_keyword_list(cfg):
    test_data_name = cfg.type
    if cfg.multiple:
        keyword_list = test_data_name.split('-')
    else:
        keyword_list = [test_data_name]

    return keyword_list


def transform2pil(image, transform=True):
    if transform:
        image = image * 255
        image = np.transpose(image, [1, 2, 0])
        image = (image).astype(np.uint8)
    image_pil = Image.fromarray(image)
    return image_pil
