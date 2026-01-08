import torch
from utils import *
from data_loader import clip_path_loader, label_loader
from config import update_config
import argparse
from fastprogress import progress_bar
from functions.text_func import make_text_embedding
from functions.qwen_vllm_func import make_instruction, load_lvlm, qwen_make_messages, lvlm_test
from functions.attn_func import winclip_attention
from functions.grid_func import grid_generation, four_generation
from functions.key_func import KFS
from functions.eval_func import evaluate_auc
import clip
from transformers import logging
logging.set_verbosity_error()

import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"


def context_generation(cfg, kfs, images, keyword, k_i, text_embedding, clip_model, device):
    key_indice = kfs.call_function(images, keyword)
    main_key = images[key_indice[0]]
    key_images = [images[idx] for idx in key_indice[1:]]  
    wa_image = winclip_attention(cfg, main_key, text_embedding, clip_model, device, cfg.class_adaption, cfg.type_ids[k_i])
    grid_image = grid_generation(cfg, key_images, keyword, clip_model, device)
    return main_key, wa_image, grid_image


def main():
    parser = argparse.ArgumentParser(description='vad_using_lvlm')
    parser.add_argument('--dataset', default='avenue', type=str)
    parser.add_argument('--type', default='bicycle', type=str)
    parser.add_argument('--multiple', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--prompt_type', default=3, type=int, help='-')
    parser.add_argument('--anomaly_detect', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--calc_auc', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--calc_video_auc', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--clip_length', default=24, type=int)
    parser.add_argument('--template_adaption', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--class_adaption', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--kfs_num', default=4, type=int, help='1: random, 2: clip, 3: grouping->clip, 4: clip->grouping')
    parser.add_argument('--lge_scale', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--mid_scale', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--sml_scale', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--stride', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--model_path', default='Qwen/Qwen2.5-VL-3B-Instruct', type=str)
    parser.add_argument('--grid_search', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--sigma_range', default='1,30', type=str)
    parser.add_argument('--weight_step', default=0.1, type=float)
    parser.add_argument('--sigma', default=15, type=int)
    parser.add_argument('--alpha', default=0.6, type=float)
    parser.add_argument('--beta', default=0.3, type=float)
    parser.add_argument('--gamma', default=0.1, type=float)

    args = parser.parse_args()
    cfg = update_config(args)
    cfg.print_cfg()

    if not cfg.grid_search:
        if not abs(cfg.alpha + cfg.beta + cfg.gamma - 1.0) < 1e-6:
            raise ValueError(f"alpha({cfg.alpha}) + beta({cfg.beta}) + gamma({cfg.gamma}) must equal 1.0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    video_names, video_paths = load_names_paths(cfg)

    predict_file_name = f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}/qwen_vllm_proposed_{cfg.dataset_name}_{cfg.type}_{cfg.prompt_type}.json'

    keyword_list = load_keyword_list(cfg)

    print('-----------------------------')
    print('keyword list:', keyword_list)
    print('-----------------------------')

    make_results_folders(cfg)

    '''
    ==============================
    Anomaly Detection
    ==============================
    '''

    if cfg.anomaly_detect:
        model, processor, sampling_params = load_lvlm(cfg.model_path)
        clip_model, preprocess = clip.load('ViT-B/32', device=device)
        kfs = KFS(cfg.kfs_num, cfg.clip_length, clip_model, preprocess, device)

        dict_arr = []
        print_check = True

        with open(predict_file_name, 'w') as file:
            for i, video_path in progress_bar(enumerate(video_paths), total=len(video_paths)):
                predicted = []
                predicted_wa = []
                predicted_tc = []

                video_name = video_names[i]
                cp_loader = clip_path_loader(video_path, cfg.clip_length)

                for cp in cp_loader:
                    max_score = 0 
                    max_score_wa = 0 
                    max_score_tc = 0 

                    for k_i, keyword in enumerate(keyword_list):
                        instruction, instruction_tc = make_instruction(cfg, keyword)
                        print_check = print_prompt(print_check, instruction, instruction_tc)

                        text_embedding = make_text_embedding(clip_model, device, text=keyword, type_list=cfg.type_list,
                                                              class_adaption=cfg.class_adaption, template_adaption=cfg.template_adaption)

                        main_key, wa_image, grid_image = context_generation(cfg, kfs, cp, keyword, k_i, text_embedding, clip_model, device)

                        messages_key = qwen_make_messages(main_key, instruction)
                        messages_wa = qwen_make_messages(wa_image, instruction)
                        messages_tc = qwen_make_messages(grid_image, instruction_tc)
                        message_list = [messages_key, messages_wa, messages_tc]

                        responses = lvlm_test(model, processor, sampling_params, message_list)
                        score = generate_output(responses[0])['score']
                        score_wa = generate_output(responses[1])['score']
                        score_tc = generate_output(responses[2])['score']

                        max_score = max(max_score, score)
                        max_score_wa = max(max_score_wa, score_wa)
                        max_score_tc = max(max_score_tc, score_tc)

                    for _ in range(cfg.clip_length):
                        predicted.append(max_score)
                        predicted_wa.append(max_score_wa)
                        predicted_tc.append(max_score_tc)

                output_dict = {'video':video_name,
                               'scores':predicted,
                               'scores_wa':predicted_wa,
                               'scores_tc':predicted_tc}
                dict_arr.append(output_dict)

                print(i, 'video:', video_path)

            json.dump(dict_arr, file, indent=4)


    '''
    ==============================
    Test AUC score
    ==============================
    '''

    if cfg.calc_auc:
        print('--------------------------------------')
        print('calculate total auc...')
        print('--------------------------------------')

        gt_loader = label_loader(cfg.cdata_root, cfg.dataset_name, cfg.type, multiple=cfg.multiple)
        gt_arr = gt_loader.load()  

        predicted = []
        predicted_wa = []
        predicted_tc = []
        label_arr = []

        with open(predict_file_name, 'r') as file:
            data = json.load(file)
            for i, item in enumerate(data):
                predicted.append(np.array(item['scores']))
                predicted_wa.append(np.array(item['scores_wa']))
                predicted_tc.append(np.array(item['scores_tc']))
                label_arr.append(gt_arr[i][:len(item['scores'])])
            predicted = np.concatenate(predicted, axis=0)
            predicted_wa = np.concatenate(predicted_wa, axis=0)
            predicted_tc = np.concatenate(predicted_tc, axis=0)
            labels = np.concatenate(label_arr, axis=0)

        eval_results = evaluate_auc(
            predicted=predicted,
            predicted_wa=predicted_wa,
            predicted_tc=predicted_tc,
            labels=labels,
            cfg=cfg,
            model_name='qwen_vllm',
            video_names=video_names,
            label_arr=label_arr
        )
        
        print('--------------------------------------')
        print('Evaluation completed!')
        print(f"Final AUC: {eval_results['combi_best_auc']:.4f}")
        print('--------------------------------------')


if __name__=="__main__":
    main()