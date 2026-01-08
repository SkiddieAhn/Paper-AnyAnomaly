import torch
from utils import *
from data_loader import frame_path_loader, label_loader
from config import update_config
import argparse
from fastprogress import progress_bar
from sklearn import metrics
from scipy.ndimage import gaussian_filter1d
from functions.MiniCPM_func import load_lvlm, lvlm_test, make_instruction
from transformers import logging
logging.set_verbosity_error()


def main():
    parser = argparse.ArgumentParser(description='vad_using_lvlm')
    parser.add_argument('--dataset', default='avenue', type=str)
    parser.add_argument('--type', default='bicycle', type=str)
    parser.add_argument('--multiple', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--prompt_type', default=3, type=int, help='0: simple, 1: consideration, 2: reasoning, 3: reasoning+consideration')
    parser.add_argument('--anomaly_detect', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--calc_auc', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--calc_video_auc', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--clip_length', default=None, type=int)
    parser.add_argument('--template_adaption', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--class_adaption', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--kfs_num', default=0, type=int, help='not used option')
    parser.add_argument('--lge_scale', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--mid_scale', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--sml_scale', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--stride', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--model_path', default='MiniCPM-Llama3-V-2_5', type=str)
    parser.add_argument('--grid_search', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--sigma_range', default='1,30', type=str)
    parser.add_argument('--weight_step', default=0, help='not used option')
    parser.add_argument('--sigma', default=15, type=int)
    parser.add_argument('--alpha', default=0, type=float, help='not used option')
    parser.add_argument('--beta', default=0, type=float, help='not used option')
    parser.add_argument('--gamma', default=0, type=float, help='not used option')

    args = parser.parse_args()
    cfg = update_config(args)
    cfg.print_cfg()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    video_names, video_paths = load_names_paths(cfg)

    predict_file_name = f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}/minicpm_{cfg.dataset_name}_{cfg.type}_{cfg.prompt_type}.json'

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
        tokenizer, model = load_lvlm(cfg.model_path, device)

        dict_arr = []
        print_check = True

        with open(predict_file_name, 'w') as file:
            for i, video_path in progress_bar(enumerate(video_paths), total=len(video_paths)):
                predicted = []
                video_name = video_names[i]
                frame_loader = frame_path_loader(video_path)

                for fr in progress_bar(frame_loader, total=len(frame_loader)):
                    max_score = 0 

                    for keyword in keyword_list:
                        instruction = make_instruction(cfg, keyword)
                        print_check = print_prompt(print_check, instruction)
                        response = lvlm_test(tokenizer, model, instruction, fr)
                        score = generate_output(response)['score']
                        max_score = max(max_score, score)
                    predicted.append(max_score)

                output_dict = {'video':video_name,
                               'scores':predicted}
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
        gt = np.concatenate(gt_arr, axis=0)

        predicted = []
        with open(predict_file_name, 'r') as file:
            data = json.load(file)
            for item in data:
                predicted.append(np.array(item['scores']))
            predicted = np.concatenate(predicted, axis=0)

        # Determine sigma search range
        if cfg.grid_search:
            sigma_start, sigma_end = map(int, cfg.sigma_range.split(','))
            sigma_range = range(sigma_start, sigma_end)
            print('--------------------------------------')
            print('Starting grid search for optimal sigma...')
            print(f'Sigma range: [{sigma_start}, {sigma_end})')
            print('--------------------------------------')
        else:
            sigma_range = [cfg.sigma]
        
        # Search for best sigma
        best_auc = 0
        best_sigma = 0
        best_predicted = None
        
        sigma_iter = progress_bar(sigma_range, total=len(sigma_range)) if cfg.grid_search else sigma_range
        
        for sigma in sigma_iter:
            g_predicted = gaussian_filter1d(predicted, sigma=sigma)
            mm_predicted = min_max_normalize(g_predicted)
            
            fpr, tpr, _ = metrics.roc_curve(gt, mm_predicted, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            
            if auc > best_auc:
                best_auc = auc
                best_sigma = sigma
                best_predicted = mm_predicted
                if cfg.grid_search:
                    print(f'\nNew best found! AUC: {best_auc:.4f}, sigma: {best_sigma}')
        
        # Print results
        if cfg.grid_search:
            print('======================================')
            print('Grid Search Results:')
            print('======================================')
        print(f'best auc: {best_auc:.4f}')
        print(f'best sigma: {best_sigma}')
        if cfg.grid_search:
            print('======================================')

        anomalies_idx = [i for i, l in enumerate(gt) if l == 1]
        graph_path = f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}/minicpm_{cfg.dataset_name}_{cfg.type}_{cfg.prompt_type}.jpg'
        save_score_auc_graph(anomalies_idx, best_predicted, best_auc, graph_path)


    if cfg.calc_video_auc:
        print('--------------------------------------')
        print('save individual anomaly scores...')
        print('--------------------------------------')

        len_past = 0
        for i in progress_bar(range(len(gt_arr)), total=len(gt_arr)):
            video_name = video_names[i]
            video_gt = gt_arr[i]
            len_present = len(gt_arr[i])

            video_pd = best_predicted[len_past:len_past + len_present]
            video_pd2 = min_max_normalize(video_pd)
            video_anomalies_idx = [j for j, l in enumerate(video_gt) if l == 1]
            graph_path = f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}/videos/minicpm_{cfg.dataset_name}_{video_name}_{cfg.type}_{cfg.prompt_type}.jpg'
            save_score_graph(video_anomalies_idx, video_pd2, graph_path)
            
            len_past += len_present


if __name__=="__main__":
    main()