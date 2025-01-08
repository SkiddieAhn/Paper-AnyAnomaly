import torch
from utils import *
from data_loader import frame_path_loader, label_loader
from lvlm_func import load_lvlm, lvlm_test, make_instruction
from config import update_config
import argparse
from fastprogress import progress_bar
from sklearn import metrics
from scipy.ndimage import gaussian_filter1d


def main():
    parser = argparse.ArgumentParser(description='vad_using_lvlm')
    parser.add_argument('--dataset', default='avenue', type=str)
    parser.add_argument('--type', default='bicycle', type=str)
    parser.add_argument('--multiple', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--prompt_type', default=1, type=int, help='0: simple, 1: complex')
    parser.add_argument('--anomaly_detect', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--calc_auc', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--calc_video_auc', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--clip_length', default=None, type=int)

    args = parser.parse_args()
    cfg = update_config(args)
    cfg.print_cfg()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    # load video names and paths
    video_names, video_paths = load_names_paths(cfg)

    # configure file
    predict_file_name = f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}/{cfg.dataset_name}_{cfg.type}_{cfg.prompt_type}.json'

    # load keyword list
    keyword_list = load_keyword_list(cfg)

    print('-----------------------------')
    print('keyword list:', keyword_list)
    print('-----------------------------')

    # make retults folders
    make_results_folders(cfg)

    '''
    ==============================
    Anomaly Detection
    ==============================
    '''

    # anomaly detection
    if cfg.anomaly_detect:
        # load lvlm
        tokenizer, model, image_processor = load_lvlm(cfg.model_path)

        # processing videos
        dict_arr = []
        print_check = True

        with open(predict_file_name, 'w') as file:
            for i, video_path in progress_bar(enumerate(video_paths), total=len(video_paths)):
                predicted = []
                video_name = video_names[i]
                frame_loader = frame_path_loader(video_path)

                # anomaly detection using LVLM (frame-level) 
                for fr in frame_loader:
                    max_score = 0 

                    # multiple keyword processing 
                    for keyword in keyword_list:
                        instruction = make_instruction(cfg.prompt_type, keyword)
                        print_check = print_prompt(print_check, instruction)
                        response = lvlm_test(tokenizer, model, image_processor, instruction, fr)
                        score = generate_output(response)['score']
                        max_score = max(max_score, score)
                    predicted.append(max_score)

                output_dict = {'video':video_name,
                               'scores':predicted}
                dict_arr.append(output_dict)

                # one video ok!
                print(i, 'video:', video_path)

            # save json file
            json.dump(dict_arr, file, indent=4)


    '''
    ==============================
    Test AUC score
    ==============================
    '''

    # calculate auc
    if cfg.calc_auc:
        print('--------------------------------------')
        print('calculate total auc...')
        print('--------------------------------------')

        gt_loader = label_loader(cfg.cdata_root, cfg.dataset_name, cfg.type, multiple=cfg.multiple)
        gt_arr = gt_loader.load()  
        gt =  np.concatenate(gt_arr, axis=0)

        predicted = []
        with open(predict_file_name, 'r') as file:
            data = json.load(file)
            for item in data:
                predicted.append(np.array(item['scores']))
            predicted = np.concatenate(predicted, axis=0)

        best_auc = 0 
        for sigma in range(1, 100):
            # post-processing
            g_predicted = gaussian_filter1d(predicted, sigma=sigma)
            mm_predicted = min_max_normalize(g_predicted)

            # get auc
            fpr, tpr, _ = metrics.roc_curve(gt, mm_predicted, pos_label=1)
            auc = metrics.auc(fpr, tpr)

            # update best auc
            if auc > best_auc:
                best_predicted = mm_predicted  
                best_auc = auc


        print('total auc:', best_auc)
        anomalies_idx = [i for i,l in enumerate(gt) if l==1] 
        graph_path = f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}/{cfg.dataset_name}_{cfg.type}_{cfg.prompt_type}.jpg'
        save_score_auc_graph(anomalies_idx, best_predicted, best_auc, graph_path)


    # save auc per video
    if cfg.calc_video_auc:
        print('--------------------------------------')
        print('save individual anomaly scores...')
        print('--------------------------------------')

        for i in progress_bar(range(len(gt_arr)), total=len(gt_arr)):
            video_name = video_names[i]
            video_gt = gt_arr[i]

            if i == 0:
                len_past = 0
            else:
                len_past = len_past+len_present
            len_present = len(gt_arr[i])

            video_pd = best_predicted[len_past:len_past+len_present]
            video_pd2 = min_max_normalize(video_pd)
            video_anomalies_idx = [j for j,l in enumerate(video_gt) if l==1] 
            graph_path = f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}/videos/{cfg.dataset_name}_{video_name}_{cfg.type}_{cfg.prompt_type}.jpg'
            save_score_graph(video_anomalies_idx, video_pd2, graph_path)


if __name__=="__main__":
    main()