import torch
from utils import *
from data_loader import clip_path_loader, label_loader
from config import update_config
import argparse
from fastprogress import progress_bar
from sklearn import metrics
from scipy.ndimage import gaussian_filter1d
from functions.text_func import make_text_embedding
from functions.chatunivi_func import load_lvlm, lvlm_test, make_instruction
from functions.attn_func import winclip_attention
from functions.grid_func import grid_generation
from functions.key_func import key_frame_selection_four_idx
import clip
from transformers import logging
logging.set_verbosity_error()


def main():
    parser = argparse.ArgumentParser(description='vad_using_lvlm')
    parser.add_argument('--dataset', default='avenue', type=str)
    parser.add_argument('--type', default='bicycle', type=str)
    parser.add_argument('--multiple', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--prompt_type', default=1, type=int, help='0: simple, 1: complex')
    parser.add_argument('--anomaly_detect', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--calc_auc', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--calc_video_auc', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--clip_length', default=16, type=int)
    parser.add_argument('--template_adaption', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--class_adaption', default=False, type=str2bool, nargs='?', const=True)
    parser.add_argument('--model_path', default='LVLM/weights/chatunivi', type=str)

    args = parser.parse_args()
    chatunivi_model_path = args.model_path
    cfg = update_config(args)
    cfg['model_path'] = chatunivi_model_path
    cfg.print_cfg()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    # load video names and paths
    video_names, video_paths = load_names_paths(cfg)

    # configure file
    predict_file_name = f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}/proposed_{cfg.dataset_name}_{cfg.type}_{cfg.prompt_type}.json'

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

        # load clip model
        clip_model, preprocess = clip.load('ViT-B/32', device=device)

        # processing videos
        dict_arr = []
        print_check = True

        with open(predict_file_name, 'w') as file:
            for i, video_path in progress_bar(enumerate(video_paths), total=len(video_paths)):
                predicted = []
                predicted_wa = []
                predicted_tc = []

                video_name = video_names[i]
                cp_loader = clip_path_loader(video_path, cfg.clip_length)

                # anomaly detection using LVLM (segment-level) 
                for cp in cp_loader:
                    max_score = 0 
                    max_score_wa = 0 
                    max_score_tc = 0 

                    # multiple keyword processing 
                    for keyword in keyword_list:
                        instruction, instruction_tc = make_instruction(cfg.prompt_type, keyword, True)
                        print_check = print_prompt(print_check, instruction, instruction_tc)

                        # text embedding
                        text_embedding = make_text_embedding(clip_model, device, text=keyword, type_list=cfg.type_list,
                                                              class_adaption=cfg.class_adaption, template_adaption=cfg.template_adaption)

                        # key frame selection
                        indice = key_frame_selection_four_idx(cfg.clip_length, cp, keyword, clip_model, preprocess, device)
                        key_image_path = cp[indice[0]]
                        image_paths = [cp[idx] for idx in indice[1:]]          

                        # position & temporal context 
                        wa_image = winclip_attention(key_image_path, text_embedding, clip_model, device, cfg.class_adaption, cfg.type_id)
                        grid_image = grid_generation(image_paths, keyword, clip_model, device)

                        # anomaly detection 
                        response = lvlm_test(tokenizer, model, image_processor, instruction, key_image_path, None)
                        response_wa = lvlm_test(tokenizer, model, image_processor, instruction, None, wa_image)
                        response_tc = lvlm_test(tokenizer, model, image_processor, instruction_tc, None, grid_image)

                        score = generate_output(response)['score']
                        score_wa = generate_output(response_wa)['score']
                        score_tc = generate_output(response_tc)['score']

                        max_score = max(max_score, score)
                        max_score_wa = max(max_score_wa, score_wa)
                        max_score_tc = max(max_score_tc, score_tc)

                    # save frame scores
                    for _ in range(cfg.clip_length):
                        predicted.append(max_score)
                        predicted_wa.append(max_score_wa)
                        predicted_tc.append(max_score_tc)

                output_dict = {'video':video_name,
                               'scores':predicted,
                               'scores_wa':predicted_wa,
                               'scores_tc':predicted_tc}
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

        '''
        =======================
        original operation
        =======================
        '''
        best_auc = 0 
        for sigma in range(1, 100):
            # post-processing
            g_predicted = gaussian_filter1d(predicted, sigma=sigma)
            mm_predicted = min_max_normalize(g_predicted)

            # get auc
            fpr, tpr, _ = metrics.roc_curve(labels, mm_predicted, pos_label=1)
            auc = metrics.auc(fpr, tpr)

            # update best auc
            if auc > best_auc:
                best_predicted = mm_predicted
                best_auc = auc

        org_best_predicted = best_predicted
        org_best_auc = best_auc

        print('org auc:', best_auc)
        print('-----------------------------------')
        anomalies_idx = [i for i,l in enumerate(labels) if l==1] 
        graph_path = f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}/proposed_org_{cfg.dataset_name}_{cfg.type}_{cfg.prompt_type}.jpg'
        save_score_auc_graph(anomalies_idx, org_best_predicted, org_best_auc, graph_path)

    
        '''
        ================================
        combination operation (org+wa)
        ================================
        '''
        best_auc = 0 
        best_alpha = 0
        for sigma in range(1, 100):
            for num in np.arange(0.0, 1.1, 0.1):
                # scoring
                a1 = round(num, 1)
                a2 = round(1-a1, 1)
                agg_predicted = a1*predicted + a2*predicted_wa

                # post-processing
                g_predicted = gaussian_filter1d(agg_predicted, sigma=sigma)
                mm_predicted = min_max_normalize(g_predicted)

                # get auc
                fpr, tpr, _ = metrics.roc_curve(labels, mm_predicted, pos_label=1)
                auc = metrics.auc(fpr, tpr)

                # update best auc
                if auc > best_auc:
                    best_auc = auc
                    best_predicted = mm_predicted
                    best_alpha = a1

        ow_best_predicted = best_predicted
        ow_best_auc = best_auc

        print('best auc:', best_auc)
        print(f'best alpha: ({best_alpha})*original+({1-best_alpha})*wa')
        print('-----------------------------------')
        anomalies_idx = [i for i,l in enumerate(labels) if l==1] 
        graph_path = f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}/proposed_ow_{cfg.dataset_name}_{cfg.type}_{cfg.prompt_type}.jpg'
        save_score_auc_graph(anomalies_idx, ow_best_predicted, ow_best_auc, graph_path)


        '''
        ================================
        combination operation (org+tc)
        ================================
        '''
        best_auc = 0 
        best_alpha = 0
        for sigma in range(1, 100):
            for num in np.arange(0.0, 1.1, 0.1):
                # scoring
                a1 = round(num, 1)
                a2 = round(1-a1, 1)
                agg_predicted = a1*predicted + a2*predicted_tc

                # post-processing
                g_predicted = gaussian_filter1d(agg_predicted, sigma=sigma)
                mm_predicted = min_max_normalize(g_predicted)

                # get auc
                fpr, tpr, _ = metrics.roc_curve(labels, mm_predicted, pos_label=1)
                auc = metrics.auc(fpr, tpr)

                # update best auc
                if auc > best_auc:
                    best_auc = auc
                    best_predicted = mm_predicted
                    best_alpha = a1

        ot_best_predicted = best_predicted
        ot_best_auc = best_auc

        print('best auc:', best_auc)
        print(f'best alpha: ({best_alpha})*original+({1-best_alpha})*tc')
        print('-----------------------------------')
        anomalies_idx = [i for i,l in enumerate(labels) if l==1] 
        graph_path = f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}/proposed_ot_{cfg.dataset_name}_{cfg.type}_{cfg.prompt_type}.jpg'
        save_score_auc_graph(anomalies_idx, ot_best_predicted, ot_best_auc, graph_path)


        '''
        ================================
        combination operation (proposed)
        ================================
        '''
        best_auc = 0 
        best_alpha = 0
        best_beta = 0
        best_gamma = 0

        for sigma in range(1, 100):
            for a1 in np.arange(0.0, 1.1, 0.1):  
                for a2 in np.arange(0.0, 1.1, 0.1):  
                    a3 = 1 - a1 - a2
                    if 0 <= a3 <= 1: 
                        agg_predicted = a1*predicted + a2*predicted_wa + a3*predicted_tc

                        # post-processing
                        g_predicted = gaussian_filter1d(agg_predicted, sigma=sigma)
                        mm_predicted = min_max_normalize(g_predicted)

                        # get auc
                        fpr, tpr, _ = metrics.roc_curve(labels, mm_predicted, pos_label=1)
                        auc = metrics.auc(fpr, tpr)

                        # update best auc
                        if auc > best_auc:
                            best_auc = auc
                            best_predicted = mm_predicted
                            best_alpha = a1
                            best_beta = a2
                            best_gamma = a3

        combi_best_predicted = best_predicted
        combi_best_auc = best_auc

        print('best auc:', best_auc)
        print(f'best alpha: ({best_alpha})*original+({best_beta})*wa+({best_gamma})*tc')
        print('-----------------------------------')
        anomalies_idx = [i for i,l in enumerate(labels) if l==1] 
        graph_path = f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}/proposed_combi_{cfg.dataset_name}_{cfg.type}_{cfg.prompt_type}.jpg'
        save_score_auc_graph(anomalies_idx, combi_best_predicted, combi_best_auc, graph_path)


    # save auc per video
    if cfg.calc_video_auc:
        print('--------------------------------------')
        print('save individual anomaly scores...')
        print('--------------------------------------')

        for i in progress_bar(range(len(label_arr)), total=len(label_arr)):
            video_name = video_names[i]
            video_gt = label_arr[i]

            if i == 0:
                len_past = 0
            else:
                len_past = len_past+len_present
            len_present = len(label_arr[i])

            video_pd = combi_best_predicted[len_past:len_past+len_present]
            video_anomalies_idx = [j for j,l in enumerate(video_gt) if l==1] 
            graph_path = f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}/videos/proposed_combi_{cfg.dataset_name}_{video_name}_{cfg.type}_{cfg.prompt_type}.jpg'
            save_score_graph(video_anomalies_idx, video_pd, graph_path)


if __name__=="__main__":
    main()