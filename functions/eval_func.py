import numpy as np
from sklearn import metrics
from scipy.ndimage import gaussian_filter1d
from fastprogress import progress_bar
from utils import min_max_normalize, save_score_auc_graph, save_score_graph


def _calculate_auc(predicted, labels, sigma):
    """Calculate AUC for given predictions and sigma value"""
    g_predicted = gaussian_filter1d(predicted, sigma=sigma)
    mm_predicted = min_max_normalize(g_predicted)
    fpr, tpr, _ = metrics.roc_curve(labels, mm_predicted, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc, mm_predicted


def _search_best_sigma(predicted, labels, cfg):
    """Search for best sigma value"""
    if cfg.grid_search:
        sigma_start, sigma_end = map(int, cfg.sigma_range.split(','))
        sigma_range = range(sigma_start, sigma_end)
    else:
        sigma_range = [cfg.sigma]
    
    best_auc = 0
    best_sigma = 0
    best_predicted = None
    
    sigma_iter = progress_bar(sigma_range, total=len(sigma_range)) if cfg.grid_search else sigma_range
    
    for sigma in sigma_iter:
        auc, mm_predicted = _calculate_auc(predicted, labels, sigma)
        
        if auc > best_auc:
            best_auc = auc
            best_sigma = sigma
            best_predicted = mm_predicted
    
    return best_auc, best_sigma, best_predicted


def _print_results(title, best_auc, best_sigma, weights=None, grid_search=False):
    """Print evaluation results"""
    if grid_search:
        print('======================================')
        print(f'{title}:')
        print('======================================')
    print(f'best auc: {best_auc:.4f}')
    print(f'best sigma: {best_sigma}')
    if weights:
        print(f'best weights: ({weights[0]:.2f})*original + ({weights[1]:.2f})*wa + ({weights[2]:.2f})*tc')
    if grid_search:
        print('======================================')
    else:
        print('-----------------------------------')


def evaluate_auc(predicted, predicted_wa, predicted_tc, labels, cfg, 
                 model_name='model', video_names=None, label_arr=None):
    """
    Evaluate AUC with either grid search or manual hyperparameters
    
    Args:
        predicted: Original anomaly scores
        predicted_wa: WinCLIP attention scores
        predicted_tc: Temporal context scores
        labels: Ground truth labels
        cfg: Configuration object
        model_name: Model name for saving results (e.g., 'chatunivi', 'minicpm')
        video_names: List of video names (for per-video evaluation)
        label_arr: List of labels per video (for per-video evaluation)
    
    Returns:
        dict: Results containing best_auc, best_predicted, and hyperparameters
    """
    
    results = {}
    anomalies_idx = [i for i, l in enumerate(labels) if l == 1]
    
    '''
    =======================
    original operation
    =======================
    '''
    if cfg.grid_search:
        print('--------------------------------------')
        print('Searching optimal sigma for original scores...')
        print(f'Sigma range: {cfg.sigma_range}')
        print('--------------------------------------')
    
    org_best_auc, org_best_sigma, org_best_predicted = _search_best_sigma(predicted, labels, cfg)
    
    _print_results('Original Score Results', org_best_auc, org_best_sigma, grid_search=cfg.grid_search)
    
    graph_path = f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}/{model_name}_proposed_org_{cfg.dataset_name}_{cfg.type}_{cfg.prompt_type}.jpg'
    save_score_auc_graph(anomalies_idx, org_best_predicted, org_best_auc, graph_path)
    
    results['org_auc'] = org_best_auc
    results['org_predicted'] = org_best_predicted
    results['org_sigma'] = org_best_sigma
    
    '''
    ================================
    combination operation (proposed)
    ================================
    '''
    if cfg.grid_search:
        print('--------------------------------------')
        print('Starting grid search for optimal hyperparameters...')
        print('--------------------------------------')
        
        sigma_start, sigma_end = map(int, cfg.sigma_range.split(','))
        total_iterations = (sigma_end - sigma_start) * len(list(np.arange(0.0, 1.1, cfg.weight_step))) ** 2
        print(f'Total iterations: {total_iterations}')
        print(f'Sigma range: [{sigma_start}, {sigma_end})')
        print(f'Weight step: {cfg.weight_step}')
        print('-----------------------------------')
        
        best_auc = 0
        best_alpha = 0
        best_beta = 0
        best_gamma = 0
        best_sigma = 0
        best_predicted = None
        
        for sigma in progress_bar(range(sigma_start, sigma_end), total=(sigma_end - sigma_start)):
            for a1 in np.arange(0.0, 1.1, cfg.weight_step):
                for a2 in np.arange(0.0, 1.1, cfg.weight_step):
                    a3 = 1 - a1 - a2
                    if 0 <= a3 <= 1:
                        agg_predicted = a1 * predicted + a2 * predicted_wa + a3 * predicted_tc
                        auc, mm_predicted = _calculate_auc(agg_predicted, labels, sigma)
                        
                        if auc > best_auc:
                            best_auc = auc
                            best_predicted = mm_predicted
                            best_alpha = a1
                            best_beta = a2
                            best_gamma = a3
                            best_sigma = sigma
                            print(f'\nNew best found! AUC: {best_auc:.4f}, sigma: {best_sigma}, '
                                  f'weights: ({best_alpha:.2f}, {best_beta:.2f}, {best_gamma:.2f})')
        
        combi_best_auc = best_auc
        combi_best_predicted = best_predicted
        
        results['best_sigma'] = best_sigma
        results['best_alpha'] = best_alpha
        results['best_beta'] = best_beta
        results['best_gamma'] = best_gamma
        
    else:
        agg_predicted = cfg.alpha * predicted + cfg.beta * predicted_wa + cfg.gamma * predicted_tc
        combi_best_auc, _, combi_best_predicted = _search_best_sigma(agg_predicted, labels, cfg)
        
        results['best_sigma'] = cfg.sigma
        results['best_alpha'] = cfg.alpha
        results['best_beta'] = cfg.beta
        results['best_gamma'] = cfg.gamma
    
    _print_results(
        'Grid Search Results' if cfg.grid_search else 'Combination Results',
        combi_best_auc,
        results['best_sigma'],
        weights=(results['best_alpha'], results['best_beta'], results['best_gamma']),
        grid_search=cfg.grid_search
    )
    
    graph_path = f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}/{model_name}_proposed_combi_{cfg.dataset_name}_{cfg.type}_{cfg.prompt_type}.jpg'
    save_score_auc_graph(anomalies_idx, combi_best_predicted, combi_best_auc, graph_path)
    
    results['combi_best_auc'] = combi_best_auc
    results['combi_best_predicted'] = combi_best_predicted
    
    '''
    ================================
    per-video evaluation (optional)
    ================================
    '''
    if cfg.calc_video_auc and video_names is not None and label_arr is not None:
        print('--------------------------------------')
        print('save individual anomaly scores...')
        print('--------------------------------------')
        
        len_past = 0
        for i in progress_bar(range(len(label_arr)), total=len(label_arr)):
            video_name = video_names[i]
            video_gt = label_arr[i]
            len_present = len(label_arr[i])
            
            video_pd = combi_best_predicted[len_past:len_past + len_present]
            video_anomalies_idx = [j for j, l in enumerate(video_gt) if l == 1]
            graph_path = f'results/{cfg.dataset_name}/{cfg.type}/{cfg.prompt_type}/videos/{model_name}_proposed_combi_{cfg.dataset_name}_{video_name}_{cfg.type}_{cfg.prompt_type}.jpg'
            save_score_graph(video_anomalies_idx, video_pd, graph_path)
            
            len_past += len_present
    
    return results