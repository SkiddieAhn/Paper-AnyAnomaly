import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if not os.path.exists('results'):
    os.mkdir('results')

share_config = {'data_root': '/home/sha/datasets',
                'cdata_root': '/home/sha/datasets/cvad_data',
                'model_path': 'LVLM/weights/chatunivi'} 

class dict2class:
    def __init__(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)

    def print_cfg(self):
        print('\n' + '-' * 30 + f'configutation' + '-' * 30)
        for k, v in vars(self).items():
            print(f'{k}: {v}')
        print()


def update_config(args=None):
    share_config['dataset_name'] = args.dataset
    share_config['json_path'] = share_config['cdata_root'] + f'/c-{args.dataset}.json'
    share_config['m_json_path'] = share_config['cdata_root'] + f'/c-{args.dataset}-multiple.json'
    share_config['type'] = args.type
    share_config['multiple'] = args.multiple
    share_config['prompt_type'] = args.prompt_type
    share_config['anomaly_detect'] = args.anomaly_detect
    share_config['calc_auc'] = args.calc_auc
    share_config['calc_video_auc'] = args.calc_video_auc
    share_config['class_adaption'] = args.class_adaption
    share_config['template_adaption'] = args.template_adaption

    if args.clip_length != None:
        share_config['clip_length'] = args.clip_length

    if share_config['multiple']:
        share_config['test_data_path'] = os.path.join(share_config['cdata_root'], 'c-' + share_config['dataset_name']) + '/multiple/' + share_config['type']
    elif share_config['dataset_name'] == 'avenue': 
        share_config['test_data_path'] = os.path.join(share_config['data_root'], 'avenue') + '/testing/frames'
        share_config['type_list'] = ["too_close", "bicycle", "throwing", "running", "dancing"]
    elif share_config['dataset_name'] == 'shtech': 
        share_config['test_data_path'] = os.path.join(share_config['data_root'], 'shanghai') + '/testing'
        share_config['type_list'] = ["car", "bicycle", "fighting", "throwing", "hand_truck", "running", "skateboarding", "falling", "jumping", "loitering", "motorcycle"]

    if args.type != None:
        type_ids = {}
        for i, type in enumerate(share_config['type_list']):
            type_ids[str(type)] = i
        share_config['type_id'] = type_ids[args.type]  

    return dict2class(share_config)