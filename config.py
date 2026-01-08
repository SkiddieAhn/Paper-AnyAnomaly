import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

if not os.path.exists('results'):
    os.mkdir('results')

share_config = {'data_root': '/home/anonymous/datasets',
                'cdata_root': 'ground_truth'
                } 

class dict2class:
    def __init__(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)

    def print_cfg(self):
        print('\n' + '-' * 30 + f'configutation' + '-' * 30)
        for k, v in vars(self).items():
            print(f'{k}: {v}')
        print()


def calculate_patches(image_size, kernel_size, stride_size):
    rows = (image_size[0] - kernel_size[0]) // stride_size[0] + 1
    columns = (image_size[1] - kernel_size[1]) // stride_size[1] + 1
    return rows, columns


def update_config(args=None):
    share_config['model_path'] = args.model_path
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
    share_config['kfs_num'] = args.kfs_num
    share_config['img_size'] = (240, 240)
    share_config['sml_scale'] = args.sml_scale
    share_config['mid_scale'] = args.mid_scale
    share_config['lge_scale'] = args.lge_scale
    share_config['stride'] = args.stride

    # for evaluation
    share_config['grid_search'] = args.grid_search
    share_config['sigma_range'] = args.sigma_range
    share_config['weight_step'] = args.weight_step
    share_config['sigma'] = args.sigma
    share_config['alpha'] = args.alpha
    share_config['beta'] = args.beta
    share_config['gamma'] = args.gamma

    if args.clip_length != None:
        share_config['clip_length'] = args.clip_length

    if share_config['dataset_name'] == 'avenue': 
        share_config['test_data_path'] = os.path.join(share_config['data_root'], 'avenue') + '/testing/frames'
        share_config['type_list'] = ["too_close", "bicycle", "throwing", "running", "dancing"]
        share_config['out_prompt'] = 'Output'
        share_config['lge_size'] = (240, 240)
        share_config['mid_size'] = (80, 80)
        share_config['sml_size'] = (48, 48)
        if share_config['stride']:
            share_config['lge_size_stride'] = (240, 240)
            share_config['mid_size_stride'] = (40, 40)
            share_config['sml_size_stride'] = (24, 24)
        else:
            share_config['lge_size_stride'] = (240, 240)
            share_config['mid_size_stride'] = (80, 80)
            share_config['sml_size_stride'] = (48, 48)

    elif share_config['dataset_name'] == 'shtech':
        share_config['test_data_path'] = os.path.join(share_config['data_root'], 'shanghai') + '/testing'
        share_config['type_list'] = ["car", "bicycle", "fighting", "throwing", "hand_truck", "running", "skateboarding", "falling", "jumping", "loitering", "motorcycle"]
        share_config['out_prompt'] = 'Response'
        share_config['lge_size'] = (120, 120)
        share_config['mid_size'] = (80, 80)
        share_config['sml_size'] = (48, 48)
        if share_config['stride']:
            share_config['lge_size_stride'] = (60, 60)
            share_config['mid_size_stride'] = (40, 40)
            share_config['sml_size_stride'] = (24, 24)
        else:
            share_config['lge_size_stride'] = (120, 120)
            share_config['mid_size_stride'] = (80, 80)
            share_config['sml_size_stride'] = (48, 48)

    share_config['lge_patch_num'] = calculate_patches(share_config['img_size'], share_config['lge_size'], share_config['lge_size_stride'])
    share_config['mid_patch_num'] = calculate_patches(share_config['img_size'], share_config['mid_size'], share_config['mid_size_stride'])
    share_config['sml_patch_num'] = calculate_patches(share_config['img_size'], share_config['sml_size'], share_config['sml_size_stride'])

    if args.type != None:
        type_ids = {}
        for i, type in enumerate(share_config['type_list']):
            type_ids[str(type)] = i

        if args.multiple:
            types = args.type.split('-')
            share_config['type_ids'] = []
            length = len(type_ids)
            for type in types:
                if not type in share_config['type_list']:
                    share_config['type_list'].append(type)
                    type_ids[str(type)] = length
                    length += 1
                share_config['type_ids'].append(type_ids[type])
        else:
            if not args.type in share_config['type_list']:
                share_config['type_list'].append(args.type)
                type_ids[args.type] = len(type_ids)
            share_config['type_ids'] = [type_ids[args.type]] 

    return dict2class(share_config)