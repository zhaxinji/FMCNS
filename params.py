import re
import os
import yaml
import torch
from logging import getLogger

class Config(object):
    def __init__(self, model=None, dataset=None, config_dict=None, mg=False):
        if config_dict is None:
            config_dict = {}
        config_dict['model'] = model
        config_dict['dataset'] = dataset
        self.final_config_dict = self._load_dataset_model_config(config_dict, mg)
        self.final_config_dict.update(config_dict)
        self._set_default_parameters()
        self._init_device()

    def _load_dataset_model_config(self, config_dict, mg):
        file_config_dict = dict()
        file_list = []
        cur_dir = os.getcwd()
        
        overall_config = {
            'gpu_id': 0,
            'use_gpu': True,
            'seed': [999],
            'data_path': './data/',
            'inter_splitting_label': 'x_label',
            'filter_out_cod_start_users': True,
            'is_multimodal_model': True,
            'checkpoint_dir': 'saved',
            'save_recommended_topk': True,
            'recommend_topk': 'recommend_topk/',
            'embedding_size': 64,
            'weight_decay': 0.0,
            'req_training': True,
            'epochs': 1000,
            'stopping_step': 20,
            'train_batch_size': 2048,
            'learner': 'adam',
            'learning_rate': 0.001,
            'learning_rate_scheduler': [1.0, 50],
            'eval_step': 1,
            'training_neg_sample_num': 1,
            'use_neg_sampling': True,
            'use_full_sampling': False,
            'NEG_PREFIX': 'neg__',
            'USER_ID_FIELD': 'user_id:token',
            'ITEM_ID_FIELD': 'item_id:token',
            'TIME_FIELD': 'timestamp:float',
            'field_separator': "\t",
            'metrics': ["Recall", "NDCG", "Precision", "MAP"],
            'topk': [5, 10, 20, 50],
            'valid_metric': 'Recall@20',
            'eval_batch_size': 4096,
            'max_txt_len': 32,
            'max_img_size': 256,
            'vocab_size': 30522,
            'type_vocab_size': 2,
            'hidden_size': 64,
            'pad_token_id': 0,
            'max_position_embeddings': 512,
            'layer_norm_eps': 1e-12,
            'hidden_dropout_prob': 0.1,
            'end2end': False,
            'hyper_parameters': ["seed"],
            'use_neighborhood_loss': False,
            'clip_grad_norm': None,
            'eval_type': 'full_sort',
            'state': None,
            'flow_lr': 0.001,
            'alpha1': 1.0,
            'alpha2': 1.0,
            'beta': 10,
            'c': 0.1,
            'cf_model': 'lightgcn',
            'degree_ratio': 0.1,
            'aug_weight': 0.1,
        }
        
        file_config_dict.update(overall_config)
        
        if config_dict['model'] == 'FMCNS':
            model_config = {
                'train_batch_size': 512,
                'learning_rate': 0.0005,
                'embedding_size': 64,
                'feat_embed_dim': 64,
                'weight_size': [64, 64],
                'lambda_coeff': 0.9,
                'reg_weight': 0.0,
                'n_mm_layers': 1,
                'n_ui_layers': 3,
                'knn_k': 10,
                'mm_image_weight': 0.1,
                'dropout': 0.8,
                'flow_weight': 0.5,
                'n_steps': 10,
                's_steps': 2,
                'time_embedding_size': 64,
                'w': [0.4, 0.5, 0.6, 0.7],
                'weight': 0.8,
                'sample_k': 0.1,
                'intervention_beta': 3.0,
                'causal_reg_weight': 0.1,
                'causal_balance_factor': 0.5,
                'hyper_parameters': ["w"]
            }
            file_config_dict.update(model_config)
        
        if config_dict['dataset'] == 'baby':
            dataset_config = {
                'USER_ID_FIELD': 'userID',
                'ITEM_ID_FIELD': 'itemID',
                'TIME_FIELD': 'timestamp',
                'filter_out_cod_start_users': True,
                'inter_file_name': 'baby.inter',
                'vision_feature_file': 'image_feat.npy',
                'text_feature_file': 'text_feat.npy',
                'user_graph_dict_file': 'user_graph_dict.npy',
                'field_separator': "\t"
            }
            file_config_dict.update(dataset_config)
        elif config_dict['dataset'] == 'sports':
            dataset_config = {
                'USER_ID_FIELD': 'userID',
                'ITEM_ID_FIELD': 'itemID',
                'TIME_FIELD': 'timestamp',
                'filter_out_cod_start_users': True,
                'inter_file_name': 'sports.inter',
                'vision_feature_file': 'image_feat.npy',
                'text_feature_file': 'text_feat.npy',
                'user_graph_dict_file': 'user_graph_dict.npy',
                'field_separator': "\t"
            }
            file_config_dict.update(dataset_config)
        elif config_dict['dataset'] == 'clothing':
            dataset_config = {
                'USER_ID_FIELD': 'userID',
                'ITEM_ID_FIELD': 'itemID',
                'TIME_FIELD': 'timestamp',
                'filter_out_cod_start_users': True,
                'inter_file_name': 'clothing.inter',
                'vision_feature_file': 'image_feat.npy',
                'text_feature_file': 'text_feat.npy',
                'user_graph_dict_file': 'user_graph_dict.npy',
                'field_separator': "\t"
            }
            file_config_dict.update(dataset_config)
        
        if mg:
            mg_config = {
                'alpha1': 1.0,
                'alpha2': 1.0,
                'beta': 10
            }
            file_config_dict.update(mg_config)

        hyper_parameters = []
        if file_config_dict.get('hyper_parameters'):
            hyper_parameters.extend(file_config_dict['hyper_parameters'])
            
        file_config_dict['hyper_parameters'] = hyper_parameters
        return file_config_dict

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        return loader

    def _set_default_parameters(self):
        smaller_metric = ['rmse', 'mae', 'logloss']
        valid_metric = self.final_config_dict['valid_metric'].split('@')[0]
        self.final_config_dict['valid_metric_bigger'] = False if valid_metric in smaller_metric else True
        if "seed" not in self.final_config_dict['hyper_parameters']:
            self.final_config_dict['hyper_parameters'] += ['seed']

    def _init_device(self):
        use_gpu = self.final_config_dict['use_gpu']
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config_dict['gpu_id'])
        self.final_config_dict['device'] = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getitem__(self, item):
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict

    def __str__(self):
        args_info = '\n'
        args_info += '\n'.join(["{}={}".format(arg, value) for arg, value in self.final_config_dict.items() ])
        args_info += '\n\n'
        return args_info

    def __repr__(self):
        return self.__str__()