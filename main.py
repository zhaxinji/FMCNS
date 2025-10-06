#!/usr/bin/env python
import argparse
import sys
import os
import platform
import torch
import ipdb
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logger import init_logger, getLogger
from params import Config
from dataHandler import RecDataset, TrainDataLoader, EvalDataLoader
from model import get_model, Trainer
from utils import init_seed, dict2str

def quick_start(model, dataset, config_dict, save_model=True, mg=False):
    config = Config(model, dataset, config_dict, mg)
    init_logger(config)
    logger = getLogger()

    dataset = RecDataset(config)
    train_dataset, valid_dataset, test_dataset = dataset.split()

    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    
    for hyper_tuple in combinators:
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('='*60)
        logger.info(f'Training Configuration {idx+1}/{total_loops}:')
        logger.info('='*60)
        
        logger.info('Hyperparameters being tuned:')
        for param, value in zip(config['hyper_parameters'], hyper_tuple):
            logger.info(f'  {param}: {value}')
        
        logger.info('')
        logger.info('Model Configuration:')
        key_params = [
            'model', 'dataset', 'embedding_size', 'feat_embed_dim', 'n_mm_layers', 'n_ui_layers',
            'knn_k', 'mm_image_weight', 'dropout', 'n_steps', 's_steps', 'time_embedding_size',
            'w', 'weight', 'sample_k', 'curriculum_start_epoch', 'curriculum_step', 
            'curriculum_end_epoch', 'learning_rate', 'train_batch_size', 'reg_weight'
        ]
        
        for param in key_params:
            if param in config and config[param] is not None:
                logger.info(f'  {param}: {config[param]}')
        
        logger.info('='*60)
        logger.info('')

        train_data.pretrain_setup()
        model = get_model(config['model'])(config, train_data).to(config['device'])

        trainer = Trainer(config, model, mg)
        best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid))

        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
            
        idx += 1

def main():
    parser = argparse.ArgumentParser(description='FMCNS: Flow Matching with Causal-aware Negative Sampling for Multimodal Recommendation')
    parser.add_argument('-m', '--model', type=str, default='FMCNS', help='Model name')
    parser.add_argument('-d', '--dataset', type=str, default='sports', help='Dataset name')
    parser.add_argument('--save_model', action='store_true', default=True, help='Whether to save the model')
    parser.add_argument('--mg', action='store_true', default=False, help='Multi-GPU training')
    
    args = parser.parse_args()
    
    config_dict = {}
    
    quick_start(
        model=args.model,
        dataset=args.dataset,
        config_dict=config_dict,
        save_model=args.save_model,
        mg=args.mg
    )

if __name__ == '__main__':
    main()