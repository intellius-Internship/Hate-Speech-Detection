import random
import torch
import argparse
import logging
import warnings

import numpy as np
import transformers
import pytorch_lightning as pl

from plm import LightningPLM
from eval import evaluation

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


warnings.filterwarnings(action='ignore')
transformers.logging.set_verbosity_error()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

SEED = 19

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Hate Speech Detection based on PLM')
    parser.add_argument('--train',
                        action='store_true',
                        default=False,
                        help='for training')

    parser.add_argument('--data_dir',
                        type=str,
                        default='data')

    parser.add_argument('--save_dir',
                        type=str,
                        default='result')

    parser.add_argument('--model_name',
                        type=str,
                        default='baseline')

    parser.add_argument('--model_type',
                        type=str,
                        default='bert')

    parser.add_argument('--num_labels',
                        type=int,
                        default=2)

    parser.add_argument('--max_len',
                        type=int,
                        default=128)

    parser.add_argument('--model_pt',
                        type=str,
                        default='baseline-last.ckpt')
            
    parser.add_argument("--gpuid", nargs='+', type=int, default=0)

    parser.add_argument('--user_input',
                        action='store_true',
                        default=False)

    parser = LightningPLM.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    set_seed(SEED)

    global DATA_DIR
    DATA_DIR = args.data_dir

    if args.train:
        checkpoint_callback = ModelCheckpoint(
            dirpath='model_ckpt',
            filename='{epoch:02d}-{avg_val_acc:.2f}',
            verbose=True,
            save_last=True,
            monitor='avg_val_acc',
            mode='max',
            prefix=f'{args.model_name}'
        )
        model = LightningPLM(args)
        model.train()
        trainer = Trainer(
                        check_val_every_n_epoch=1, 
                        checkpoint_callback=checkpoint_callback, 
                        flush_logs_every_n_steps=100, 
                        gpus=args.gpuid, 
                        gradient_clip_val=1.0, 
                        log_every_n_steps=50, 
                        logger=True, 
                        max_epochs=args.max_epochs,
                        num_processes=1,
                        accelerator='ddp' if args.model_type in ['bert', 'electra'] else None)
        
        trainer.fit(model)
        logging.info('best model path {}'.format(checkpoint_callback.best_model_path))

    else:
        with torch.cuda.device(args.gpuid[0]):
            evaluation(args)