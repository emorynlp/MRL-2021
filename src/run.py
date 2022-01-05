import util
from data_process import DataProcessor, LABEL_PAD_ID
from model import Tagger

import os
from os.path import join
from argparse import ArgumentParser
import logging
from datetime import datetime
import time
from pytz import timezone
import numpy as np
import pickle
import json

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule

from seqeval.metrics import classification_report
from seqeval.metrics.sequence_labeling import precision_recall_fscore_support


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()

class Runner :
    def __init__(self, hparams) :
        self.hparams = hparams
        
        ##! make ckpt, log dirs
        self.name_suffix = datetime.now(timezone('Asia/Seoul')).strftime('%b%d_%H-%M-%S')
        self.ckpt_exp_time_dir = join(hparams.ckpt_dir, hparams.exp_name, self.name_suffix)
        os.makedirs(self.ckpt_exp_time_dir, exist_ok=True)
        self.log_exp_dir = join(hparams.log_dir, hparams.exp_name)
        os.makedirs(self.log_exp_dir, exist_ok=True)
        
        ##! Set up logger
        log_path = join(self.log_exp_dir, 'log_' + self.name_suffix +'.txt')
        logger.addHandler(logging.FileHandler(log_path, 'a'))
        logger.info('Log file path: %s' % log_path)
        logger.info('=====================Initialization=====================')
        logger.info(f'Parameters :\n{hparams}')
        
        ##! Set up seed
        if hparams.seed :
            util.set_seed(hparams.seed)
        
        ##! Set up device
        self.device = torch.device('cpu' if hparams.gpu_id is None else f'cuda:{hparams.gpu_id}')
        logger.info(f'Set device to cuda:{hparams.gpu_id}')
        
        ##! Set up data
        self.data_processor = DataProcessor(hparams) 
        logger.info(f'Initilized data_processor')
        
        ##! Set up model
        self.model = Tagger(hparams, self.data_processor.num_labels) 
        logger.info(f'Initilized model to {hparams.pretrain_model}')

    def get_optimizer(self, model) :
        no_decay = ['bias', 'LayerNorm.weight']
        grouped_param = [
            {
                'params' : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], # not a and not b <=> not (a or b)
                'weight_decay' : self.hparams.adam_weight_decay
            },
            {
                'params' : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay' : 0.0
            }
        ]
        optimizer = AdamW(grouped_param, lr=self.hparams.learning_rate, eps= self.hparams.adam_eps)
        return optimizer
    
    def get_scheduler(self, optimizer, total_update_steps) :
        warmup_steps = int(total_update_steps) * self.hparams.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps,
                                                    num_training_steps = total_update_steps)
        return scheduler
    
    def prepare_inputs(self, batch, with_labels = True) :
        inputs = {
            'input_ids' : batch[0],
            'attention_mask' : batch[1],
            'token_type_ids' : batch[2]
        }
        if with_labels :
            inputs['labels'] = batch[-1]
        
        return inputs
    
    def run(self) :
        # Train the model with data, optimizer, scheduler
        logger.info('Train')
        epochs = self.hparams.max_epochs
        batch_size = self.hparams.batch_size
        grad_accum = self.hparams.gradient_accumulation_steps
        
        ##! Set up data
        train_data = self.data_processor.get_data(self.hparams.data_dir, 'en', 'train')
        dev_data = self.data_processor.get_data(self.hparams.data_dir, 'en', 'dev')
        test_data = self.data_processor.get_data(self.hparams.data_dir, 'ko', 'test')
        
        train_dataloader = DataLoader(train_data,
                                      sampler = RandomSampler(train_data),
                                      batch_size = batch_size,
                                      drop_last = False,
                                      pin_memory = True,
                                      num_workers = 1)
        ##! Set up model
        self.model.to(self.device)
        
        ##! Set up optimizer and schedular
        total_update_steps = len(train_dataloader) * epochs // grad_accum
        optimizer = self.get_optimizer(self.model)
        scheduler = self.get_scheduler(optimizer, total_update_steps)
        trained_params = self.model.parameters() # clip gradient norm
        
        ##! Start training
        logger.info('=====================Train=====================')
        logger.info(f'Batch size : {self.hparams.batch_size}, Learning rate : {self.hparams.learning_rate}, Max epochs : {self.hparams.max_epochs}')
        logger.info(f'Freeze layer : {self.hparams.freeze_layer}')
        # logger.info(F'Num examples : {}')
        logger.info(f'Num features : {len(train_data)}')
        logger.info(f'Total steps : {total_update_steps}')

        loss_during_accum = [] # To compute effective loss at each update
        loss_during_report = 0.0 # Effective loss during logging step
        loss_history = [] # Full history of effective loss; length equals total update steps
        max_f1 = 0
        start_time = time.time()
        self.model.zero_grad()
        
        ##! Epoch
        for epoch in range(epochs) :
            logger.info(f'========== EPOCH {epoch} ==========')
            #! results
            logger.info(f'Epoch {epoch} : adding epoch result at results.jsonl...')
            # get dev_en_loss, dev_en_metrics
            dev_en_loss, dev_en_metrics = self.evaluate(self.model, dev_data, len(loss_history), dropout = False)
            # get test_ko_loss, test_ko_metricss
            test_ko_loss, test_ko_metrics = self.evaluate(self.model, test_data, len(loss_history), dropout = False)
            # add to results.jsonl
            result = {'dev_en_loss' : dev_en_loss,
                      'dev_en_f1' : dev_en_metrics['f1'],
                      'test_ko_loss' : test_ko_loss,
                      'test_ko_f1' : test_ko_metrics['f1'],
                      'step' : len(loss_history)}
            results_path = join(self.ckpt_exp_time_dir, 'results.jsonl')
            with open(results_path, 'a', encoding = 'utf-8') as f :
                f.write(json.dumps(result) + '\n')
                
            
            ##! Batch
            for batch in train_dataloader : # prepare inputs - forward - backward - update
                self.model.train()
                inputs = self.prepare_inputs(batch, with_labels = True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                loss, _ = self.model(**inputs) # loss, log_probs
                if grad_accum > 1 :
                    loss /= grad_accum
                loss.backward()
                loss_during_accum.append(loss.item())
                
                ##! Update after gradient accumulation
                if len(loss_during_accum) % grad_accum == 0 :
                    if self.hparams.max_grad_norm :
                        torch.nn.utils.clip_grad_norm_(trained_params, self.hparams.max_grad_norm) # max_grad_norm : 1
                    optimizer.step()
                    self.model.zero_grad()
                    scheduler.step()
                    
                    # Compute effective loss
                    effective_loss = np.sum(loss_during_accum).item()
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)
                    
                    ##! Report if True
                    if len(loss_history) % self.hparams.report_frequency == 0 :
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / self.hparams.report_frequency
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info(f'Step : {len(loss_history)}, avg loss : {avg_loss:.3f}, steps/sec : {self.hparams.report_frequency / (end_time - start_time):.2f}')
                        start_time = end_time
                        

                        
                    ##! Evaluate if True
                    if len(loss_history) > 0 and len(loss_history) % self.hparams.eval_frequency == 0 :
                        pass
        
        logger.info(f'===== Finished training =====')
        logger.info(f'Actual update steps : {len(loss_history)}')
        
        #! results
        logger.info(f'Epoch {epoch} : adding epoch result at results.jsonl...')
        # get dev_en_loss, dev_en_metrics
        dev_en_loss, dev_en_metrics = self.evaluate(self.model, dev_data, len(loss_history), dropout = False)
        # get test_ko_loss, test_ko_metricss
        test_ko_loss, test_ko_metrics = self.evaluate(self.model, test_data, len(loss_history), dropout = False)
        # add to results.jsonl
        result = {'dev_en_loss' : dev_en_loss,
                    'dev_en_f1' : dev_en_metrics['f1'],
                    'test_ko_loss' : test_ko_loss,
                    'test_ko_f1' : test_ko_metrics['f1'],
                    'step' : len(loss_history)}
        results_path = join(self.ckpt_exp_time_dir, 'results.jsonl')
        with open(results_path, 'a', encoding = 'utf-8') as f :
            f.write(json.dumps(result) + '\n')
        
        
        
        
    
    def evaluate(self, model, dataset, step = 0, dropout = False) :
        '''
        Given (dev or test) data,
        return loss and metrics(f1, precision, recall)
        '''
        hparams = self.hparams
        dataloader = DataLoader(dataset, sampler = SequentialSampler(dataset), batch_size = hparams.eval_batch_size) #! SequentialSampler
        
        loss_history = []
        # Get loss & logits (not masked) and labels (id)
        model.eval()
        if dropout :
            model.train()
        model.to(self.device)
        
        all_logits, all_labels = [], []
        for batch in dataloader :
            inputs = self.prepare_inputs(batch, with_labels = True) #!
            inputs = {k : v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad() :
                loss, logits = model(**inputs)
                
            loss_history.append(loss.item())
            
            all_logits.append(logits.detach().cpu()) # B, max_len, num_labels
            all_labels.append(inputs['labels'].detach().cpu()) # B, max_len
        
        # Get avg loss #
        avg_loss = np.sum(loss_history) / len(loss_history)
        
        all_logits = torch.cat(all_logits, dim = 0).numpy() # num featuers * max_len * num_labels
        all_labels = torch.cat(all_labels, dim = 0).numpy() # num featuers * max_len
        
        # Get predictions and golds
        label_map = self.data_processor.id2label
        
        masked_all_logits, predictions, golds = [], [], []
        for i in range(all_logits.shape[0]) :
            mask = all_labels[i] != LABEL_PAD_ID # max_len ; True or False
            masked_all_logits.append(all_logits[i][mask]) # mask_len * num_labels
            
            prediction = [label_map[label_id] for label_id in masked_all_logits[-1].argmax(axis = -1).tolist()]
            predictions.append(prediction)
            
            gold = [label_map[label_id] for label_id in all_labels[i][mask].tolist()]
            golds.append(gold)
        
        # Get metrics #
        p, r, f, _ = precision_recall_fscore_support(golds, predictions, beta=1, average = 'micro') # Weighted average
        metrics = {'f1' : f, 'precision' : p, 'recall' : r}
        
        return avg_loss, metrics
        
        
        
if __name__ == '__main__' :
    ##! add argument
    parser = ArgumentParser()
    #! 주요 디렉토리 변수 설정
    parser.add_argument("--data_dir", required = True, type=str)
    parser.add_argument("--ckpt_dir", required = True, type=str)
    parser.add_argument("--log_dir", required = True, type=str)
    parser.add_argument("--exp_name", required = True, type=str)
  
    #! misc
    parser.add_argument("--seed", default = 42, type=int)
    parser.add_argument("--gpu_id", default = 0, type=int)
    
    #! data
    parser.add_argument("--max_len", default=512, type=int)
    parser.add_argument('--train_langs', nargs="+", type=str)
    parser.add_argument('--dev_langs', nargs="+", type=str)
    parser.add_argument('--test_langs', nargs="+", type=str)

  
    #! model
    # pretrain
    parser.add_argument("--pretrain_model", required = True, type=str)
    # freeze_layer
    parser.add_argument("--freeze_layer", required = True, type=int)
    # dropout
    parser.add_argument("--dropout", required = True, type=float)
    # use_crf
    parser.add_argument("--use_crf", required = True, type=util.str2bool)
    
    #! train
    parser.add_argument('--batch_size', required = True, type=int)
    parser.add_argument('--learning_rate', required = True, type=float)
    parser.add_argument('--max_epochs', required = True, type=int)
    parser.add_argument('--gradient_accumulation_steps', default = 1, type = int)
    parser.add_argument('--adam_weight_decay', default = 1e-4, type=float)
    parser.add_argument('--adam_eps', default = 1e-8, type = float)
    parser.add_argument('--warmup_ratio', default = 0.1, type=float)
    parser.add_argument('--max_grad_norm', default=1, type=float)
    
    #! report & eval
    parser.add_argument('--report_eval_every_epoch', required=True, type=util.str2bool)
    parser.add_argument('--report_frequency' , default = 100, type = int)
    parser.add_argument('--eval_frequency', default = 700, type = int) #! change default value to 700 later
    parser.add_argument('--eval_batch_size', default = 128, type = int)

    ##! parse arguments
    hparams = parser.parse_args()

    ##! Runner 객체 생성
    runner = Runner(hparams)
    
    ##! run
    runner.run()
    