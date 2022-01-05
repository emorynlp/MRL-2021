from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer

import util
from src.crf import ChainCRF
from data_process import LABEL_PAD_ID

class Tagger(nn.Module) :
    def __init__(self, hparams, num_labels) :
        super(Tagger, self).__init__()
        
        self.hparams = hparams
        ##! Dataset dependent
        self.num_labels = num_labels
        
        ##! modules
        # encoder module #!
        self.encoder = self.build_encoder()
        self.hidden_size = self.encoder.config.hidden_size
        self.freeze_encoder_layers()
        
        # crf or linear module
        if self.hparams.use_crf :
            self.crf = ChainCRF(self.hidden_size, self.num_labels)
        else :
            self.linear = nn.Linear(self.hidden_size, self.num_labels)
        
        # droupout
        self.dropout = nn.Dropout(self.hparams.dropout) # 0.2
        
    def build_encoder(self) :
        config = AutoConfig.from_pretrained(self.hparams.pretrain_model, output_hidden_states = True)
        encoder = AutoModel.from_pretrained(self.hparams.pretrain_model, config=config)
        return encoder
    
    def freeze_encoder_layers(self) :
        if self.hparams.freeze_layer == -1 :
            return
        elif self.hparams.freeze_layer >= 0 :
            for i in range(self.hparams.freeze_layer + 1) :
                if i == 0 :
                    print("freeze embeddings")
                    self.freeze_encoder_embeddings()
                else :
                    print(f'freeze layer {i}')
                    self.freeze_encoder_layer(i)
                    
    def freeze_encoder_embeddings(self) :
        for param in self.encoder.embeddings.parameters() :
            param.requires_grad = False
        # util.freeze(self.model.embeddings)
        
    def freeze_encoder_layer(self, layer) :
        for param in self.encoder.encoder.layer[layer - 1].parameters() :
            param.requires.grad = False
        # util.freeze(self.model.encoder.layer[layer - 1])
    
    def encode_sent(self, sent: Tensor, langs: Optional[List[str]], segment: Optional[Tensor] = None) :
        _, _, hidden_states = self.model(input_ids = sent, attention_mask = mask, token_type_ids = segment) # last_hidden_state, pooler_output, hidden_states
        hs = hidden_states[-1]
        return hs
    
    def forward(self, input_ids = None, attention_mask = None, token_type_ids = None, labels = None) :
        inputs = {'input_ids' : input_ids, 'attention_mask' : attention_mask, 'token_type_ids' : token_type_ids}
        outputs = self.encoder(**inputs) # last_hiddent_state, pooler_output, hidden_states
        last_hidden_state = outputs[0]
        
        if self.hparams.use_crf :
            pass
        else :
            logits = self.linear(self.dropout(last_hidden_state)) # (B, seq_len, hidden_size) -> (B, seq_len, num_labels)
            log_probs = F.log_softmax(logits, dim = -1) # score -> prob -> lob
            loss = F.nll_loss( # input, target, ignore_nidex
                log_probs.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index= LABEL_PAD_ID
            )
        return loss, log_probs
            