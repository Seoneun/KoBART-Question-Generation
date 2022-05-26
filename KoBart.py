import torch.nn as nn
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

class KoBARTConditionalGeneration(nn.Module):
    def __init__(self, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v1').to(device)
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.sep_token = '<unused0>'
        self.device = device

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
        self.pad_token_id = self.tokenizer.pad_token_id

    def forward(self, input_ids, decoder_input_ids, labels):
        attention_mask = input_ids.ne(self.pad_token_id).float()
        decoder_attention_mask = decoder_input_ids.ne(self.pad_token_id).float()

        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          decoder_input_ids=decoder_input_ids,
                          decoder_attention_mask=decoder_attention_mask,
                          labels=labels, return_dict=True)
