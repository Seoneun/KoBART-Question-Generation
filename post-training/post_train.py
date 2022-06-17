import logging
import os
import KoBart
import dataset
import post_dataset
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
import math

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Config
batch_size = 8
epochs = 10
warmup_ratio = 0.1
learning_rate = 3e-5
grad_clip = 1.0
train_log_interval = 100
validation_interval = 8000
save_interval = 8000

# Model, tokenizer init
model = KoBart.KoBARTConditionalGeneration()
tokenizer = model.tokenizer

# Data file path
#post_train_path = 'data/post_train.tsv'
train_path = 'data/train.tsv'
dev_path = 'data/dev.tsv'

# dataset, dataloader
# pretrain(infilling)
train_dataset = post_dataset.KoBARTQGPostDataset(train_path, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size)

dev_dataset = dataset.KoBARTQGDataset(dev_path, tokenizer)
dev_dataloader = DataLoader(dev_dataset, batch_size)

# QG
post_dataset = post_dataset.KoBARTQGDataset(train_path, tokenizer)
post_dataloader = DataLoader(post_dataset, batch_size)

# optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, correct_bias=False)

# scheduler
data_len = len(train_dataloader)
num_train_steps = int(data_len / batch_size * epochs)
num_warmup_steps = int(num_train_steps * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

# logging data info
logging.info(f'data length {data_len}')
logging.info(f'num_train_steps : {num_train_steps}')
logging.info(f'num_warmup_steps : {num_warmup_steps}')

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# dev
def _validate(
        model: KoBart.KoBARTConditionalGeneration,
        dev_dataloader: DataLoader,
        device: torch.device,
        logger: logging.Logger,
        global_step: int,
):
    model.eval()
    loss_list = []
    for batch_data in tqdm(dev_dataloader, desc="[EVAL]"):
        with torch.no_grad():
            input_ids, decoder_input_ids, labels = tuple(value.to(device) for value in batch_data.values())
            model_outputs = model.forward(input_ids, decoder_input_ids, labels)
            loss_list.append(model_outputs.loss.item())

    mean_loss = np.mean(loss_list)
    logger.info(f"[EVAL] global_step:{global_step} loss:{mean_loss:.4f} perplexity:{math.exp(mean_loss):.4f}")
    model.train()

    return mean_loss

model.train()
loss_list_between_log_interval = []

least_dev_loss = 999
for epoch_id in range(epochs):
    for step_index, batch_data in tqdm(enumerate(zip(train_dataloader, post_dataloader)), f"[TRAIN] EP:{epoch_id}", total=min(len(train_dataloader), len(post_dataloader))):
            global_step = len(train_dataloader) * epoch_id + step_index + 1
            optimizer.zero_grad()

            input_ids_1, decoder_input_ids_1, labels_1 = tuple(value.to(device) for value in batch_data[0].values())
            input_ids_2, decoder_input_ids_2, labels_2 = tuple(value.to(device) for value in batch_data[1].values())

            model_outputs1 = model.forward(input_ids_1, decoder_input_ids_1, labels_1)
            model_outputs2 = model.forward(input_ids_2, decoder_input_ids_2, labels_2)

            loss = model_outputs1.loss + model_outputs2.loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            # for logging
            loss_list_between_log_interval.append(loss.item())

            if global_step % train_log_interval == 0:
                mean_loss = np.mean(loss_list_between_log_interval)
                logger.info(
                    f"EP:{epoch_id} global_step:{global_step} "
                    f"loss:{mean_loss:.4f} perplexity:{math.exp(mean_loss):.4f}"
                )
                loss_list_between_log_interval.clear()

            if global_step % validation_interval == 0:
                dev_loss = _validate(model, dev_dataloader, device, logger, global_step)
                state_dict = model.state_dict()
                if dev_loss.item() < least_dev_loss:
                    least_dev_loss = dev_loss.item()
                    model_path = os.path.join('output_post2', f"kobart_best.pth")
                    logger.info(f"Save best model")
                    torch.save(state_dict, model_path)

            if global_step % save_interval == 0:
                state_dict = model.state_dict()
                model_path = os.path.join('output_post2', f"kobart_step_{global_step}.pth")
                logger.info(f"global_step: {global_step} model saved at {model_path}")
                torch.save(state_dict, model_path)
