import torch
import KoBart
import dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data file path
dev_path = 'data/dev.tsv'

# config
batch_size = 16

# model, tokenizer init
model = KoBart.KoBARTConditionalGeneration().to(device)
model.load_state_dict(torch.load('output/kobart_step_36000.pth'))
tokenizer = model.tokenizer

# dataset
dev_dataset = dataset.KoBARTQGDataset(dev_path, tokenizer)
dev_dataloader = DataLoader(dev_dataset, batch_size)

with open('output/output_36000.txt', 'w', encoding='utf-8') as f:
    for step_index, batch_data in tqdm( enumerate(dev_dataloader), f"[GENERATE]", total=len(dev_dataloader)):

        input_ids, decoder_input_ids, labels = tuple(value.to(device) for value in batch_data.values())

        output = model.model.generate(input_ids=input_ids, eos_token_id=tokenizer.eos_token_id, max_length=100, num_beams=5)

        for o in output:
            output = tokenizer.decode(o, skip_special_tokens=True)
            f.write(output+'\n')
