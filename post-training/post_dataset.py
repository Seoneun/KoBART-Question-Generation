import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class KoBARTQGPostDataset(Dataset):
    def __init__(self, file, tokenizer, max_len = 512, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = pd.read_csv(file, sep='\t', encoding='cp949')
        self.len = self.docs.shape[0]

        self.pad_index = self.tokenizer.pad_token_id
        self.ignore_index = ignore_index
        self.mask_index = self.tokenizer.mask_token_id

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def text_infiling(self, inputs):
        token_length = len(inputs)
        mask_length = inputs.count(self.mask_index)

        while (mask_length / token_length) < 0.3:
            poisson_num = np.random.poisson(3, 1).item()

            if poisson_num == 0:
                index = int(np.random.uniform(low=1, high=len(inputs) - 2))
                inputs.insert(index, self.mask_index)
            else:
                start_index = int(np.random.uniform(low=1, high=len(inputs) - 2 - poisson_num))
                inputs[start_index] = self.mask_index

                for i in range(start_index + 1, start_index + poisson_num):
                    del inputs[start_index + 1]

            mask_length += poisson_num

        return inputs

    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        input_ids = self.tokenizer.encode(instance['content'])
        input_ids = self.text_infiling(input_ids)
        input_ids.append(self.tokenizer.eos_token_id)
        input_ids = self.add_padding_data(input_ids)

        label_ids = self.tokenizer.encode(instance['content'])
        label_ids += [self.tokenizer.eos_token_id]
        dec_input_ids = [self.tokenizer.bos_token_id]
        dec_input_ids += label_ids
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)

        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)}

    def __len__(self):
        return self.len

class KoBARTQGDataset(Dataset):
    def __init__(self, file, tokenizer, max_len = 512, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = pd.read_csv(file, sep='\t', encoding='utf-8')
        self.len = self.docs.shape[0]

        self.pad_index = self.tokenizer.pad_token_id
        self.ignore_index = ignore_index

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def text_infiling(self, inputs):
        token_length = len(inputs)
        mask_length = inputs.count(self.mask_index)

        while (mask_length / token_length) < 0.3:
            poisson_num = np.random.poisson(3, 1).item()

            if poisson_num == 0:
                index = int(np.random.uniform(low=1, high=len(inputs) - 2))
                inputs.insert(index, self.mask_index)
            else:
                start_index = int(np.random.uniform(low=1, high=len(inputs) - 2 - poisson_num))
                inputs[start_index] = self.mask_index

                for i in range(start_index + 1, start_index + poisson_num):
                    del inputs[start_index + 1]

            mask_length += poisson_num

        return inputs

    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        input_ids = self.tokenizer.encode(instance['content'])
        input_ids = self.add_padding_data(input_ids)

        label_ids = self.tokenizer.encode(instance['question'])
        label_ids.append(self.tokenizer.eos_token_id)
        dec_input_ids = [self.tokenizer.eos_token_id]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)

        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)}

    def __len__(self):
        return self.len

