from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForQuestionAnswering, BertModel, BertConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Defining some key variables that will be used later on in the training
MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.keyword
        self.targets = self.data.real_label
        self.max_len = max_len
    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            # padding='longest',
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'pid': self.data.iloc[index]['pid'],
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

    def get_decode_item(self, index):
        comment_text = str(self.comment_text[index])
        return comment_text


import transformers

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.
class BERTClass(torch.nn.Module):
    def __init__(self, output_numbers=5):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-multilingual-cased',
                                                         problem_type='multi_label_classification', num_labels=output_numbers, output_attentions=True)  # 5
        self.config = self.l1.config
        # self.l1 = AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', problem_type='multi_label_classification', num_labels=5)
        self.l2 = torch.nn.Dropout(0.3)
        # self.l3 = torch.nn.Linear(768, 9)
        self.l3 = torch.nn.Linear(768, output_numbers)

    # def forward(self, x):
        # _, output_1 = self.l1(**x, return_dict=False)
    def forward(self, input_ids, attention_mask, token_type_ids):
        # _, output_1 = self.l1(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        _, output_1, _ = self.l1(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

class BERTClassTest(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-multilingual-cased',
                                                         problem_type='multi_label_classification', num_labels=5)  # 5

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        return output_1


class BERTbinaryClass(torch.nn.Module):
    def __init__(self):
        super(BERTbinaryClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-multilingual-cased',
                                                         problem_type='multi_label_classification', num_labels=2)  # 5
        self.config = self.l1.config

        # self.l1 = AutoModelForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', problem_type='multi_label_classification', num_labels=5)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 2)  # 5

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, output_1 = self.l1(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

class Custominference(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.keyword
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        try:
            comment_text = str(self.comment_text[index])
        except KeyError:
            comment_text = str(self.comment_text.values[0])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }
