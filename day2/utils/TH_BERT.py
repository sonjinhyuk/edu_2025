from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch
from transformers import BertTokenizer, BertForQuestionAnswering, BertModel, BertConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
from tqdm.notebook import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, auc, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
MAX_LEN = 512
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

class CustomDataset(Dataset):
    def __init__(self, indexlist, dataframe, real_label, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.indexlist = indexlist
        self.comment_text = dataframe
        self.targets = real_label
        if max_len == -1:
            self.max_len = None
        else:
            self.max_len = max_len
        self.MAX_LEN = MAX_LEN
    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])[:self.max_len]

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.MAX_LEN,
            padding='max_length',
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'pid': self.indexlist[index],
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
    def __init__(self, output_numbers=13, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-multilingual-cased',
                                                         problem_type='multi_label_classification', num_labels=output_numbers, output_attentions=True)  # 5
        self.config = self.l1.config
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, output_numbers)
        self.device = device

    def forward(self, input_ids, attention_mask, token_type_ids):
        # _, output_1 = self.l1(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        _, output_1, _ = self.l1(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.
class BERTMLPClass(torch.nn.Module):
    def __init__(self, output_numbers=13, tabular_input_dim=77):
        super(BERTMLPClass, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(
            'bert-base-multilingual-cased',
            output_attentions=True,
            output_hidden_states=True  # True로 하면 hidden layer도 XAI 대상 가능
        )
        self.bert_dropout = nn.Dropout(0.3)

        self.config = self.bert.config

        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, 128),
            nn.LayerNorm(128),  # or BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Final classification MLP
        combined_dim = 768 + 64  # BERT CLS + Tabular output
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_numbers)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, tabular_data):
        last_hidden_state, pooled_output, hidden_states, attentions = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )

        cls_token = self.bert_dropout(last_hidden_state[:, 0])

        # Tabular features
        tabular_out = self.tabular_mlp(tabular_data)

        # Concatenate BERT + Tabular
        combined = torch.cat((cls_token, tabular_out), dim=1)

        # Final classification
        output = self.fc(combined)
        return output, attentions


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



def setting_bert_model(device, bert_model_path=None, output_numbers=13):
    if bert_model_path is None:
        model = BERTClass(output_numbers=output_numbers, device=device)
        model.to(device)
    else:
        model = BERTClass(output_numbers=output_numbers).to(device)
        model.load_state_dict(torch.load(f'{bert_model_path}', map_location=device))
    return model

def validation(model, x, epoch, device):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    predbs = []
    pids = []
    with torch.no_grad():
        # for _, data in enumerate(testing_loader):
        for _, data in tqdm(enumerate(x), total=len(x), desc=f'Val Epoch {epoch}', leave=False):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.int64)
            outputs = model(ids, mask, token_type_ids)
            probs = F.softmax(outputs, dim=-1)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            outputs = outputs.argmax(dim=1)
            max_probs, _ = torch.max(probs, dim=-1)
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            pids.extend(data['pid'].cpu().detach().numpy().tolist())
            predbs.extend(max_probs.cpu().detach().numpy().tolist())

    return fin_outputs, fin_targets, predbs, pids

def bert_validation(device, datas, model_output, max_lan=MAX_LEN, output_numbers=13,
                    bert_check_point=None, resume=False):
    nlp_data = datas[0][:5000]  # Assuming nlp_data is a DataFrame or similar structure
    labels = datas[1][:5000]
    nlp_index = datas[2][:5000]
    cur_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    validation_set = CustomDataset(nlp_index, nlp_data, labels, cur_tokenizer, max_lan)
    params = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 0
    }
    validation_loader = DataLoader(validation_set, **params)
    bert_model = setting_bert_model(device, bert_check_point, output_numbers=output_numbers)
    output_list = []
    target_list = []
    outputs, targets, preds, idxs = validation(bert_model, validation_loader, epoch="validation", device=device)
    test_acc = accuracy_score(targets, outputs)
    test_f1 = f1_score(targets, outputs, average='weighted')
    df = pd.DataFrame([idxs, targets, outputs, preds], columns = ['pid', 'target', 'output', 'preds']).T
    df.to_csv('validation_result.csv', index=False)
    print(test_acc, test_f1)

def bert_training(device, datas, model_output, loss_fn = None,
                  max_len=MAX_LEN, output_numbers=13,
                  bert_check_point=None, resume=False):
    train_dataset = datas[0]
    test_dataset = datas[1]
    train_labels = datas[2]
    test_labels = datas[3]
    print("-"*15,"bert model training start","-"*15)
    bert_model = setting_bert_model(device, bert_check_point, output_numbers=output_numbers)
    bert_model_output = model_output
    cur_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    optimizer = torch.optim.Adam(params=bert_model.parameters(), lr=LEARNING_RATE)
    # __init__(self, indexlist, dataframe, real_label, tokenizer, max_len)
    training_set = CustomDataset(train_dataset, train_dataset, train_labels, cur_tokenizer, max_len)
    testing_set = CustomDataset(test_dataset, test_dataset, test_labels, cur_tokenizer, max_len)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }
    group_name = "Hyb2"
    tags = ["SCI", "Hybrid", "MLP"]
    notes = "Hyb2 model training"
    wandb_init(LEARNING_RATE, EPOCHS, TRAIN_BATCH_SIZE, group_name=group_name, tags=tags, notes=notes, resume=resume)
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    best_loss = 1000000000
    best_f1 = 0
    best_acc = 0
    for epoch in tqdm(range(EPOCHS)):
        loss_result, acc, pre, recall, f1, cm = train(bert_model, training_loader, epoch=epoch, optimizer=optimizer, device=device, loss_fn=loss_fn)
        outputs, targets, _, _ = validation_bert(bert_model, testing_loader, epoch=epoch, device=device)
        test_acc = accuracy_score(targets, outputs)
        test_f1 = f1_score(targets, outputs, average='weighted')
        if loss_result < best_loss:
            print(f"Best loss:{epoch}, loss:{loss_result}, test_acc:{test_acc}, test_f1:{test_f1}")
            best_loss = loss_result
            torch.save(bert_model.state_dict(), f'{bert_model_output}_best_loss_model.pth')

        if test_f1 > best_f1:
            print(f"Best F1:{epoch}, loss:{loss_result}, train_f1:{f1}, test_f1:{test_f1}")
            best_f1 = f1
            torch.save(bert_model.state_dict(), f'{bert_model_output}_best_f1_model.pth')

        if test_acc > best_acc:
            print(f"Best acc:{epoch}, loss:{loss_result}, train_acc:{acc}, test_acc:{test_acc}")
            best_acc = acc
            torch.save(bert_model.state_dict(), f'{bert_model_output}_best_acc_model.pth')
        wandb.log({
            "tr_accuracy": acc, "tr_loss": loss_result, "tr_f1": f1,
            "test_accuracy": test_acc, "test_f1": test_f1
        })
    print("-"*15,"bert model training end","-"*15)
    print("best loss model:", f'{bert_model_output}_best_loss_model.pth')
    print("best f1 model:", f'{bert_model_output}_best_f1_model.pth')
    torch.save(bert_model.state_dict(), f'{bert_model_output}.pth')