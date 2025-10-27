import os
import math
import shap
import torch
from utils.bert_util import MAX_LEN
from utils import create_directory
import numpy as np
from shap.maskers._text import TokenGroup, Token
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import transformers
from tqdm import tqdm

def get_tokenized_data(tokenizer, input_text):
    input_text = " ".join(input_text.split())
    tokenized_sent = tokenizer.encode_plus(
        input_text,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        pad_to_max_length=True,
        return_token_type_ids=True,
        truncation=True,
    )
    ids = tokenized_sent['input_ids']
    mask = tokenized_sent['attention_mask']
    token_type_ids = tokenized_sent["token_type_ids"]

    return {
        'input_ids': torch.tensor(ids, dtype=torch.long),
        'attention_mask': torch.tensor(mask, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
    }

def post_process_shap_value(shap_values, new_word_index, token, tokenizer, output_names, device):
    def partition_tree(decoded_tokens, tokenizer, special_tokens=None):
        def merge_score(group1, group2, special_tokens):
            """Compute the score of merging two token groups.

            special_tokens: tokens (such as separator tokens) that should be grouped last
            """
            score = 0

            # ensures special tokens are combined last, so 1st subtree is 1st sentence and 2nd subtree is 2nd sentence
            if len(special_tokens) > 0:
                if group1[-1].s in special_tokens and group2[0].s in special_tokens:
                    score -= math.inf  # subtracting infinity to create lowest score and ensure combining these groups last

            # merge broken-up parts of words first
            if group2[0].s.startswith("##"):
                score += 20

            # merge apostrophe endings next
            if group2[0].s == "'" and (len(group2) == 1 or (len(group2) == 2 and group2[1].s in ["t", "s"])):
                score += 15
            if group1[-1].s == "'" and group2[0].s in ["t", "s"]:
                score += 15

            start_ctrl = group1[0].s.startswith("[") and group1[0].s.endswith("]")
            end_ctrl = group2[-1].s.startswith("[") and group2[-1].s.endswith("]")

            if (start_ctrl and not end_ctrl) or (end_ctrl and not start_ctrl):
                score -= 1000
            if group2[0].s in openers and not group2[0].balanced:
                score -= 100
            if group1[-1].s in closers and not group1[-1].balanced:
                score -= 100

            # attach surrounding an openers and closers a bit later
            if group1[0].s in openers and group2[-1] not in closers:
                score -= 2

            # reach across connectors later
            if group1[-1].s in connectors or group2[0].s in connectors:
                score -= 2

            # reach across commas later
            if group1[-1].s == ",":
                score -= 10
            if group2[0].s == ",":
                if len(group2) > 1:  # reach across
                    score -= 10
                else:
                    score -= 1

            # reach across sentence endings later
            if group1[-1].s in [".", "?", "!"]:
                score -= 20
            if group2[0].s in [".", "?", "!"]:
                if len(group2) > 1:  # reach across
                    score -= 20
                else:
                    score -= 1

            score -= len(group1) + len(group2)
            # print(group1, group2, score)
            return score

        openers = {
            "(": ")"
        }
        closers = {
            ")": "("
        }
        enders = [".", ","]
        connectors = ["but", "and", "or"]
        """Build a heriarchial clustering of tokens that align with sentence structure.
        Note that this is fast and heuristic right now.
        TODO: Build this using a real constituency parser.
        """
        if special_tokens is None:
            special_tokens = [tokenizer.sep_token]
        token_groups = [TokenGroup([Token(t)], i) for i, t in enumerate(decoded_tokens)]
        M = len(decoded_tokens)
        new_index = M
        clustm = np.zeros((M - 1, 4))
        for i in range(len(token_groups) - 1):
            scores = [merge_score(token_groups[i], token_groups[i + 1], special_tokens) for i in
                      range(len(token_groups) - 1)]
            ind = np.argmax(scores)

            lind = token_groups[ind].index
            rind = token_groups[ind + 1].index
            clustm[new_index - M, 0] = token_groups[ind].index
            clustm[new_index - M, 1] = token_groups[ind + 1].index
            clustm[new_index - M, 2] = -scores[ind]
            clustm[new_index - M, 3] = (clustm[lind - M, 3] if lind >= M else 1) + (
                clustm[rind - M, 3] if rind >= M else 1)

            token_groups[ind] = token_groups[ind] + token_groups[ind + 1]
            token_groups[ind].index = new_index

            # track balancing of openers/closers
            if token_groups[ind][0].s in openers and token_groups[ind + 1][-1].s == openers[token_groups[ind][0].s]:
                token_groups[ind][0].balanced = True
                token_groups[ind + 1][-1].balanced = True

            token_groups.pop(ind + 1)
            new_index += 1

        # negative means we should never split a group, so we add 10 to ensure these are very tight groups
        # (such as parts of the same word)
        clustm[:, 2] = clustm[:, 2] + 10

        clustm[:, 2] = clustm[:, 3]
        clustm[:, 2] /= clustm[:, 2].max()
        return clustm

    values = shap_values[0].values
    new_values = post_values(values, device, new_word_index)
    new_clustering = partition_tree(token, tokenizer, special_tokens=None)
    new_clustering = new_clustering.reshape(-1, new_clustering.shape[0], new_clustering.shape[1])
    shap_values.data = (token,)
    shap_values.feature_names = token
    shap_values.values = (new_values,)
    shap_values.clustering = new_clustering
    shap_values.output_names = output_names
    shap_values.hierarchical_values = None
    return shap_values

def bert_shap(explainer: shap.Explainer,
              word_index: list, token: list,
              tokenizer: object, input_text: object,
              label: object, label_types: list,
              in_dict: dict, output_name: str, device:object):
    output_name = f"{output_name}"
    if os.path.exists(f"{output_name}.csv") and os.path.exists(f"{output_name}.html") \
            and os.path.exists(f"{output_name}.json"):
        with open(f"{output_name}.json", 'r', encoding='cp949') as f:
            in_dict = json.load(f)
        return in_dict
    else:
        shap_values_original = explainer([input_text], silent=True)
        shap_values = shap_values_original.__copy__()
        # shap_values = post_process_shap_value(shap_values, word_index, token, tokenizer,
        #                                       output_names=label_types, device=device)
        shap_values.output_names = label_types
        data = shap.plots.text(shap_values[0, :, :], display=False)
        with open(f"{output_name}.html", "w") as file:
            file.write(data)
        df_columns = label_types.copy()
        df_columns[label] = str(df_columns[label]) + "_o"
        temp_df = pd.DataFrame(shap_values.values[0][1:-1], columns=label_types, index=shap_values.data[0][1:-1]).T
        temp_df.to_csv(f"{output_name}.csv", encoding='cp949')
        in_dict['key_words'] = list(shap_values.data[0][1:-1])
        in_dict['base_values'] = list(shap_values.base_values[0])
        for c in label_types:
            in_dict[c] = temp_df.loc[c].to_list()
        with open(f"{output_name}.json", 'w', encoding='cp949') as f:
            json.dump(in_dict, f, indent=4, ensure_ascii=False)


def get_model_input(inputs, device):
    input_shape = inputs['input_ids'].shape[0]
    ids = inputs['input_ids'].to(device, dtype=torch.long).view(-1, input_shape)
    mask = inputs['attention_mask'].to(device, dtype=torch.long).view(-1, input_shape)
    token_type_ids = inputs['token_type_ids'].to(device, dtype=torch.long).view(-1, input_shape)
    return ids, mask, token_type_ids

def xai_post_process(cls_data=None, tokenizer=None, device=None, labels=[], _type="multi5",
                     model=None, output_dir = "output_xai", xai_types=['shap']):
    def _can_generate():
        return False
    explainer = None
    pred_pipline = None
    for xai in xai_types:
        if "shap" == xai:
            model.can_generate = _can_generate
            pred_pipline = transformers.pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=2,
                # return_all_scores=True,
                top_k=None,
                truncation=True
            )
            explainer = shap.Explainer(pred_pipline)
    label_dict = {}
    for i in range(len(labels)):
        label_dict[i] = 0
    label_types = labels
    j_index = 0

    for row in tqdm(cls_data.itertuples(index=False), total=len(cls_data), mininterval=0.001):
        input_text = row.nlp_data
        try:
            label = int(row.label)
        except TypeError:
            print(row.label)
            continue
        if label_dict[label] > 20:
            continue
        out_path = f"{output_dir}/{label}"
        create_directory(out_path)
        output_name = f"{out_path}/{j_index}"
        inputs = get_tokenized_data(tokenizer, input_text)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'])  # Convert input ids to token strings
        model_pred = pred_pipline(input_text)

        # inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        # inputs = {k: v.to(model.device) for k, v in inputs.items()}
        #
        # with torch.no_grad():
        #     logits = model(**inputs)  # BERTClass forward

        in_dict = {}
        in_dict['label'] = label
        in_dict['pred_label'] = model_pred
        in_dict['index'] = label_types
        in_dict['pred_values'] = [0 for _ in range(len(label_types))]
        max_value_label = ""
        max_value = 0
        for pred in model_pred[0]:
            label_ = pred['label']
            label_index = int(label_.split("LABEL_")[-1])
            in_dict['pred_values'][label_index] = pred['score']
            if max_value_label == "" or max_value < pred['score']:
                max_value_label = label_
                max_value = pred['score']
                in_dict['p_label'] = label_index
        in_dict['correct'] = 1 if label == in_dict['p_label'] else 0
        if in_dict['correct'] == 0:
            continue
        label_dict[label] += 1
        new_ward_token_index, new_toekn = post_tokenizer(tokens)
        if "shap" in xai_types:
            bert_shap(explainer, new_ward_token_index, new_toekn, tokenizer,
                      input_text, label, label_types, in_dict, output_name, device)
        j_index+=1

def post_tokenizer(tokens):
    new_word_index = []
    new_toekn = []
    for i, t in enumerate(tokens):
        if i == 0:
            new_word_index.append([i])
        elif t.startswith("##"):
            new_word_index[-1].append(i)
        elif t == "[SEP]":
            new_word_index.append([i])
            break
        else:
            new_word_index.append([i])
    for nwi in new_word_index:
        new_toekn.append("".join([tokens[i].replace("#", "") for i in nwi]))
    return new_word_index, new_toekn

def post_values(values, device, new_word_index):
    new_values = []
    if type(values) == tuple:
        for value in values:

            temp_layer = []
            temp_head = []
            # value = value[0][-1].cpu().detach().numpy()
            heads = value[0]
            for head in heads:
                vs = head
                ## pad 다 버림
                temp_value = []
                for i in range(len(new_word_index)):
                    v = vs[i].cpu().detach().numpy()
                    row_temp = []
                    for nwi in new_word_index:
                        try:
                            row_temp.append(v[nwi].sum())
                        except IndexError:
                            break
                    if len(row_temp) != 0:
                        temp_value.append(row_temp)
                    else:
                        break
                temp_head.append(temp_value)
            temp_layer.append(temp_head)
            temp_layer = torch.tensor(temp_layer).to(device)
            new_values.append(temp_layer)
    elif len(values.shape) == 3:
        value = values[0]
        for nwi in new_word_index:
            new_values.append(value[nwi].sum(axis=0).tolist())
        new_values = torch.tensor(new_values).to(device)
    elif len(values.shape) == 2:
        for nwi in new_word_index:
            new_values.append(values[nwi].sum(axis=0).tolist())
        new_values = np.asarray(new_values)
    return new_values




def geneate_html(html, output_name):
    if ".html" not in output_name:
        output_name += ".html"
    with open(f"{output_name}", "w") as file:
        file.write(html.data)


from xgboost import plot_importance
from sklearn.inspection import permutation_importance

def get_plot_importance(model, output_dir, output_name):
    plot_importance(model)  # model은 학습된 xgboost 모델
    plt.savefig(f"{output_dir}_{output_name}.png")
    plt.close()


def get_shap(model, tabular_X_test, output_dir, output_name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(tabular_X_test)

    with PdfPages(f'{output_dir}_{output_name}.pdf') as pdf:
        # 1. shap_values가 리스트면 (multi-class)
        if isinstance(shap_values, list):
            for i in range(len(shap_values)):
                shap.summary_plot(shap_values[i], tabular_X_test, show=False)
                plt.title(f"SHAP Summary for Class {i}")
                pdf.savefig(bbox_inches='tight')
                plt.close()
        # 2. shap_values가 배열이면 (binary-class)
        else:
            shap.summary_plot(shap_values, tabular_X_test, show=False)
            plt.title(f"SHAP Summary (Binary Classification)")
            pdf.savefig(bbox_inches='tight')
            plt.close()

def get_permutation_importance(model, tabular_X_test, tabular_y_test, output_dir, output_name):
    r = permutation_importance(model, tabular_X_test, tabular_y_test, n_repeats=30, random_state=0)
    sorted_idx = r.importances_mean.argsort()

    plt.barh(tabular_X_test.columns[sorted_idx], r.importances_mean[sorted_idx])
    plt.title("Permutation Importance")
    plt.savefig(f"{output_dir}_{output_name}.png")
    plt.close()
