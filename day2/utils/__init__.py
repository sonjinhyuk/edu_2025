from functools import wraps
from time import time
import pandas as pd
import numpy as np
from utils.TH_BERT import BERTClass, setting_bert_model, BertTokenizer, CustomDataset, DataLoader
import torch
from tqdm import tqdm
from pickle import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, auc, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import argparse
import ast
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap

# require_columns = ["src_ip", "dst_ip", "src_port", "dst_port", "src_mac", "dst_mac", "flow_duration", "tot_fwd_pkts", "tot_bwd_pkts", "totlen_fwd_pkts", "totlen_bwd_pkts", "fwd_pkt_len_max", "fwd_pkt_len_min", "fwd_pkt_len_mean", "fwd_pkt_len_std", "bwd_pkt_len_max", "bwd_pkt_len_min", "bwd_pkt_len_mean", "bwd_pkt_len_std", "flow_byts_s", "flow_pkts_s", "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min", "fwd_iat_tot", "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max", "fwd_iat_min", "bwd_iat_tot", "bwd_iat_mean", "bwd_iat_std", "bwd_iat_max", "bwd_iat_min", "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags", "fwd_header_len", "bwd_header_len", "fwd_pkts_s", "bwd_pkts_s", "pkt_len_min", "pkt_len_max", "pkt_len_mean", "pkt_len_std", "pkt_len_var", "fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt", "psh_flag_cnt", "ack_flag_cnt", "urg_flag_cnt", "cwe_flag_count", "ece_flag_cnt", "down_up_ratio", "pkt_size_avg", "fwd_seg_size_avg", "bwd_seg_size_avg", "fwd_byts_b_avg", "fwd_pkts_b_avg", "fwd_blk_rate_avg", "bwd_byts_b_avg", "bwd_pkts_b_avg", "bwd_blk_rate_avg", "subflow_fwd_pkts", "subflow_fwd_byts", "subflow_bwd_pkts", "subflow_bwd_byts", "init_fwd_win_byts", "init_bwd_win_byts", "fwd_act_data_pkts", "fwd_seg_size_min", "active_mean", "active_std", "active_max", "active_min", "idle_mean", "idle_std", "idle_max", "idle_min", "payload_len", 'label']
require_columns = ["src_port", "dst_port", "flow_duration", "tot_fwd_pkts", "tot_bwd_pkts", "totlen_fwd_pkts", "totlen_bwd_pkts", "fwd_pkt_len_max", "fwd_pkt_len_min", "fwd_pkt_len_mean", "fwd_pkt_len_std", "bwd_pkt_len_max", "bwd_pkt_len_min", "bwd_pkt_len_mean", "bwd_pkt_len_std", "flow_byts_s", "flow_pkts_s", "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min", "fwd_iat_tot", "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max", "fwd_iat_min", "bwd_iat_tot", "bwd_iat_mean", "bwd_iat_std", "bwd_iat_max", "bwd_iat_min", "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags", "fwd_header_len", "bwd_header_len", "fwd_pkts_s", "bwd_pkts_s", "pkt_len_min", "pkt_len_max", "pkt_len_mean", "pkt_len_std", "pkt_len_var", "fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt", "psh_flag_cnt", "ack_flag_cnt", "urg_flag_cnt", "cwe_flag_count", "ece_flag_cnt", "down_up_ratio", "pkt_size_avg", "fwd_seg_size_avg", "bwd_seg_size_avg", "fwd_byts_b_avg", "fwd_pkts_b_avg", "fwd_blk_rate_avg", "bwd_byts_b_avg", "bwd_pkts_b_avg", "bwd_blk_rate_avg", "subflow_fwd_pkts", "subflow_fwd_byts", "subflow_bwd_pkts", "subflow_bwd_byts", "init_fwd_win_byts", "init_bwd_win_byts", "fwd_act_data_pkts", "fwd_seg_size_min", "active_mean", "active_std", "active_max", "active_min", "idle_mean", "idle_std", "idle_max", "idle_min", "payload_len", 'label']
def encode_labels(labels):
    le = LabelEncoder()
    new_labels = le.fit_transform(labels)
    mapping_table = pd.DataFrame({
        'original_label': le.classes_,
        'new_label': range(len(le.classes_))
    })

    return new_labels, mapping_table


def check_discontinuous_labels(labels):
    labels = np.array(labels)
    unique_labels = np.sort(np.unique(labels))
    expected_labels = np.arange(unique_labels.min(), unique_labels.max() + 1)
    missing_labels = np.setdiff1d(expected_labels, unique_labels)

    if len(missing_labels) == 0 and unique_labels.min() == 0:
        return labels, None
    else:
        new_labels, new_mapping = encode_labels(labels)
        labels = new_labels

    return labels, new_mapping
@timing
def data_load(base_dir="./model_data/",
              bert_flie="bert_inputs.txt",
              encoding="utf-8",
              MAXLEN=-1):

    if MAXLEN == -1:
        MAXLEN = None
    nlp_indexs = []

    if os.path.exists(f"{base_dir}/{bert_flie}"):
        file_path = f"{base_dir}/{bert_flie}"
        # with open(file_path, "r") as f:
        #     total_lines = sum(1 for _ in f)
        total_lines = 1000
        with open(f"{base_dir}/{bert_flie}", "r") as f:
            nlp_data = []
            labels = []
            temp_lnp_index = 0
            start_nlp_index = 1
            for line in tqdm(f, desc="loading nlp data", total=total_lines, unit="line", leave=False):
                try:
                    l = line.strip().split(",")
                    label = int(l[-1].strip())
                    try:
                        nlp_indexs.append(int(l[0]))
                    except ValueError:
                        nlp_indexs.append(temp_lnp_index)
                        temp_lnp_index += 1
                        start_nlp_index = 0
                    l = ",".join(l[start_nlp_index:-1])
                    nlp_data.append(l[:MAXLEN])
                    labels.append(label)
                    if len(labels) > total_lines:
                        break
                except Exception as e:
                    print(f"[Parsing error] {e} :: {line.strip()}")
                    continue
        return (nlp_data, labels, nlp_indexs)



import os
def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)
        exit()

def get_eval_value(y, pred):
    acc = accuracy_score(y, pred)
    presision = precision_score(y, pred, average='weighted')
    recall = recall_score(y, pred, average='weighted')
    f1 = f1_score(y, pred, average='weighted')
    cm = confusion_matrix(y, pred)

    return [acc, presision, recall, f1], cm


def save_metrics_and_cm(result, cm, output_path_prefix):
    # result = [accuracy, precision, recall, f1]
    result_df = pd.DataFrame([result], columns=["Accuracy", "Precision", "Recall", "F1-Score"])
    cm_df = pd.DataFrame(cm)
    if cm.shape[0] == 2:
        cm_df.columns = ["Benign", "Malware"]
        cm_df.index = ["Benign", "Malware"]
        labels = ["Benign", "Malware"]
    elif cm.shape[0] == 6:
        cm_df.columns = [i for i in range(6)]
        cm_df.index = [i for i in range(6)]
        labels = [i for i in range(6)]
    else:
        labels = [
            "Benign", "Banking_Emotet_Family", "Banking_Zeus_Family", "Banking_Other",
            "Infostealer", "RAT", "Exploit_Kit", "Malspam_Phishing", "Ransomware",
            "Recon_C2", "Other_Generic", "Cobalt_Strike"
        ]

    # Flatten true and pred labels for metrics calculation
    y_true = []
    y_pred = []

    for true_label in range(len(cm_df)):
        for pred_label in range(len(cm_df.columns)):
            count = cm_df.iloc[true_label, pred_label]
            if count > 0:  # 값이 있을 때만
                y_true.extend([true_label] * count)
                y_pred.extend([pred_label] * count)
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    cm_df.index = labels
    cm_df.columns = labels
    with pd.ExcelWriter(f"{output_path_prefix}_evaluation.xlsx") as writer:
        result_df.to_excel(writer, sheet_name="Result", index=False)
        pd.concat([cm_df, df_report.iloc[:-3, :]], axis=1).to_excel(writer, sheet_name="Confusion Matrix")



def set_device(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,5,6"  # Set the GPU 2 to use
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    return device