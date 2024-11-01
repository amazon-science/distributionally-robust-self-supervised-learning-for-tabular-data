import argparse
import random

from sklearn.preprocessing import MinMaxScaler

from tab_transformer.ft_transformer import FTTransformer, MaskedFeatureModel, SupervisedModel
from tab_transformer.tab_transformer_pytorch import TabTransformer
# from src.tab_transformer.ft_transformer import FTTransformer

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from pathlib import Path
from torch import optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import os
import time

import matplotlib.pyplot as plt


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", type=str)
    # parser.add_argument("--full_csv", default="data/bank/bank-full.csv", type=str)
    parser.add_argument("--full_csv",
                        default="bank-additional-full.csv",
                        type=str)
    parser.add_argument("--model_type", default="FTTransformer", type=str)
    parser.add_argument("--output_path", default="output/JTT", type=str)
    # parser.add_argument('--categories', nargs='+',
    #                     default=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
    #                              'day', 'poutcome'],
    #                     help='List of categories to process (default: %(default)s)')
    parser.add_argument('--categories', nargs='+',
                        default=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                                 'day_of_week', 'poutcome'],
                        help='List of categories to process (default: %(default)s)')

    parser.add_argument('--num_cols', nargs='+',
                        default=None,
                        # default=['age', 'balance'],
                        help='List of categories to process (default: %(default)s)')
    parser.add_argument("--output_col", default="y", type=str)
    parser.add_argument("--using_gpu", default=True, type=bool)
    parser.add_argument("--max_epoch_phase1A", default=1, type=int)
    parser.add_argument("--max_epoch_phase1B", default=1, type=int)
    parser.add_argument("--max_epoch_phase2B", default=5, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--runs", default=1, type=int)
    parser.add_argument("--lr", default=1e-2, type=int)
    parser.add_argument("--dim", default=32, type=int)
    parser.add_argument("--depth", default=6, type=int)
    parser.add_argument("--heads", default=8, type=int)
    parser.add_argument("--dim_head", default=16, type=int)
    parser.add_argument("--dim_out", default=1, type=int)
    parser.add_argument("--attn_dropout", default=0.1, type=int)
    parser.add_argument("--ff_dropout", default=0.1, type=int)
    parser.add_argument("--mask_val", default=0.1, type=float)
    parser.add_argument("--seed", default=43, type=int)
    parser.add_argument("--upweight_factor", default=5, type=int)
    parser.add_argument("--dataset", default="bank", type=str)

    return parser.parse_args()


def calculate_accuracy_per_category(
        positive_samples, categories, avg_accuracy, pred_col, correct_col, output_col='y',
        text_to_show="Detailed Accuracy per Feature Category for anomalous samples",
        phase="Phase1B"
):
    results = []
    accuracies_dict = {}
    print("===================== Performance wrt different features ==================")
    print(text_to_show)

    for category in categories:
        category_data = positive_samples.groupby(category).agg(
            num_rows=pd.NamedAgg(column=pred_col, aggfunc='size'),
            accuracy=pd.NamedAgg(column=correct_col, aggfunc='mean')  # Ensures column name is used correctly
        ).reset_index()

        # avg_accuracy = category_data['accuracy'].mean()
        unique_values = category_data[category].tolist()
        print(
            f"=============== {category}: {unique_values}, Avg acc on the subset: {avg_accuracy:.4f} ===============")

        for _, row in category_data.iterrows():
            print(
                f"Num of samples: {row['num_rows']}, Accuracy for {category}={row[category]} with {output_col}=1: {row['accuracy']:.4f}")

            if category not in accuracies_dict:
                accuracies_dict[category] = {}
            if phase not in accuracies_dict[category]:
                accuracies_dict[category][phase] = []

            accuracies_dict[category][phase].append((row[category], row['accuracy']))

        results.append({
            'category': category,
            'data': category_data,
            'average_accuracy': avg_accuracy
        })

    return accuracies_dict


def plot_accuracies(accuracies_dict_phase1, accuracies_dict_phase2, categories, save_path, file_name):
    bar_width = 0.35
    for category in categories:
        phase1B_accuracies = [acc for val, acc in sorted(accuracies_dict_phase1[category]['Phase1B'])]
        phase2B_accuracies = [acc for val, acc in sorted(accuracies_dict_phase2[category]['Phase2B'])]
        unique_values = [val for val, acc in sorted(accuracies_dict_phase1[category]['Phase1B'])]

        index = np.arange(len(unique_values))

        plt.figure(figsize=(15, 8))

        bar1 = plt.bar(index, phase1B_accuracies, bar_width, label=f'Phase1B ERM')
        bar2 = plt.bar(index + bar_width, phase2B_accuracies, bar_width, label=f'Phase2B DFR')

        plt.xlabel(f'{category} Categories')
        plt.ylabel('Accuracy')
        plt.title(f'Comparison of Accuracies for {category}')
        plt.xticks(index + bar_width / 2, unique_values, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path / file_name.format(f"{category}.png"))
        plt.clf()



def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def print_first_parameter_weights(model, text=None):
    params = list(model.parameters())
    if len(params) > 0:
        first_param = params[0]
        print(f"[{text}]: Weights of the first parameter:")
        print(first_param.data[0:10])
    else:
        print("The model does not have any parameters.")


def get_input(df, entities):
    """
    from raw df with string features
    generate mapping file and id input format for training data
        df: dataframe with raw input
        entities: list of entities to use for training
    """
    df_id = df.copy()
    all_id_map = {}
    for col in entities:
        new_id, order = pd.factorize(df[col])
        id_map = {e: i for i, e in enumerate(order)}
        df_id[col] = new_id
        all_id_map[col] = id_map

    return df_id, all_id_map


def get_test_input(df, entities, all_id_map):
    """
    from raw df with string features and mapping file for training data
    generate id input format for test data
        df: dataframe with raw input
        all_id_map: dictionary with mapping for each entity in entities
        entities: list of entities to use for training
    """
    df_id = df.copy()
    for col in entities:
        id_map = all_id_map[col]
        df_id[col] = [id_map.get(val, 0) for val in df[col]]

    return df_id


def preprocess_data(args, run):
    feature_map = None
    count_class_0, count_class_1, train, dev, test = 0, 0, None, None, None
    data_dir = Path(args.data_dir)
    if args.dataset.lower() == 'bank':
        print(f"Dataset: {args.dataset.lower()}")
        file_path = data_dir / args.full_csv
        with open(file_path, 'r') as file:
            first_lines = [next(file) for _ in range(5)]

        print(first_lines)
        bank_data = pd.read_csv(file_path, delimiter=';', quotechar='"', engine='python')
        scaler = MinMaxScaler()
        bank_data[args.num_cols] = scaler.fit_transform(bank_data[args.num_cols])
        print(bank_data.columns)
        indexed_train_df, feature_map = get_input(
            df=bank_data, entities=args.categories + [args.output_col])

        indexed_train_df_cat = indexed_train_df[args.categories]
        # print(indexed_train_df_cat)
        # gr = 1
        # for cat in args.categories:
        #     gr = gr * len(list(set(indexed_train_df_cat[cat].values)))
        #     print(cat)
        #     print(list(set(indexed_train_df_cat[cat].values)))
        #     # print(len(list(set(indexed_train_df_cat[cat].values))))
        # print(gr)
        # print(xxxxx)

        count_class_0 = indexed_train_df[indexed_train_df[args.output_col] == 0][args.output_col].shape[0]
        count_class_1 = indexed_train_df[indexed_train_df[args.output_col] == 1][args.output_col].shape[0]
        print(feature_map)
        train, remaining = train_test_split(indexed_train_df, train_size=0.7, random_state=42)
        dev, test = train_test_split(remaining, test_size=2 / 3, random_state=42)

        train = train.reset_index(drop=True)
        dev = dev.reset_index(drop=True)
        test = test.reset_index(drop=True)
        print(train.shape, dev.shape, test.shape)

        print(train.columns, dev.columns, test.columns)
        print(train.head(3))
        print(dev.head(3))
        print(test.head(3))

    elif args.dataset.lower() == 'kdd':
        print(f"Dataset: {args.dataset.lower()}")
        file_path = data_dir / args.full_csv
        with open(file_path, 'r') as file:
            first_lines = [next(file) for _ in range(5)]

        print(first_lines)
        column_names = [
            "duration", "protocol_type", "service", "flag", "src_bytes",
            "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
            "num_failed_logins", "logged_in", "num_compromised", "root_shell",
            "su_attempted", "num_root", "num_file_creations", "num_shells",
            "num_access_files", "num_outbound_cmds", "is_host_login",
            "is_guest_login", "count", "srv_count", "serror_rate",
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
            "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate", "label"
        ]

        # Load the data
        kdd_data = pd.read_csv(file_path, names=column_names, header=None)
        scaler = MinMaxScaler()
        kdd_data[args.num_cols] = scaler.fit_transform(kdd_data[args.num_cols])

        print(kdd_data.columns)

        indexed_train_df, feature_map = get_input(
            df=kdd_data, entities=args.categories + [args.output_col])

        def convert_to_binary(label):
            return 0 if label == 0 else 1

        indexed_train_df['label'] = indexed_train_df['label'].apply(convert_to_binary)
        count_class_0 = indexed_train_df[indexed_train_df[args.output_col] == 0][args.output_col].shape[0]
        count_class_1 = indexed_train_df[indexed_train_df[args.output_col] == 1][args.output_col].shape[0]

        feature_map.pop('label', None)
        print(feature_map)
        train, remaining = train_test_split(indexed_train_df, train_size=0.7, random_state=42)
        dev, test = train_test_split(remaining, test_size=2 / 3, random_state=42)

        train = train.reset_index(drop=True)
        dev = dev.reset_index(drop=True)
        test = test.reset_index(drop=True)
        print(train.shape, dev.shape, test.shape)

    elif args.dataset.lower() == 'solar-flare':
        print(f"Dataset: {args.dataset.lower()}")
        file_path = data_dir / args.full_csv
        with open(file_path, 'r') as file:
            first_lines = [next(file) for _ in range(5)]
        print(first_lines[1:])

        column_names = ["Modified Zurich Class", "Largest Spot Size", "Spot Distribution", "Activity", "Evolution",
                        "Previous 24 hour flare activity code", "Historically Complex",
                        "Region Became Historically Complex on This Pass", "Area",
                        "Area of the Largest Spot", "C-class Flares", "M-class Flares", "X-class Flares"]

        # Load the data
        flare_data = pd.read_csv(file_path, header=None, names=column_names, delimiter=' ')
        flare_data = flare_data.drop(flare_data.index[0])
        print(flare_data.head())
        flare_data.rename(columns={'Modified Zurich Class': 'Modified_Zurich_Class'}, inplace=True)
        flare_data.rename(columns={'Largest Spot Size': 'Largest_Spot_Size'}, inplace=True)
        flare_data.rename(columns={'Spot Distribution': 'Spot_Distribution'}, inplace=True)
        flare_data.rename(columns={'Previous 24 hour flare activity code': 'Previous_24_hour_flare_activity_code'},
                          inplace=True)
        flare_data.rename(columns={'Area of the Largest Spot': 'Area_of_the_Largest_Spot'}, inplace=True)
        flare_data.rename(columns={'Historically Complex': 'Historically_Complex'}, inplace=True)
        flare_data.rename(columns={
            'Region Became Historically Complex on This Pass': 'Region_Became_Historically_Complex_on_This_Pass'},
            inplace=True)
        flare_data.rename(columns={'Activity': 'Activity'}, inplace=True)
        flare_data.rename(columns={'Evolution': 'Evolution'}, inplace=True)
        print(flare_data.columns)

        indexed_train_df, feature_map = get_input(
            df=flare_data, entities=args.categories + [args.output_col])

        def convert_to_binary(label):
            return 0 if label == 0 else 1

        indexed_train_df['M-class Flares'] = indexed_train_df['M-class Flares'].apply(convert_to_binary)
        indexed_train_df['C-class Flares'] = indexed_train_df['M-class Flares'].apply(convert_to_binary)
        indexed_train_df['X-class Flares'] = indexed_train_df['M-class Flares'].apply(convert_to_binary)
        count_class_0 = indexed_train_df[indexed_train_df[args.output_col] == 0][args.output_col].shape[0]
        count_class_1 = indexed_train_df[indexed_train_df[args.output_col] == 1][args.output_col].shape[0]

        feature_map.pop('C-class Flares', None)
        feature_map.pop('M-class Flares', None)
        feature_map.pop('X-class Flares', None)

        indexed_train_df['Area'] = indexed_train_df['Area'].astype(int)
        print(feature_map)
        train, remaining = train_test_split(indexed_train_df, train_size=0.7, random_state=42)
        dev, test = train_test_split(remaining, test_size=2 / 3, random_state=42)

        train = train.reset_index(drop=True)
        dev = dev.reset_index(drop=True)
        test = test.reset_index(drop=True)
        print(train.shape, dev.shape, test.shape)


    elif args.dataset.lower() == 'census':
        train_file_name = args.full_csv + ".data"
        file_path = data_dir / train_file_name
        column_names = [
            "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income(>=50k)"
        ]
        census_data_train = pd.read_csv(file_path, header=None, names=column_names, delimiter=',')
        test_file_name = args.full_csv + ".test"
        file_path = data_dir / test_file_name

        # Load the data
        census_data_test = pd.read_csv(file_path, header=None, names=column_names, delimiter=',')
        census_data_test = census_data_test.drop(census_data_test.index[0])

        scaler = MinMaxScaler()
        census_data_train[args.num_cols] = scaler.fit_transform(census_data_train[args.num_cols])
        census_data_test[args.num_cols] = scaler.fit_transform(census_data_test[args.num_cols])

        indexed_train_df, feature_map = get_input(
            df=census_data_train,
            entities=args.categories + [args.output_col]
        )

        indexed_test_df, _ = get_input(
            df=census_data_test,
            entities=args.categories + [args.output_col]
        )

        # print(census_data_train.head(4))
        # print(census_data_test.head(4))
        # print(indexed_train_df.head(4))
        # print(indexed_test_df.head(4))

        feature_map.pop('income(>=50k)', None)

        count_class_0 = indexed_train_df[indexed_train_df[args.output_col] == 0][args.output_col].shape[0]
        count_class_1 = indexed_train_df[indexed_train_df[args.output_col] == 1][args.output_col].shape[0]
        print(feature_map)

        train, dev = train_test_split(indexed_train_df, train_size=0.7, random_state=42)

        train = train.reset_index(drop=True)
        dev = dev.reset_index(drop=True)
        test = indexed_test_df

        print(train.shape, dev.shape, test.shape)
        # print(train.columns, dev.columns, test.columns)

    return count_class_0, count_class_1, train, dev, test, feature_map


class TabularDataset(Dataset):
    def __init__(self, df, categories, num_cols, output_col):
        self.data = df
        self.categories = categories
        self.num_cols = num_cols
        self.output_col = output_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        category_values = torch.tensor([row[c] for c in self.categories], dtype=torch.long)

        # Handle None for num_cols
        if self.num_cols is None:
            numerical_values = torch.tensor([], dtype=torch.float32)
        else:
            numerical_values = torch.tensor([row[c] for c in self.num_cols], dtype=torch.float32)

        output = torch.tensor(row[self.output_col], dtype=torch.float32)

        return category_values, numerical_values, output


# def validate_batch(model, data_loader, device, eval_metric):
#     model.eval()
#     predictions = []
#     true_labels = []
#     with torch.no_grad():
#         for embeddings, labels in data_loader:
#             embeddings = embeddings.to(device)
#             outputs = model(embeddings).squeeze()
#             predictions.extend(torch.sigmoid(outputs).cpu().numpy())
#             true_labels.extend(labels.numpy())
#     val_auroc = roc_auc_score(true_labels, predictions)
#     return val_auroc

def validate_batch(model, data_loader, device, eval_metric):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for embeddings, labels in data_loader:
            embeddings = embeddings.to(device)
            outputs = model(embeddings).squeeze()
            if eval_metric == "auroc":
                # Assuming sigmoid output for binary classification
                predictions.extend(torch.sigmoid(outputs).cpu().numpy())
            else:
                # Assuming softmax output for multiclass classification
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    if eval_metric == "auroc":
        val_auroc = roc_auc_score(true_labels, predictions)
        return val_auroc
    elif eval_metric == "accuracy":
        val_accuracy = accuracy_score(true_labels, predictions)
        return val_accuracy
    else:
        raise ValueError(f"Unsupported eval_metric: {eval_metric}")


def train_supervised_model(train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels,
                           epochs=10, lr=0.01, batch_size=64, save_path='best_model.pth', eval_metric="accuracy"):
    print(f"Training for {epochs}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SupervisedModel(input_dim=train_embeddings.shape[1]).to(device)
    train_dataset = TensorDataset(torch.tensor(train_embeddings, dtype=torch.float32),
                                  torch.tensor(train_labels, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.tensor(val_embeddings, dtype=torch.float32),
                                torch.tensor(val_labels, dtype=torch.float32))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = TensorDataset(torch.tensor(test_embeddings, dtype=torch.float32),
                                 torch.tensor(test_labels, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if not os.path.isfile(save_path):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        best_metric = 0
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for embeddings, labels in train_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(embeddings).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            val_metric = validate_batch(model, val_loader, device, eval_metric)

            if val_metric > best_metric:
                best_metric = val_metric
                print(f'Saving best sup model with best_{eval_metric}: {best_metric} in epoch: {epoch}')
                torch.save(model.state_dict(), save_path)
        print(f"[INFO] Supervised model training complete. Best model saved to {save_path}")

    if eval_metric == "auroc":
        model.load_state_dict(torch.load(save_path))
        predictions, labels = [], []
        for embeddings, label in test_loader:
            embeddings = embeddings.to(device)
            with torch.no_grad():
                output = model(embeddings).squeeze()
            predicted_probs = torch.sigmoid(output).cpu().numpy()
            predictions.extend(predicted_probs)
            labels.extend(label.numpy())

        predictions = np.array(predictions)
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        j_scores = tpr - fpr
        max_j_index = np.argmax(j_scores)
        roc_opt_threshold = thresholds[max_j_index]
        binary_predictions = (predictions > roc_opt_threshold).astype(int)
        test_auroc = roc_auc_score(labels, predictions)
        test_precision = precision_score(labels, binary_predictions)
        test_recall = recall_score(labels, binary_predictions)
        test_f1 = f1_score(labels, binary_predictions)
        test_accuracy = accuracy_score(labels, binary_predictions)
        cm = confusion_matrix(labels, binary_predictions)
        accuracy_y1 = cm[1, 1] / cm[1, :].sum() if cm[1, :].sum() > 0 else 0
        accuracy_y0 = cm[0, 0] / cm[0, :].sum() if cm[0, :].sum() > 0 else 0

        print(
            f"Test Metrics -> AUROC: {test_auroc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, "
            f"F1: {test_f1:.4f}, Accuracy: {test_accuracy:.4f}, Optimal ROC Threshold: {roc_opt_threshold:.4f}, "
            f"Accuracy (y=1): {accuracy_y1:.4f}, Accuracy (y=0): {accuracy_y0:.4f}")

        return predictions, binary_predictions, accuracy_y1, accuracy_y0


def get_reconstructions(data_loader, model, device):
    model.eval()
    original_categories = []
    reconstructed_categories = []

    with torch.no_grad():
        for cat_x, num_x, _ in data_loader:
            cat_x, num_x = cat_x.to(device), num_x.to(device)
            mask_cont, mask_cat = model.generate_masks(num_x.shape[0], mask_val=0, mode="test")
            _, reconstructed_cat_logits = model(num_x, cat_x, mask_cont, mask_cat)

            reconstructed_cats = [torch.max(logits, dim=1)[1] for logits in reconstructed_cat_logits]
            reconstructed_cat = torch.stack(
                reconstructed_cats, dim=1) if reconstructed_cats else torch.tensor([], dtype=torch.long,
                                                                                   device=device)

            reconstructed_categories.append(reconstructed_cat)
            original_categories.append(cat_x)

    # Concatenate all the gathered data
    original_categories = torch.cat(original_categories, dim=0).cpu()
    reconstructed_categories = torch.cat(reconstructed_categories, dim=0).cpu()

    return original_categories, reconstructed_categories


def calculate_sample_weights(error_samples, num_samples, num_features, upweight_factor=10):
    weights = torch.ones((num_samples, num_features))

    for feature_index, sample_indices in error_samples.items():
        for sample_index in sample_indices:
            weights[sample_index, feature_index] = upweight_factor

    return weights


def identify_error_samples(original_features, reconstructed_features, threshold=0.1):
    print(f"original_features: {original_features.shape}")
    print(f"reconstructed_features: {reconstructed_features.shape}")
    error_samples = {i: [] for i in range(original_features.shape[1])}
    ### Fix it
    for i in range(original_features.shape[0]):
        for j in range(original_features.shape[1]):
            if original_features[:, j].dtype == torch.int64:
                if original_features[i, j] != reconstructed_features[i, j]:
                    error_samples[j].append(i)
            else:
                if abs(original_features[i, j] - reconstructed_features[i, j]) > threshold:
                    error_samples[j].append(i)

    for key, val in error_samples.items():
        print(f"key: {key}, val: {len(val)}")

    return error_samples


def MLM_loss_weighted(reconstructed_cont, original_cont, mask_cont, reconstructed_cat, original_cat, mask_cat, weights):
    mse_loss = torch.mean(
        weights[:, :reconstructed_cont.shape[1]] * ((reconstructed_cont - original_cont) ** 2 * mask_cont))
    cat_loss = 0.0
    for i in range(len(reconstructed_cat)):
        feature_loss = torch.nn.functional.cross_entropy(reconstructed_cat[i],
                                                         original_cat[:, i].long(),
                                                         reduction='none')
        weighted_feature_loss = feature_loss * mask_cat[:, i].float() * weights[:, reconstructed_cont.shape[1] + i]
        cat_loss += torch.mean(weighted_feature_loss)

    if len(reconstructed_cat) > 0:
        cat_loss /= len(reconstructed_cat)

    return mse_loss + cat_loss


def train_MLM_epoch_weighted(masked_feature_model, data_loader, optimizer, device, weights):
    masked_feature_model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Training weighted MLM')
    data_iter = iter(data_loader)

    for batch_index, (cat_x, num_x, _) in enumerate(progress_bar):
        cat_x, num_x = cat_x.to(device), num_x.to(device)
        batch_weights = weights[batch_index * data_loader.batch_size:(batch_index + 1) * data_loader.batch_size].to(
            device)

        mask_cont, mask_cat = masked_feature_model.generate_masks(num_x.shape[0], mask_val=args.mask_val)
        reconstructed_cont, reconstructed_cat = masked_feature_model(num_x, cat_x, mask_cont, mask_cat)
        loss = MLM_loss_weighted(reconstructed_cont, num_x, mask_cont, reconstructed_cat, cat_x, mask_cat,
                                 batch_weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'Loss': loss.item()})

    return total_loss / len(data_loader)


def MLM_loss(reconstructed_cont, original_cont, mask_cont, reconstructed_cat, original_cat, mask_cat):
    mse_loss = torch.mean((reconstructed_cont - original_cont) ** 2 * mask_cont)
    cat_loss = 0.0
    for i in range(len(reconstructed_cat)):
        feature_loss = torch.nn.functional.cross_entropy(reconstructed_cat[i],
                                                         original_cat[:, i].long(),
                                                         reduction='none')
        cat_loss += torch.mean(feature_loss * mask_cat[:, i].float())

    cat_loss /= len(reconstructed_cat)
    return mse_loss + cat_loss


def train_MLM_epoch(masked_feature_model, data_loader, optimizer, device):
    masked_feature_model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Training MLM')
    for cat_x, num_x, _ in progress_bar:
        cat_x, num_x = cat_x.to(device), num_x.to(device)
        mask_cont, mask_cat = masked_feature_model.generate_masks(num_x.shape[0], mask_val=args.mask_val)
        reconstructed_cont, reconstructed_cat = masked_feature_model(num_x, cat_x, mask_cont, mask_cat)
        loss = MLM_loss(reconstructed_cont, num_x, mask_cont, reconstructed_cat, cat_x, mask_cat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'Loss': loss.item()})

    return total_loss / len(data_loader)


import numpy as np
import pandas as pd


def validate_MLM_epoch(model, data_loader, device, save_dir, test_df, feature_names=None, save_reconstructions=True):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc='Validating')

    all_reconstructed_categories = []

    # Initialize accuracy counts and sample tracking
    if feature_names is not None:
        feature_accuracy_counts = {name: {} for name in feature_names}
        total_samples = {name: {} for name in feature_names}

    with torch.no_grad():
        for cat_x, num_x, _ in progress_bar:
            cat_x, num_x = cat_x.to(device), num_x.to(device)

            mask_cont, mask_cat = model.generate_masks(num_x.shape[0], mask_val=args.mask_val)
            reconstructed_cont, reconstructed_cat = model(num_x, cat_x, mask_cont, mask_cat)
            loss = MLM_loss(reconstructed_cont, num_x, mask_cont, reconstructed_cat, cat_x, mask_cat)
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})

            if save_reconstructions:
                reconstructed_cats = torch.stack([torch.argmax(logits, dim=1) for logits in reconstructed_cat], dim=1)
                all_reconstructed_categories.append(reconstructed_cats.cpu().numpy())

            if feature_names is not None:
                for i, logits in enumerate(reconstructed_cat):
                    predicted = torch.argmax(logits, dim=1)
                    correct = (predicted == cat_x[:, i])

                    for value in cat_x[:, i].unique():
                        value = value.item()
                        mask = (cat_x[:, i] == value)
                        if value not in feature_accuracy_counts[feature_names[i]]:
                            feature_accuracy_counts[feature_names[i]][value] = []
                            total_samples[feature_names[i]][value] = 0
                        feature_accuracy_counts[feature_names[i]][value].append(correct[mask].float().mean().item())
                        total_samples[feature_names[i]][value] += mask.sum().item()

    average_loss = total_loss / len(data_loader)

    if save_reconstructions:
        all_reconstructed_categories = np.concatenate(all_reconstructed_categories, axis=0)
        reconstructed_df = pd.DataFrame(all_reconstructed_categories,
                                        columns=[f'JTT_reconstructed_{col}' for col in feature_names])
        test_df = pd.concat([test_df.reset_index(drop=True), reconstructed_df], axis=1)
        test_df.to_csv(save_dir / "test_df_recons.csv", index=False)
        print("Reconstructed categories and updated test_df have been saved.")

    if feature_names is not None:
        print("\nDetailed Accuracy Report per Category:")
        for feature, values in feature_accuracy_counts.items():
            print(f"\n{feature}:")
            for value, accs in values.items():
                avg_accuracy = sum(accs) / len(accs)
                num_samples = total_samples[feature][value]
                print(f"  Value {value}: Avg Accuracy = {avg_accuracy:.2f}, Count = {num_samples}")

    return average_loss, test_df


def do_MLM(
        masked_feature_model, train_loader, optimizer, device, val_loader, test_loader, save_dir, saved_model,
        categories,
        max_epoch, test_df, text, weights=None, train=True):
    best_loss = float('inf')
    best_epoch = 0
    if os.path.isfile(saved_model):
        print(saved_model)
        res = masked_feature_model.load_state_dict(torch.load(saved_model))
        print(f"loading from: {saved_model}")
        print(res)
    else:
        for epoch in range(max_epoch):
            print(f"Epoch {epoch + 1}/{max_epoch} [{text}]")

            if weights is not None:
                train_loss = train_MLM_epoch_weighted(masked_feature_model, train_loader, optimizer, device, weights)
            else:
                train_loss = train_MLM_epoch(masked_feature_model, train_loader, optimizer, device)
            print(f"Training Loss: {train_loss:.4f}")

            val_loss, test_df = validate_MLM_epoch(masked_feature_model, val_loader, device, save_dir, test_df,

                                          feature_names=categories)
            print(f"Validation Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                torch.save(masked_feature_model.state_dict(), saved_model)
                print(f"New best model saved at Epoch {epoch + 1} with Loss: {best_loss:.4f}")

        masked_feature_model.load_state_dict(torch.load(saved_model))
        print(f"Loaded best model from Epoch {best_epoch + 1}")

    if test_loader is not None:
        print("\nValidate_MLM_epoch")
        test_loss, test_df = validate_MLM_epoch(masked_feature_model, test_loader, device, save_dir,
                                                test_df, feature_names=categories)
        print(f"Test Loss: {test_loss:.4f}")

        # print("\nValidate_MLM_epoch_with_worst_group_accuracy")
        # val_loss, worst_group_accuracies = validate_MLM_epoch_with_worst_group_accuracy(masked_feature_model,
        #                                                                                 val_loader, device,
        #                                                                                 feature_names=categories)
        # print(f"val Loss: {val_loss:.4f}")
        # print(f"val Worst Group Accuracy: {worst_group_accuracies:.2f}")
    return masked_feature_model


def extract_embeddings_from_masked_model(data_loader, model, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for cat_x, num_x, y in data_loader:
            cat_x, num_x = cat_x.to(device), num_x.to(device)
            emb = model.base_model(cat_x, num_x)
            embeddings.append(emb.cpu())
            labels.append(y.cpu())

    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()

    return embeddings, labels


def get_model_optim(args, num_cat, feature_map_counts, device, check_pt=None, feature_index=None):
    model = None
    optimizer = None
    masked_feature_model = None
    if args.model_type.lower() == "fttransformer":
        model = FTTransformer(
            n_cont_features=len(args.num_cols),
            cat_cardinalities=num_cat,
            d_out=args.dim_out,
            **FTTransformer.get_default_kwargs(),
        )
        masked_feature_model = MaskedFeatureModel(
            model, feature_map_counts, n_cont_features=len(args.num_cols), device=device,
            trainable_feature_indices=feature_index
        )

        optimizer = model.make_default_optimizer()
    elif args.model_type.lower() == "tabtransformer":
        model = TabTransformer(
            categories=num_cat,
            num_continuous=len(args.num_cols),
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            dim_out=args.dim_out,
            attn_dropout=args.attn_dropout,
            ff_dropout=args.ff_dropout,
            mlp_hidden_mults=(4, 2),
            mlp_act=torch.nn.ReLU(),
        )
        masked_feature_model = MaskedFeatureModel(
            model, feature_map_counts, n_cont_features=len(args.num_cols), device=device
        )
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if check_pt is not None:
        res = masked_feature_model.load_state_dict(torch.load(check_pt))
        print(f"loading from: {check_pt}")
        print(res)
    masked_feature_model.to(device)
    return model, masked_feature_model, optimizer


def pretrain_JTT(test_df, feature_names, error_samples, train_dataset, data_dir, device,
                 val_loader, test_loader, num_cat, feature_map_counts, check_pt, save_dir, all_models_exists=True):
    models = {}

    for feature_index, feature_name in enumerate(feature_names):
        print("===========" * 40)

        feature_error_indices = error_samples[feature_index]
        upsampled_indices = feature_error_indices * args.upweight_factor
        upsampled_dataset = Subset(train_dataset, upsampled_indices)
        if len(upsampled_dataset) == 0:
            print(f"\n[WARNING] No error samples for feature {feature_name}, skipping training.")
            continue

        print(f"\n[INFO] Training model for {feature_name}")
        print(f"Initial dataset size: {len(train_dataset)}")
        print(f"Upsampled dataset size: {len(upsampled_dataset)}")

        # Create DataLoader for the upsampled dataset
        train_loader = DataLoader(upsampled_dataset, batch_size=args.batch_size, shuffle=True)

        # Initialize and train model
        _, masked_feature_model, optimizer = get_model_optim(args, num_cat, feature_map_counts, device,
                                                             check_pt=check_pt, feature_index=[feature_index])
        hyper_parameter = f"mr-{args.mask_val}-upwt-{args.upweight_factor}"
        saved_model_path = data_dir / f"{args.output_path}/{args.dataset}/{args.model_type}/{hyper_parameter}/best_model-phase2A-MLM-FTT-feature-{feature_name}.pth"
        trained_model = do_MLM(
            masked_feature_model, train_loader, optimizer, device, val_loader, test_loader,
            save_dir,
            saved_model_path, categories=feature_names, max_epoch=10, test_df=test_df,
            text=f"MLM Training for {feature_name}", weights=None)

        models[feature_name] = trained_model
        print("===========" * 40)


def init(categories, num_cols, output_col, data_dir, run, device):
    count_class_0, count_class_1, train, dev, test, feature_map = preprocess_data(args, run)
    feature_map_counts = {key: len(value) for key, value in feature_map.items() if key != 'y'}
    print(f"feature_map_counts: {feature_map_counts}")
    train_dataset = TabularDataset(train, categories, num_cols, output_col)
    val_dataset = TabularDataset(dev, categories, num_cols, output_col)
    test_dataset = TabularDataset(test, categories, num_cols, output_col)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    num_cat = [len(train_dataset.data[cat].unique()) for cat in categories]

    criterion = None
    if args.dataset.lower() == "bank" or args.dataset.lower() == "census":
        weight_for_1 = count_class_0 / count_class_1
        pos_weight = torch.tensor([weight_for_1], dtype=torch.float32).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    _, masked_feature_model, optimizer = get_model_optim(args, num_cat, feature_map_counts, device)

    return masked_feature_model, optimizer, criterion, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, train, test, dev, feature_map_counts, num_cat


def perform_phase1ERM(masked_feature_model, train_loader, optimizer, device, val_loader, test_loader,
                      saved_model, save_dir, test, eval_metric):
    print(
        "\n============================= [INFO] Phase1A MLM Training FTTransformer =============================")
    masked_feature_model = do_MLM(
        masked_feature_model, train_loader, optimizer, device, val_loader, test_loader,
        save_dir, saved_model, args.categories, args.max_epoch_phase1A, test, text="Phase1A MLM Training FTTransformer",
        weights=None, train=True)

    print(
        "\n============================= [INFO] Phase1A Extracting embeddings =============================")
    train_embeddings, train_labels = extract_embeddings_from_masked_model(train_loader, masked_feature_model, device)
    val_embeddings, val_labels = extract_embeddings_from_masked_model(val_loader, masked_feature_model, device)
    test_embeddings, test_labels = extract_embeddings_from_masked_model(test_loader, masked_feature_model, device)

    print(
        "\n========================= [INFO] Phase1B Supervised Training FTTransformer ========================")
    supervised_model_path = save_dir / "best_model-phase1B-MLM-FTT-sup.pth"
    test_proba, test_bin_pred, accuracy_y1, accuracy_y0 = train_supervised_model(
        train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels,
        epochs=args.max_epoch_phase1B, lr=args.lr, batch_size=args.batch_size, save_path=supervised_model_path,
        eval_metric=eval_metric
    )

    test['test_proba_phase1B'] = test_proba
    test['test_pred_phase1B'] = test_bin_pred
    test['correct_phase1B'] = (test['test_pred_phase1B'] == test[args.output_col]).astype(int)  # Correctness check
    positive_samples = test[test[args.output_col] == 1]
    negative_samples = test[test[args.output_col] == 0]
    print(f"[INFO] Supervised model evaluation complete. Best model loaded from {supervised_model_path}")
    accuracies_dict_pos = calculate_accuracy_per_category(
        positive_samples, args.categories, accuracy_y1, pred_col="test_pred_phase1B",
        correct_col="correct_phase1B", output_col=args.output_col,
        text_to_show="Detailed Accuracy per Feature Category for anomalous samples after Phase1B",
        phase="Phase1B")
    accuracies_dict_neg = calculate_accuracy_per_category(
        negative_samples, args.categories, accuracy_y0, pred_col="test_pred_phase1B",
        correct_col="correct_phase1B", output_col=args.output_col,
        text_to_show="Detailed Accuracy per Feature Category for non-anomalous samples after Phase1B",
        phase="Phase1B")

    return accuracies_dict_pos, accuracies_dict_neg


def load_models(feature_names, args, num_cat, feature_map_counts, device, save_dir):
    models = {}
    for feature_index, feature_name in enumerate(feature_names):
        model_path = save_dir / f"best_model-phase2A-MLM-FTT-feature-{feature_name}.pth"

        if not os.path.exists(model_path):
            print(f"No checkpoint found for {feature_name} at {model_path}")
            model_path = save_dir / f"best_model-phase1A-MLM-FTT.pth"

        _, masked_feature_model, optimizer = get_model_optim(
            args, num_cat, feature_map_counts, device,
            check_pt=model_path, feature_index=[feature_index]
        )

        models[feature_name] = masked_feature_model
        print(f"Model for {feature_name} loaded from {model_path}")

    return models


# def extract_ensemble_embeddings_from_masked_models(data_loader, masked_model, models_dict, device):
#     masked_model.eval()
#     embeddings = []
#     labels = []
#
#     with torch.no_grad():
#         for cat_x, num_x, y in data_loader:
#             cat_x, num_x = cat_x.to(device), num_x.to(device)
#             print(cat_x.shape, num_x.shape)
#             y = y.to(device)
#
#             mask_cont, mask_cat = masked_model.generate_masks(num_x.shape[0], mask_val=args.mask_val)
#             reconstructed_cont, reconstructed_cats = masked_model(num_x, cat_x, None, None)
#             # reconstructed_cont, reconstructed_cats = masked_model(num_x, cat_x, mask_cont, mask_cat)
#
#             # Compute reconstruction losses for categorical features
#             losses = []
#             for idx, (reconstructed_cat, true_cat) in enumerate(zip(reconstructed_cats, torch.unbind(cat_x, dim=1))):
#                 loss = torch.nn.functional.cross_entropy(reconstructed_cat, true_cat, reduction='none')
#                 losses.append(loss)
#
#             losses = torch.stack(losses, dim=1)
#             max_loss_indices = torch.argmax(losses, dim=1)
#
#             sample_embeddings = []
#             for i, max_idx in enumerate(max_loss_indices):
#                 feature_name = list(models_dict.keys())[max_idx]
#                 model = models_dict[feature_name]
#                 model.eval()
#                 print(max_idx)
#                 embedding = model.base_model(cat_x[i:i + 1], num_x[i:i + 1])
#                 print(embedding.shape)
#                 sample_embeddings.append(embedding.cpu())
#
#             embeddings.extend(sample_embeddings)
#             labels.append(y.cpu())
#
#     embeddings = torch.cat(embeddings).numpy()
#     labels = torch.cat(labels).numpy()
#
#     return embeddings, labels


def extract_ensemble_embeddings_from_masked_models(data_loader, masked_model, models_dict, device):
    masked_model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for cat_x, num_x, y in tqdm(data_loader, desc="Processing Data"):
            cat_x, num_x = cat_x.to(device), num_x.to(device)
            y = y.to(device)

            mask_cont, mask_cat = masked_model.generate_masks(num_x.shape[0], mask_val=args.mask_val)
            reconstructed_cont, reconstructed_cats = masked_model(num_x, cat_x, None, None)
            # reconstructed_cont, reconstructed_cats = masked_model(num_x, cat_x, mask_cont, mask_cat)

            # Compute reconstruction losses for categorical features
            losses = []
            for idx, (reconstructed_cat, true_cat) in enumerate(zip(reconstructed_cats, torch.unbind(cat_x, dim=1))):
                loss = torch.nn.functional.cross_entropy(reconstructed_cat, true_cat, reduction='none')
                losses.append(loss)

            losses = torch.stack(losses, dim=1)
            max_loss_indices = torch.argmax(losses, dim=1)

            sample_embeddings = []
            for i, max_idx in enumerate(max_loss_indices):
                feature_name = list(models_dict.keys())[max_idx]
                model = models_dict[feature_name]
                model.eval()
                embedding = model.base_model(cat_x[i:i + 1], num_x[i:i + 1])
                sample_embeddings.append(embedding.cpu())

            embeddings.extend(sample_embeddings)
            labels.append(y.cpu())

    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()

    return embeddings, labels


def perform_phase2(
        train_loader, masked_feature_model, device, train_dataset, val_loader, test_loader, num_cat, feature_map_counts,
        data_dir, save_dir, feature_names, test, eval_metric):
    print(
        "\n====================== [INFO] Phase2A Error-set creation for JTT on FTTransformer =================")
    original_features, reconstructed_features = get_reconstructions(train_loader, masked_feature_model, device)
    error_samples = identify_error_samples(original_features.numpy(), reconstructed_features.numpy(), threshold=0.1)

    all_models_exists = True
    for feature_index, feature_name in enumerate(feature_names):
        model_path = save_dir / f"best_model-phase2A-MLM-FTT-feature-{feature_name}.pth"

        if not os.path.exists(model_path):
            print(f"No checkpoint found for {feature_name} at {model_path}")
            all_models_exists = False
            break

    pretrain_JTT(
        test,
        args.categories, error_samples, train_dataset, data_dir, device,
        val_loader, test_loader, num_cat, feature_map_counts,
        check_pt=save_dir / "best_model-phase1A-MLM-FTT.pth", save_dir=save_dir, all_models_exists=all_models_exists)

    loaded_models = load_models(feature_names, args, num_cat, feature_map_counts, device, save_dir)
    print(
        "\n============================= [INFO] Phase2A Extracting embeddings =============================")

    start_time = time.time()
    train_embeddings, train_labels = extract_ensemble_embeddings_from_masked_models(
        train_loader, masked_feature_model, loaded_models, device)
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = elapsed_time // 3600
    remaining_seconds = elapsed_time % 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60
    print(f"Time taken to load train embeddings: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    start_time = time.time()
    val_embeddings, val_labels = extract_ensemble_embeddings_from_masked_models(
        val_loader, masked_feature_model, loaded_models, device)
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = elapsed_time // 3600
    remaining_seconds = elapsed_time % 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60
    print(f"Time taken to load val embeddings: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    start_time = time.time()
    test_embeddings, test_labels = extract_ensemble_embeddings_from_masked_models(
        test_loader, masked_feature_model, loaded_models, device)
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = elapsed_time // 3600
    remaining_seconds = elapsed_time % 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60
    print(f"Time taken to load test embeddings: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    print(
        "\n========================== [INFO] Phase2B Supervised Training FTTransformer =========================")
    supervised_model_path = save_dir / "best_model-phase2B-MLM-FTT-sup.pth"
    test_proba, test_bin_pred, accuracy_y1, accuracy_y0 = train_supervised_model(
        train_embeddings, train_labels, val_embeddings, val_labels, test_embeddings, test_labels,
        epochs=args.max_epoch_phase2B, lr=args.lr, batch_size=args.batch_size, save_path=supervised_model_path,
        eval_metric=eval_metric
    )
    test['test_proba_phase2B'] = test_proba
    test['test_pred_phase2B'] = test_bin_pred
    test['correct_phase2B'] = (test['test_pred_phase2B'] == test[args.output_col]).astype(int)
    positive_samples = test[test[args.output_col] == 1]
    negative_samples = test[test[args.output_col] == 0]
    print(f"[INFO] Supervised model training complete. Best model saved to {supervised_model_path}")
    accuracies_dict_pos = calculate_accuracy_per_category(
        positive_samples, args.categories, accuracy_y1, pred_col="test_pred_phase2B",
        correct_col="correct_phase2B", output_col=args.output_col,
        text_to_show="Detailed Accuracy per Feature Category for anomalous samples after Phase2B",
        phase="Phase2B"
    )
    accuracies_dict_neg = calculate_accuracy_per_category(
        negative_samples, args.categories, accuracy_y0, pred_col="test_pred_phase1B",
        correct_col="correct_phase1B", output_col=args.output_col,
        text_to_show="Detailed Accuracy per Feature Category for non-anomalous samples after Phase1B",
        phase="Phase2B")

    return accuracies_dict_pos, accuracies_dict_neg


def main(args):
    seed_all(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    categories = args.categories
    num_cols = args.num_cols
    output_col = args.output_col
    data_dir = Path(args.data_dir)
    eval_metric = "auroc"
    if args.dataset.lower() == "kdd":
        args.full_csv = args.full_csv + ".data"

    for run in range(args.runs):
        print(
            f'=======================================>>>>>> Run {run} <<<<<<=======================================')
        masked_feature_model, optimizer, criterion, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, train, test, dev, feature_map_counts, num_cat = init(
            categories, num_cols, output_col, data_dir, run, device)

        save_dir = data_dir / args.output_path / args.dataset / args.model_type / f"mr-{args.mask_val}-upwt-{args.upweight_factor}"
        os.makedirs(save_dir, exist_ok=True)
        saved_model = save_dir / "best_model-phase1A-MLM-FTT.pth"

        accuracies_dict_pos_phase1, accuracies_dict_neg_phase1 = perform_phase1ERM(
            masked_feature_model, train_loader, optimizer, device, val_loader,
            test_loader,
            saved_model, save_dir, test, eval_metric=eval_metric)




        accuracies_dict_pos_phase2, accuracies_dict_neg_phase2 = perform_phase2(
            train_loader, masked_feature_model, device, train_dataset, val_loader, test_loader, num_cat,
            feature_map_counts,
            data_dir, save_dir, categories, test, eval_metric=eval_metric)

        print("<<<<================================ Pos: ================================>>>>>")
        print(accuracies_dict_pos_phase1)
        print(accuracies_dict_pos_phase2)
        print("<<<<================================ Pos: ================================>>>>>")
        print("\n\n")

        print("<<<<================================ Neg: ================================>>>>>")
        print(accuracies_dict_neg_phase1)
        print(accuracies_dict_neg_phase2)
        print("<<<<================================ Neg: ================================>>>>>")

        os.makedirs(save_dir / "Plots", exist_ok=True)
        plot_accuracies(accuracies_dict_pos_phase1, accuracies_dict_pos_phase2, args.categories,
                        save_path=save_dir / "Plots", file_name="anamolous_{}")
        plot_accuracies(accuracies_dict_neg_phase1, accuracies_dict_neg_phase2, args.categories,
                        save_path=save_dir / "Plots", file_name="non-anamolous_{}")
        print(f"Everything is saved at {save_dir}")


if __name__ == "__main__":
    args = parse_config()
main(args)
