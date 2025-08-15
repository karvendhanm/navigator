import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from transformers import AutoTokenizer

# importing local modules
from assorted_learning.hugging_face import config

# initializing AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_ckpt)


# Class of different styles
class Style:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


def draw_density_plot(df_embedding, cmaps, labels):
    # visualizing the different emotions in 2D
    fig, axes = plt.subplots(2, 3, figsize=(7, 5))
    axes = axes.flatten()

    for i, (label, cmap) in enumerate(zip(labels, cmaps)):
        df_emb_sub = df_embedding.query(f'label == {i}')
        axes[i].hexbin(df_emb_sub['X'], df_emb_sub['Y'], cmap=cmap, gridsize=20, linewidth=(0,))
        axes[i].set_title(label)
        axes[i].set_xticks([]), axes[i].set_yticks([])

    plt.tight_layout()
    plt.show()
    return None


def plot_confusion_matrix(y_pred, y_true, labels):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='.2f', ax=ax, colorbar=False)
    plt.title('normalized confusion matrix')
    plt.show()
    return None


def compute_metrics(pred):
    # pred is a named tuple with attributes predictions and label_ids
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}


def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)
