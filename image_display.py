# # image_display.py
# from PIL import Image
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt

# def display_image(inputs, save_path="output_image.png"):
#     transform = transforms.ToPILImage()
#     image = transform(inputs[-1])
#     image.save(save_path)
#     print(f"Image saved to {save_path}")



from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc, roc_curve
from matplotlib.patches import Patch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_curve
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

def create_prediction_image(y_true, y_pred, output_path='prediction_visualization.png'):
    """
    Creates a high-quality (300 DPI) image visualizing predictions versus ground truth, 
    confusion matrix, AU ROC, AU PRC, and totals.

    Parameters:
    - y_true (np.ndarray): 1D array of ground truth values (0 or 1).
    - y_pred (np.ndarray): 1D array of prediction values (between 0 and 1).
    - output_path (str): Path where the image will be saved.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true.cpu())
    y_pred = np.array(y_pred.cpu())

    # Define colors
    correct_color = 'green'
    incorrect_color = 'red'
    high_red = plt.cm.Reds
    high_green = plt.cm.Greens

    # Determine table data
    rows = len(y_true)
    data = []
    cell_colours = []
    correct_count = 0
    incorrect_count = 0
    for true_val, pred_val in zip(y_true, y_pred):
        true_color = correct_color if true_val == 0 else incorrect_color
        pred_color = high_green(1 - pred_val) if pred_val < 0.5 else high_red(pred_val)
        correct_wrong = '✔️' if ((true_val == 0 and pred_val < 0.5) or (true_val == 1 and pred_val >= 0.5)) else '❌'
        data.append([true_val, pred_val, correct_wrong])
        cell_colours.append([true_color, pred_color, None])
        if correct_wrong == '✔️':
            correct_count += 1
        else:
            incorrect_count += 1

    # Calculate totals
    total_zeros = np.sum(y_true == 0)
    total_ones = np.sum(y_true == 1)
    total_correct = correct_count
    total_incorrect = incorrect_count

    # Add totals to the data
    data.append([total_zeros, '', total_correct])
    data.append([total_ones, '', total_incorrect])
    cell_colours.append([None, None, None])
    cell_colours.append([None, None, None])

    # Create the figure with subplots for the table, confusion matrix, ROC, and PRC
    fig = plt.figure(figsize=(20, 12), dpi=300)
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 2], height_ratios=[1, 1, 1])

    ax_table = fig.add_subplot(gs[:, 0])
    ax_cm = fig.add_subplot(gs[0, 1])
    ax_roc = fig.add_subplot(gs[1, 1])
    ax_pr = fig.add_subplot(gs[2, 1])

    # Create the table
    ax_table.axis('off')
    table = ax_table.table(cellText=data,
                          	colLabels=['Ground Truth', 'Prediction', 'Correct/Wrong'],
                          	cellLoc='center',
                          	loc='center',
                          	cellColours=cell_colours)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Add totals labels to the table
    for key, cell in table.get_celld().items():
        if key[0] == rows:
            cell.set_text_props(text='Total 0s')
        if key[0] == rows + 1:
            cell.set_text_props(text='Total 1s')
        if key[0] == rows and key[1] == 2:
            cell.set_text_props(text='Correct')
        if key[0] == rows + 1 and key[1] == 2:
            cell.set_text_props(text='Incorrect')

    # Add legend
    legend_elements = [Patch(facecolor=correct_color, edgecolor='black', label='0 (Correct)'),
                       Patch(facecolor=incorrect_color, edgecolor='black', label='1 (Incorrect)'),
                       Patch(facecolor=high_green(0), edgecolor='black', label='Prediction ≈ 0'),
                       Patch(facecolor=high_red(1), edgecolor='black', label='Prediction ≈ 1')]
    ax_table.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=10, title='Legend')

    # Add Confusion Matrix
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2:
        cm = confusion_matrix(y_true, (y_pred >= 0.5).astype(int))
        ax_cm.matshow(cm, cmap='coolwarm', alpha=0.6)
        for i in range(2):
            for j in range(2):
                ax_cm.text(x=j, y=i, s=cm[i, j], va='center', ha='center', fontsize=12)
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(['Predicted 0', 'Predicted 1'], fontsize=10)
        ax_cm.set_yticklabels(['Actual 0', 'Actual 1'], fontsize=10)
        ax_cm.set_title('Confusion Matrix', fontsize=12)
    else:
        ax_cm.text(0.5, 0.5, 'Confusion matrix not available\n(only one class present)',
                   va='center', ha='center', fontsize=12)
        ax_cm.set_xticks([])
        ax_cm.set_yticks([])

    # Add ROC Curve
    if len(unique_classes) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        ax_roc.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('Receiver Operating Characteristic')
        ax_roc.legend(loc="lower right")
    else:
        ax_roc.text(0.5, 0.5, 'ROC curve not available\n(only one class present)',
                    va='center', ha='center', fontsize=12)
        ax_roc.set_xticks([])
        ax_roc.set_yticks([])

    # Add Precision-Recall Curve
    if len(unique_classes) == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)
        ax_pr.plot(recall, precision, color='blue', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
        ax_pr.set_xlim([0.0, 1.0])
        ax_pr.set_ylim([0.0, 1.05])
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title('Precision-Recall Curve')
        ax_pr.legend(loc="lower left")
    else:
        ax_pr.text(0.5, 0.5, 'PR curve not available\n(only one class present)',
                   va='center', ha='center', fontsize=12)
        ax_pr.set_xticks([])
        ax_pr.set_yticks([])

    # Save the image
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()



def create_prediction_image_simple(y_true, y_pred, output_path='prediction_visualization.png'):
    """
    Creates a high-quality (300 DPI) image visualizing predictions versus ground truth.

    Parameters:
    - y_true (np.ndarray): 1D array of ground truth values (0 or 1).
    - y_pred (np.ndarray): 1D array of prediction values (between 0 and 1).
    - output_path (str): Path where the image will be saved.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Define colors
    correct_color = 'green'
    incorrect_color = 'red'
    high_red = plt.cm.Reds
    high_green = plt.cm.Greens

    # Determine table data
    rows = len(y_true)
    data = []
    cell_colours = []
    for true_val, pred_val in zip(y_true, y_pred):
        true_color = correct_color if true_val == 0 else incorrect_color
        pred_color = high_green(1 - pred_val) if pred_val < 0.5 else high_red(pred_val)
        correct_wrong = '✔️' if ((true_val == 0 and pred_val < 0.5) or (true_val == 1 and pred_val >= 0.5)) else '❌'
        data.append([true_val, pred_val, correct_wrong])
        cell_colours.append([true_color, pred_color, None])

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, rows / 2), dpi=300)
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=data,
                    colLabels=['Ground Truth', 'Prediction', 'Correct/Wrong'],
                    cellLoc='center',
                    loc='center',
                    cellColours=cell_colours,  # Correctly map colors
                    colColours=[None, None, None])

    # Add legend
    legend_elements = [Patch(facecolor=correct_color, edgecolor='black', label='0 (Correct)'),
                       Patch(facecolor=incorrect_color, edgecolor='black', label='1 (Incorrect)'),
                       Patch(facecolor=high_green(0), edgecolor='black', label='Prediction ≈ 0'),
                       Patch(facecolor=high_red(1), edgecolor='black', label='Prediction ≈ 1')]
    ax.legend(handles=legend_elements, loc='best', fontsize=10, title='Legend')

    # Save the image
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def display_image(inputs, save_path="output_image.png", rows=4, cols=8):
    transform = transforms.ToPILImage()
    images = [transform(img) for img in inputs[:rows * cols]]
    width = max(img.width for img in images)
    height = max(img.height for img in images)
    combined_image = Image.new("RGB", (width * cols, height * rows))
    for i, img in enumerate(images):
        x = (i % cols) * width
        y = (i // cols) * height
        combined_image.paste(img, (x, y))
    combined_image.save(save_path)

    