from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
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
import seaborn as sns
import pandas as pd
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.CRITICAL)
logging.getLogger('matplotlib.pyplot').setLevel(logging.CRITICAL)
logging.getLogger('matplotlib.colorbar').setLevel(logging.CRITICAL)
logging.getLogger('matplotlib.category').setLevel(logging.CRITICAL)
def create_prediction_image(y_true, y_pred, output_path='create_prediction_image.png'):
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

def create_prediction_table(y_true, y_preds_dict, output_path='prediction_analysis.png', r=3):
    """
    Analyzes multiple prediction arrays against the ground truth labels.
    
    Parameters:
    - y_true (np.ndarray): 1D array of ground truth values (0 or 1).
    - y_preds_dict (dict): Dictionary where keys are column names and values are 1D arrays of prediction values (between 0 and 1).
    - output_path (str): Path where the image will be saved.
    """


    y_true = np.array(y_true>.5,dtype=np.float32).round(r)
    data = {'Ground Truth': y_true}
    for key, value in y_preds_dict.items():
        data[key] = np.array(value, dtype=np.float32).round(r)
    df = pd.DataFrame(data)
    rows = len(y_true)
    cell_colours = []
    correct_color = 'green'
    incorrect_color = 'red'
    high_red = plt.cm.Reds
    high_green = plt.cm.Greens
    correct_count = 0
    incorrect_count = 0


    # Coloring cells
    for idx, row in df.iterrows():
        true_val = row['Ground Truth']
        row_colors = [correct_color if true_val == 0 else incorrect_color]
        correct_wrong = '✔️' if all((true_val == 0 and row[pred] < 0.5) or (true_val == 1 and row[pred] >= 0.5) for pred in y_preds_dict) else '❌'
        for key in y_preds_dict:
            pred_val = row[key]
            row_colors.append(high_green(1 - pred_val) if pred_val < 0.5 else high_red(pred_val))
        cell_colours.append(row_colors)
        if correct_wrong == '✔️':
            correct_count += 1
        else:
            incorrect_count += 1

    total_zeros = np.sum(y_true == 0)
    total_ones = np.sum(y_true == 1)
    total_correct = correct_count
    total_incorrect = incorrect_count
    df.loc[rows] = [total_zeros] + [None] * (len(y_preds_dict))  # Adding 3 None values for 3 predictions
    df.loc[rows + 1] = [total_ones] + [None] * (len(y_preds_dict))  # Adding 3 None values for 3 predictions
    cell_colours.append([None] * (len(y_preds_dict) + 1))
    cell_colours.append([None] * (len(y_preds_dict) + 1))


    # Grid
    fig = plt.figure(figsize=(30, 20), dpi=300)
    fig.suptitle(output_path, fontsize=24, fontweight='bold')
    gs = plt.GridSpec( len(y_preds_dict) + 1, 5, figure=fig, height_ratios=  [2] * len(y_preds_dict) + [6], width_ratios=[3,1, 1, 1,1], wspace=0.3 ,hspace=1)
    ax_table = fig.add_subplot(gs[:, 0])
    ax_cms = [fig.add_subplot(gs[i, 1]) for i in range(len(y_preds_dict))]
    ax_roc = [fig.add_subplot(gs[ i, 2]) for i in range(len(y_preds_dict))]
    ax_pr = [fig.add_subplot(gs[ i, 3]) for i in range(len(y_preds_dict))]
    ax_scatters = [fig.add_subplot(gs[i, 4]) for i in range(len(y_preds_dict))]
    ax_heatmap = fig.add_subplot(gs[len(y_preds_dict):, 1:])


    # table
    ax_table.axis('off')
    table = ax_table.table(cellText=df.values.round(r),
                            colLabels=['Ground Truth'] + list(y_preds_dict.keys()),
                            cellLoc='center',
                            loc='center',
                            cellColours=cell_colours)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    table[(rows+1, 0)].set_text_props(text=f'Total 0s : {total_zeros}')
    table[(rows +1+ 1, 0)].set_text_props(text=f'Total 1s : {total_ones}')
    legend_elements = [Patch(facecolor=correct_color, edgecolor='black', label='0 (Correct)'),
                        Patch(facecolor=incorrect_color, edgecolor='black', label='1 (Incorrect)'),
                        Patch(facecolor=high_green(0), edgecolor='black', label='Prediction ≈ 0'),
                        Patch(facecolor=high_red(1), edgecolor='black', label='Prediction ≈ 1')]
    ax_table.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=10, title='Legend')


    # correlation matrix
    correlation_matrix = df[data.keys()].corr()
    sns.heatmap(correlation_matrix.round(r), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax_heatmap)
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=8)  # Rotate x-axis labels
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(),  rotation=45, fontsize=8)  # Set font size for y-axis labels


    # confusion matrix
    for idx, (key, value) in enumerate(y_preds_dict.items()):
        cm = confusion_matrix((value.round(r) >= 0.5).astype(int),y_true)
        ax_cms[idx].matshow(cm, cmap='coolwarm', alpha=0.6)
        for i in range(2):
            for j in range(2):
                ax_cms[idx].text(x=j, y=i, s=f"{cm[i, j]}", va='center', ha='center', fontsize=12)
        ax_cms[idx].set_xticks([0, 1])
        ax_cms[idx].set_yticks([0, 1])
        ax_cms[idx].set_yticklabels(['Predicted 0', 'Predicted 1'], rotation=45, fontsize=10)
        ax_cms[idx].set_xticklabels(['Actual 0', 'Actual 1'], rotation=45, fontsize=10)
        ax_cms[idx].set_title(f'Confusion Matrix\n({key})', fontsize=12)


    # ROC
    for i, (key, value) in enumerate(y_preds_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, value)
        roc_auc = auc(fpr, tpr)    
        ax_roc[i].plot(fpr, tpr, lw=2, label=f'{key} (AUC = {roc_auc:.2f})')
        ax_roc[i].plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
        ax_roc[i].set_xlim([0.0, 1.0])
        ax_roc[i].set_ylim([0.0, 1.05])
        ax_roc[i].set_xlabel('False Positive Rate')
        ax_roc[i].set_ylabel('True Positive Rate')
        ax_roc[i].set_title(f'ROC Curve ({key})')
        ax_roc[i].legend(loc="lower right")


    # PR
    for i, (key, value) in enumerate(y_preds_dict.items()):
        precision, recall, _ = precision_recall_curve(y_true, value)
        pr_auc = auc(recall, precision)
        ax_pr[i].plot(recall, precision, lw=2, label=f'{key} (AUC = {pr_auc:.2f})')
        ax_pr[i].set_xlim([0.0, 1.0])
        ax_pr[i].set_ylim([0.0, 1.05])
        ax_pr[i].set_xlabel('Recall')
        ax_pr[i].set_ylabel('Precision')
        ax_pr[i].set_title('Precision-Recall Curve')
        ax_pr[i].legend(loc="lower left")


    # Scatter
    for i, (key, value) in enumerate(y_preds_dict.items()):
        ax_scatters[i].scatter(y_true, value, alpha=0.5)
        ax_scatters[i].set_xlabel('Ground Truth')
        ax_scatters[i].set_ylabel('Prediction')
        ax_scatters[i].set_title(f'Scatter Plot\n({key})')

        data = {'Ground Truth': pd.to_numeric(y_true).round(r), 'Prediction': pd.to_numeric(value).round(r)}
        df = pd.DataFrame(data)
        df['Ground Truth'] = pd.Categorical(df['Ground Truth'])                 # [07/18 17:22:17][INFO] category.py: 223: Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.
        sns.violinplot(data=df, x='Ground Truth', y='Prediction', ax=ax_scatters[i], hue='Ground Truth', palette='Blues', alpha=0.5, inner='quartile', legend=False)




    # Save the image
    plt.savefig(output_path, bbox_inches='tight')
    print(f'{output_path} <- Saved !')
    plt.close()


# y_true = np.array([0.00277324, 0.02503062, 0.01627464, 0.00823948, 0.03734906,
#        0.01152304, 0.01020121, 0.03656771, 0.00184068, 0.00220103,
#        0.0401891 , 0.00184643, 0.00179759, 0.00472615, 0.00168309,
#        0.00154074, 0.00230096, 0.9, 0.00179441, 0.00879381,
#        0.0011342 , 0.00101321, 0.00113289, 0.00106683, 0.00171589,
#        0.00172409, 0.00356652, 0.00181674, 0.9, 0.00186809,
#        0.9, 0.01095785, 0.0033058 , 0.00237545, 0.00375982,
#        0.00462792, 0.01257213, 0.03409356, 0.02576222, 0.01015837,
#        0.02600421, 0.9, 0.9, 0.00176463, 0.00536698,
#        0.0016775 , 0.00181593, 0.00158619, 0.01288571, 0.9,
#        0.01022568, 0.00549939, 0.00902387, 0.00245089, 0.02045997])
# y_preds_dict = {
#     'Prediction 1': np.array([0.00277324, 0.02503062, 0.01627464, 0.00823948, 0.03734906,
#        0.01152304, 0.01020121, 0.03656771, 0.00184068, 0.00220103,
#        0.0401891 , 0.00184643, 0.00179759, 0.00472615, 0.00168309,
#        0.00154074, 0.00230096, 0.00410278, 0.00179441, 0.00879381,
#        0.0011342 , 0.00101321, 0.00113289, 0.00106683, 0.00171589,
#        0.00172409, 0.00356652, 0.00181674, 0.01123961, 0.00186809,
#        0.00813375, 0.01095785, 0.0033058 , 0.00237545, 0.00375982,
#        0.00462792, 0.01257213, 0.03409356, 0.02576222, 0.01015837,
#        0.02600421, 0.00167327, 0.00220492, 0.00176463, 0.00536698,
#        0.0016775 , 0.00181593, 0.00158619, 0.01288571, 0.01889576,
#        0.01022568, 0.00549939, 0.00902387, 0.00245089, 0.02045997]),
#     'Prediction 2': np.array([0.00277324, 0.02503062, 0.01627464, 0.00823948, 0.03734906,
#        0.01152304, 0.01020121, 0.03656771, 0.00184068, 0.00220103,
#        0.0401891 , 0.00184643, 0.00179759, 0.00472615, 0.00168309,
#        0.00154074, 0.00230096, 0.00410278, 0.00179441, 0.00879381,
#        0.0011342 , 0.00101321, 0.00113289, 0.00106683, 0.00171589,
#        0.00172409, 0.00356652, 0.00181674, 0.01123961, 0.00186809,
#        0.00813375, 0.01095785, 0.0033058 , 0.00237545, 0.00375982,
#        0.00462792, 0.01257213, 0.03409356, 0.02576222, 0.01015837,
#        0.02600421, 0.00167327, 0.00220492, 0.00176463, 0.00536698,
#        0.0016775 , 0.00181593, 0.00158619, 0.01288571, 0.01889576,
#        0.01022568, 0.00549939, 0.00902387, 0.00245089, 0.02045997]),
#     'Prediction 3': np.array([0.00277324, 0.02503062, 0.01627464, 0.00823948, 0.03734906,
#        0.01152304, 0.01020121, 0.03656771, 0.00184068, 0.00220103,
#        0.0401891 , 0.00184643, 0.00179759, 0.00472615, 0.00168309,
#        0.00154074, 0.00230096, 0.00410278, 0.00179441, 0.00879381,
#        0.0011342 , 0.00101321, 0.00113289, 0.00106683, 0.00171589,
#        0.00172409, 0.00356652, 0.00181674, 0.01123961, 0.00186809,
#        0.00813375, 0.01095785, 0.0033058 , 0.00237545, 0.00375982,
#        0.00462792, 0.01257213, 0.03409356, 0.02576222, 0.01015837,
#        0.02600421, 0.00167327, 0.00220492, 0.00176463, 0.00536698,
#        0.0016775 , 0.00181593, 0.00158619, 0.01288571, 0.01889576,
#        0.01022568, 0.00549939, 0.00902387, 0.00245089, 0.02045997])
# }

# create_prediction_table(y_true, y_preds_dict)



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

    