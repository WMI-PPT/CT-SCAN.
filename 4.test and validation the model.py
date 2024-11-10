import matplotlib.pyplot as plt
import numpy as np
import keras
import cv2
import os
from sklearn.model_selection import KFold  # Import KFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
from models import create_model
from load_datas import load_data

label_path = r"C:\Users\jiaoj\Desktop\label.txt"
data_path = r"C:\Users\jiaoj\Desktop\train image"
result_path = r"C:\Users\jiaoj\Desktop\result"

# Read preprocessed data
labels, data_x, data_y, data_z = load_data(label_path, data_path)

# Set KFold parameters
k = 5  # Set 5-fold cross-validation
kf = KFold(n_splits=k, shuffle=True, random_state=42)  # KFold cross-validation, shuffle=True to shuffle the data

# Storage for loss and accuracy of each fold
all_train_loss = []
all_train_acc = []
all_val_loss = []
all_val_acc = []

# Storage for ROC curve data
fpr_all = []
tpr_all = []
roc_auc_all = []

# Storage for Precision-Recall curve data
precision_all = []
recall_all = []
avg_precision_all = []

# K-Fold cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(data_x)):
    print(f"Training fold {fold + 1}/{k}...")

    # Split into training and validation sets
    X_train, X_val = data_x[train_idx], data_x[val_idx]
    Y_train, Y_val = data_y[train_idx], data_y[val_idx]
    Z_train, Z_val = data_z[train_idx], data_z[val_idx]
    labels_train, labels_val = labels[train_idx], labels[val_idx]

    # Build the model
    model, callbacks = create_model()

    # Train the model
    H = model.fit([X_train, Y_train, Z_train], labels_train,
                  epochs=5, batch_size=50, validation_data=([X_val, Y_val, Z_val], labels_val),
                  callbacks=callbacks)

    # Save training results for each fold
    all_train_loss.append(H.history['loss'])
    all_train_acc.append(H.history['accuracy'])
    all_val_loss.append(H.history['val_loss'])
    all_val_acc.append(H.history['val_accuracy'])

    # Save the model after each fold
    print(f"Saving model for fold {fold + 1}...")
    model.save(os.path.join(result_path, f'model_fold_{fold + 1}.keras'))

    # Get the predicted probabilities on the validation set
    y_pred_prob = model.predict([X_val, Y_val, Z_val])

    # Use LabelBinarizer to binarize the labels
    lb = LabelBinarizer()
    lb.fit(labels)
    y_val_bin = lb.transform(labels_val)

    # Calculate ROC curve for each class
    fpr, tpr, _ = roc_curve(y_val_bin.ravel(), y_pred_prob.ravel())
    roc_auc = auc(fpr, tpr)

    fpr_all.append(fpr)
    tpr_all.append(tpr)
    roc_auc_all.append(roc_auc)

    # Calculate Precision-Recall curve for each class
    precision, recall, _ = precision_recall_curve(y_val_bin.ravel(), y_pred_prob.ravel())
    avg_precision = average_precision_score(y_val_bin, y_pred_prob, average='micro')

    precision_all.append(precision)
    recall_all.append(recall)
    avg_precision_all.append(avg_precision)

    # Get the predicted labels for the confusion matrix
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_val_bin, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Fold {fold+1}')

    # Save the confusion matrix plot for each fold
    plt.savefig(os.path.join(result_path, f'confusion_matrix_fold_{fold+1}.jpg'), dpi=800)

# Aggregate training and validation loss and accuracy
train_loss = np.mean(all_train_loss, axis=0)
train_acc = np.mean(all_train_acc, axis=0)
val_loss = np.mean(all_val_loss, axis=0)
val_acc = np.mean(all_val_acc, axis=0)

# Plot the loss and accuracy curves during training
plt.style.use('ggplot')
plt.figure()

# Plot training loss
plt.plot(np.arange(0, 5), train_loss, label="train_loss")
# Plot training accuracy
plt.plot(np.arange(0, 5), train_acc, label="train_acc")
# Plot validation loss
plt.plot(np.arange(0, 5), val_loss, label="val_loss")
# Plot validation accuracy
plt.plot(np.arange(0, 5), val_acc, label="val_acc")

# Set plot title and labels
plt.title("K-Fold Cross Validation: Training and Validation Loss/Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

# Save the plot to a file
plt.savefig(os.path.join(result_path, 'kfold_training_curve.jpg'), dpi=800)  # Save as kfold_training_curve.jpg

# Plot ROC curve for all classes
plt.figure()
for i in range(k):
    plt.plot(fpr_all[i], tpr_all[i], label=f'Fold {i+1} (AUC = {roc_auc_all[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

# Save the ROC curve plot
plt.savefig(os.path.join(result_path, 'roc_curve.jpg'), dpi=800)

# Plot Precision-Recall curve for all classes
plt.figure()
for i in range(k):
    plt.plot(recall_all[i], precision_all[i], label=f'Fold {i+1} (AP = {avg_precision_all[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')

# Save the Precision-Recall curve plot
plt.savefig(os.path.join(result_path, 'precision_recall_curve.jpg'), dpi=800)
