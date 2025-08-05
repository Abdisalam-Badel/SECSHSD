from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

"""
We use the test data, with label and text, to  cross-check against the training dataset, with text and predicted label.
"""
my_dataset = pd.read_csv('data/test_data.csv', encoding='latin-1')
results = pd.read_csv('Topk-values/gpt-3.5-turbo4.csv', encoding='latin-1')
logger.info("Classification Report:")
report = classification_report(
    my_dataset['Label'],
    results['predicted_Class'],
    labels=[0, 1],
    zero_division=1,
    digits=4
)

"""
Print Result
"""
logger.info("Overall performance printed")
logger.info(report)

"""
Confusion matrix
"""
logger.info("Confusion matrix created")
conf_matrix = confusion_matrix(
    my_dataset['Label'],
    results['predicted_Class'],
    labels=[0, 1]
)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix_normalized,
    display_labels=['normal', 'hate']
)
disp.plot(cmap='Reds', values_format='.2f')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
for text in disp.text_.ravel():
    text.set_fontsize(16)
plt.savefig("my_result/hybrid/confi.pdf", dpi=600, bbox_inches='tight')
plt.show()
