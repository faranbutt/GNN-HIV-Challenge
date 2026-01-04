# working/Molecular Graph/scoring/scoring_script.py
import pandas as pd
from sklearn.metrics import roc_auc_score
import sys

submission_file = sys.argv[1]
submission = pd.read_csv(submission_file)
truth = pd.read_csv('data/test_labels.csv')

score = roc_auc_score(truth['target'], submission['probability'])
print(f'Submission ROC-AUC Score: {score:.4f}')