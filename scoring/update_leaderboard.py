import pandas as pd
from datetime import datetime
import os

leaderboard_csv = 'leaderboard.csv'
leaderboard_md = 'leaderboard.md'


try:
    leaderboard = pd.read_csv(leaderboard_csv)
except FileNotFoundError:
    leaderboard = pd.DataFrame(columns=['Rank', 'User', 'Submission File', 'ROC-AUC', 'Date'])


user = os.getenv('PR_USER', 'Anonymous')
submission_file = os.getenv('PR_SUBMISSION', 'submission.csv')
roc_auc = float(os.getenv('PR_SCORE', 0))


new_entry = pd.DataFrame([{
    'Rank': len(leaderboard)+1,
    'User': user,
    'Submission File': submission_file,
    'ROC-AUC': roc_auc,
    'Date': datetime.now().strftime('%Y-%m-%d')
}])
leaderboard = pd.concat([leaderboard, new_entry], ignore_index=True)
leaderboard = leaderboard.sort_values(by='ROC-AUC', ascending=False).reset_index(drop=True)
leaderboard['Rank'] = leaderboard.index + 1

leaderboard.to_csv(leaderboard_csv, index=False)

with open(leaderboard_md, 'w') as f:
    f.write('# GNN HIV Challenge Leaderboard\n\n')
    f.write('| Rank | User | Submission File | ROC-AUC | Date |\n')
    f.write('|------|------|----------------|---------|------|\n')
    for _, row in leaderboard.iterrows():
        f.write(f"| {row['Rank']} | {row['User']} | {row['Submission File']} | {row['ROC-AUC']:.4f} | {row['Date']} |\n")
