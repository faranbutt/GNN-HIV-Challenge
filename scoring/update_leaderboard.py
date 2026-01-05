import pandas as pd
from datetime import datetime
import os
import re
leaderboard_csv = 'leaderboard.csv'
leaderboard_md = 'leaderboard.md'
readme_file = 'README.md'
try:
    leaderboard = pd.read_csv(leaderboard_csv)
except FileNotFoundError:
    leaderboard = pd.DataFrame(columns=['Rank', 'User', 'Submission File', 'ROC-AUC', 'Date'])
user = os.getenv('PR_USER', 'Anonymous')
submission_file = os.getenv('PR_SUBMISSION', 'submission.csv')
roc_auc = float(os.getenv('PR_SCORE', 0))
new_entry = pd.DataFrame([{
    'Rank': len(leaderboard) + 1,
    'User': user,
    'Submission File': submission_file,
    'ROC-AUC': roc_auc,
    'Date': datetime.now().strftime('%Y-%m-%d')
}])

leaderboard = pd.concat([leaderboard, new_entry], ignore_index=True)
leaderboard = leaderboard.sort_values(by='ROC-AUC', ascending=False).reset_index(drop=True)
leaderboard['Rank'] = leaderboard.index + 1

leaderboard.to_csv(leaderboard_csv, index=False)
table_lines = [
    '| Rank | User | Submission File | ROC-AUC | Date |',
    '|------|------|----------------|---------|------|'
]
for _, row in leaderboard.iterrows():
    table_lines.append(
        f"| {row['Rank']} | {row['User']} | "
        f"{row['Submission File']} | "
        f"{row['ROC-AUC']:.4f} | {row['Date']} |"
    )

leaderboard_table = '\n'.join(table_lines)
with open(leaderboard_md, 'w') as f:
    f.write('## 🏆 Leaderboard\n\n')
    f.write(leaderboard_table + '\n')
with open(readme_file, 'r') as f:
    readme_content = f.read()
pattern = r'(<!-- LEADERBOARD-START -->).*?(<!-- LEADERBOARD-END -->)'
replacement = f'<!-- LEADERBOARD-START -->\n\n{leaderboard_table}\n\n<!-- LEADERBOARD-END -->'

updated_readme = re.sub(pattern, replacement, readme_content, flags=re.DOTALL)

with open(readme_file, 'w') as f:
    f.write(updated_readme)
