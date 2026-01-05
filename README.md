# GNN HIV Challenge: Molecular Graph Classification for Drug Discovery

## 🚀 Project Overview
The **GNN HIV Challenge** is a benchmark for graph neural networks on molecular property prediction.  
The goal is to classify molecular graphs to predict anti-HIV activity using **GCN, GAT, and GIN models**.  

**Dataset:**
- 5,000 molecular graphs  
  - 3,000 training  
  - 1,000 test  
- Features: Node-level descriptors, adjacency matrices  
- Class distribution: 25% positive, 75% negative  

**Evaluation Metric:** ROC-AUC  

**Baseline Performance:** ~0.76 ROC-AUC  

---
## 📁 Repository Structure
```

├── 📁 .github
│   └── 📁 workflows
│       └── ⚙️ score_submission.yml
├── 📁 data
│   ├── 📄 graph_structures.pkl
│   ├── 📄 node_features.pkl
│   ├── 📄 test.csv
│   ├── 📄 test_labels.csv
│   └── 📄 train.csv
├── 📁 scoring
│   ├── 🐍 scoring_script.py
│   └── 🐍 update_leaderboard.py
├── 📁 starter_code
│   ├── 🐍 baseline.py
│   ├── 🐍 data_loader.py
│   ├── 🐍 gnn_models.py
│   └── 🐍 train.py
├── ⚙️ .gitignore
├── 📝 README.md
├── ⚙️ pyproject.toml
└── 📄 requirements.txt

```

## 🏆 Leaderboard

<!-- LEADERBOARD-START -->

| Rank | User | Submission File | ROC-AUC | Date |
|------|------|----------------|---------|------|
| 1 | faranbutt | submissions/submission_samples.csv | 0.4747 | 2026-01-05 |
| 2 | faranbutt | submissions/submission_sampless.csv | 0.4747 | 2026-01-05 |

<!-- LEADERBOARD-END -->
