import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import AUROC
import pandas as pd

from data_loader import get_dataloaders
from gnn_models import BaselineGCN, GATGNN, GINGNN

def train_model(model_type='gcn', epochs=15, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = BaselineGCN() if model_type=='gcn' else GATGNN() if model_type=='gat' else GINGNN()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training {model_type.upper()} on {device}...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f}")

    # Inference
    model.eval()
    records = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            probs = torch.sigmoid(model(batch))
            for gid, p in zip(batch.graph_id, probs.cpu().numpy()):
                records.append((gid, float(p)))

    df = pd.DataFrame(records, columns=['graph_id', 'probability'])
    df.sort_values('graph_id', inplace=True)
    df.to_csv(f'submissions/pyg_{model_type}.csv', index=False)

if __name__ == "__main__":
    train_model('gcn')

