""""" functions """""
import numpy as np
import torch
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


""""" for classification task """""

def train_epoch(device, model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for data in loader:  # data = batch of graphs
        data = data.to(device)
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index, data.batch, data.clinical_x)

        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()

        total += data.num_graphs

    return total_loss / total, correct / total


def evaluate(device, model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_targets, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _ = model(data.x, data.edge_index, data.batch, data.clinical_x)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs

            probs = torch.softmax(out, dim=1)
            pred = out.argmax(dim=1)  # class with the highest prob
            correct += (pred == data.y).sum().item()

            all_targets.extend(data.y.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            total += data.num_graphs

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    num_classes_in_target = len(np.unique(all_targets))
    if num_classes_in_target == 2:
        auc_val = roc_auc_score(all_targets, all_probs[:, 1])
    elif num_classes_in_target > 2:  # Multi-Class AUC --> OvR strategy (One-vs-Rest)
        auc_val = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    else:
        auc_val = 0.0

    metrics = {
        'acc': np.mean(all_preds == all_targets),  # acc = correct/total
        'precision': precision_score(all_targets, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_targets, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_targets, all_preds, average='macro', zero_division=0),
        'auc': auc_val,
    }

    return total_loss/total, metrics


""""" for survival task """""

def train_epoch_survival(device, model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # output = Log-Hazard (relative risk)
        risk_pred, _ = model(data.x, data.edge_index, data.batch, data.clinical_x)

        # Cox Loss based on (risk_pred, event, time)
        loss = criterion(risk_pred, data.y_event, data.y_time)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        total_samples += data.num_graphs

    return total_loss / total_samples


def evaluate_survival(device, model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_risk_scores = []
    all_times = []
    all_events = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            risk_pred, _ = model(data.x, data.edge_index, data.batch, data.clinical_x)

            loss = criterion(risk_pred, data.y_event, data.y_time)
            total_loss += loss.item() * data.num_graphs

            all_risk_scores.extend(risk_pred.squeeze(-1).cpu().numpy())
            all_times.extend(data.y_time.cpu().numpy())
            all_events.extend(data.y_event.cpu().numpy())

    all_risk_scores = np.array(all_risk_scores)
    all_times = np.array(all_times)
    all_events = np.array(all_events)

    # higher risk_score means higher risk (lower survival time), we pass the opposite (-risk_scores) to concordance_index
    c_index = concordance_index(
        event_times=all_times,
        predicted_scores=-all_risk_scores,
        event_observed=all_events
    )

    metrics = {
        'c_index': c_index
    }

    return total_loss / len(all_times), metrics
