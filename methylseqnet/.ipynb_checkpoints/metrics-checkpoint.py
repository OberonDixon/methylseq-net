from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, auc, f1_score

def all_metrics(targets,probabilities):
    predictions = (probabilities >= 0.5).astype('int')
    
    # Calculate accuracy, precision, and recall
    accuracy = accuracy_score(targets, predictions)
    precision_overall = precision_score(targets, predictions,zero_division=0)
    recall_overall = recall_score(targets, predictions,zero_division=0)
    # Calculate precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(targets, probabilities)
    # Calculate the area under the precision-recall curve
    pr_auc = auc(recall, precision)
    f1 = f1_score(targets, predictions, average='binary')
    
    return {
            "accuracy":accuracy,
            "precision":precision_overall,
            "recall":recall_overall,
            "f1":f1,
            "prc_auc":pr_auc,
        }