from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, classification_report

def compute_metrics(y_true, y_pred, target_names=['0: No DR', '1: Mild', '2: Moderate', '3: Severe', '4: Proliferative']):
    """
    Computes all classification metrics required for DR grading.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    return {
        'accuracy': acc,
        'f1_macro': f1,
        'precision': precision,
        'recall': recall,
        'qwk': qwk,
        'classification_report': report
    }
