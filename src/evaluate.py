from sklearn.metrics import confusion_matrix


def cost_sensitive_evaluation(
    y_true,
    y_pred,
    cost_fp=100,
    cost_fn=10
):
    """
    Evaluate classification results using cost-sensitive metrics.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    cost_fp : float
        Cost of false positive (approving bad borrower)
    cost_fn : float
        Cost of false negative (rejecting good borrower)

    Returns
    -------
    dict
        Dictionary containing confusion matrix and cost metrics
    """

    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    expected_cost = FP * cost_fp + FN * cost_fn
    cost_per_applicant = expected_cost / len(y_true)

    return {
        "confusion_matrix": cm,
        "false_positives": FP,
        "false_negatives": FN,
        "expected_cost": expected_cost,
        "cost_per_applicant": cost_per_applicant
    }

