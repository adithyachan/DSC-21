#predictions_dict: {true label: topk predictions}

def top_k_accuracy(predictions_dict):
    correct = 0
    total = 0

    for true_label, predictions in predictions_dict.items():
        # Check if the true label is among the top k predictions
        if true_label in predictions:
            correct += 1
        total += 1

    # Compute the top-k accuracy
    accuracy = correct / total if total > 0 else 0
    return accuracy
