import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device


# Accuracy per emotion per model (manually copied from test results)
emotion_weights = {
    'happy':    [0.85, 1.0, 0.98],
    'sad':      [0.73, 1.0, 0.98],
    'fear':     [0.64, 0.93, 1.0],
    'neutral':  [0.71, 1.0, 0.99],
    'surprise': [0.84, 1.0, 0.97],
    'anger':    [0.73, 1.0, 0.96],
    'disgust':  [0.71, 0.99, 1.0]
}

# Normalize labels
label_map = {
    'angry': 'anger',
    'anger': 'anger',
    'fear': 'fear',
    'scared': 'fear',
    'sadness': 'sad',
    'sad': 'sad',
    'disgust': 'disgust',
    'happy': 'happy',
    'happiness': 'happy',
    'surprise': 'surprise',
    'suprise': 'surprise',
    'neutral': 'neutral',
    'contempt': 'neutral'
}


# Full confusion matrices for each model
full_conf_matrices = [
    {  # Model 1 - Thao2202/vit-Facial-Expression-Recognition
        'anger':    {'anger': 837, 'disgust': 2, 'fear': 42, 'happy': 9, 'neutral': 31, 'sad': 34, 'surprise': 3},
        'disgust':  {'anger': 5, 'disgust': 98, 'fear': 1, 'happy': 2, 'neutral': 3, 'sad': 1, 'surprise': 1},
        'fear':     {'anger': 27, 'disgust': 4, 'fear': 840, 'happy': 8, 'neutral': 42, 'sad': 84, 'surprise': 19},
        'happy':    {'anger': 15, 'disgust': 0, 'fear': 10, 'happy': 1705, 'neutral': 25, 'sad': 4, 'surprise': 15},
        'neutral':  {'anger': 19, 'disgust': 0, 'fear': 18, 'happy': 29, 'neutral': 1117, 'sad': 46, 'surprise': 4},
        'sad':      {'anger': 40, 'disgust': 1, 'fear': 35, 'happy': 13, 'neutral': 71, 'sad': 1078, 'surprise': 9},
        'surprise': {'anger': 7, 'disgust': 0, 'fear': 37, 'happy': 10, 'neutral': 3, 'sad': 1, 'surprise': 773}
    },
    {  # Model 2
        'anger':    {'anger': 861, 'disgust': 1, 'fear': 28, 'happy': 7, 'neutral': 29, 'sad': 28, 'surprise': 4},
        'disgust':  {'anger': 6, 'disgust': 101, 'fear': 0, 'happy': 1, 'neutral': 2, 'sad': 0, 'surprise': 1},
        'fear':     {'anger': 44, 'disgust': 2, 'fear': 810, 'happy': 8, 'neutral': 37, 'sad': 96, 'surprise': 27},
        'happy':    {'anger': 12, 'disgust': 0, 'fear': 5, 'happy': 1712, 'neutral': 30, 'sad': 3, 'surprise': 12},
        'neutral':  {'anger': 16, 'disgust': 1, 'fear': 6, 'happy': 19, 'neutral': 1156, 'sad': 35, 'surprise': 0},
        'sad':      {'anger': 37, 'disgust': 1, 'fear': 29, 'happy': 7, 'neutral': 55, 'sad': 1108, 'surprise': 10},
        'surprise': {'anger': 8, 'disgust': 0, 'fear': 17, 'happy': 8, 'neutral': 2, 'sad': 4, 'surprise': 792}
    },
    {  # Model 3
        'anger':    {'anger': 827, 'disgust': 4, 'fear': 46, 'happy': 7, 'neutral': 40, 'sad': 30, 'surprise': 4},
        'disgust':  {'anger': 3, 'disgust': 102, 'fear': 3, 'happy': 0, 'neutral': 3, 'sad': 0, 'surprise': 0},
        'fear':     {'anger': 28, 'disgust': 3, 'fear': 872, 'happy': 5, 'neutral': 37, 'sad': 62, 'surprise': 17},
        'happy':    {'anger': 9, 'disgust': 0, 'fear': 14, 'happy': 1699, 'neutral': 32, 'sad': 5, 'surprise': 15},
        'neutral':  {'anger': 15, 'disgust': 2, 'fear': 15, 'happy': 29, 'neutral': 1144, 'sad': 28, 'surprise': 0},
        'sad':      {'anger': 28, 'disgust': 1, 'fear': 45, 'happy': 10, 'neutral': 65, 'sad': 1088, 'surprise': 10},
        'surprise': {'anger': 9, 'disgust': 0, 'fear': 40, 'happy': 8, 'neutral': 2, 'sad': 3, 'surprise': 769}
    }
]

from collections import defaultdict

# Generates a nested dictionary confusion matrix for a single model
def generate_confusion_matrix_dict(true_labels, pred_labels):
    """
    Generates a nested dictionary confusion matrix:
    conf[true_label][pred_label] = count

    Both inputs should be lists of equal length.
    """
    conf = defaultdict(lambda: defaultdict(int))
    for true, pred in zip(true_labels, pred_labels):
        true_norm = label_map.get(true.lower(), true.lower())
        pred_norm = label_map.get(pred.lower(), pred.lower())
        conf[true_norm][pred_norm] += 1
    return conf

def get_weighted_majority_vote(predictions):
    weighted_vote_counter = {}
    for i, pred in enumerate(predictions):
        normalized = label_map.get(pred.lower(), pred.lower())
        weight = emotion_weights.get(normalized, [0, 0, 0])[i]
        weighted_vote_counter[normalized] = weighted_vote_counter.get(normalized, 0) + weight
    return max(weighted_vote_counter, key=weighted_vote_counter.get)

def get_confusion_matrix_vote(predictions):
    """
    Uses full confusion matrices to trust predictions more if historically accurate.

    Parameters:
    - predictions: list of predicted class labels (one from each model)
    - conf_matrices: list of dict-of-dict confusion matrices for each model:
      e.g., conf[model][true_label][pred_label] = count

    Returns:
    - label with highest adjusted probability
    """
    vote_scores = {}

    for i, pred in enumerate(predictions):
        norm_pred = label_map.get(pred.lower(), pred.lower())
        conf = full_conf_matrices[i]
        numerator = conf.get(norm_pred, {}).get(norm_pred, 0)
        denominator = sum(conf.get(true, {}).get(norm_pred, 0) for true in conf)
        prob = (numerator / denominator) if denominator > 0 else 0
        vote_scores[norm_pred] = vote_scores.get(norm_pred, 0) + prob

    return max(vote_scores, key=vote_scores.get)


predictions = ['happy', 'neutral', 'happy']  # model predictions
result = get_confusion_matrix_vote(predictions)
print(result)