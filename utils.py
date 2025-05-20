import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device


# Accuracy per emotion per model (manually copied from test results)
emotion_weights = {
    'happy':    [1.0, 0.0, 0.9],
    'sad':      [0.5, 0.6, 0.7],
    'fear':     [0.6, 0.9, 0.7],
    'neutral':  [0.9, 0.6, 0.5],
    'surprise': [1.0, 0.8, 1.0],
    'anger':    [0.9, 0.8, 0.9],
    'disgust':  [0.6, 1.0, 1.0]
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

def get_weighted_majority_vote(predictions):
    weighted_vote_counter = {}
    for i, pred in enumerate(predictions):
        normalized = label_map.get(pred.lower(), pred.lower())
        weight = emotion_weights.get(normalized, [0, 0, 0])[i]
        weighted_vote_counter[normalized] = weighted_vote_counter.get(normalized, 0) + weight
    return max(weighted_vote_counter, key=weighted_vote_counter.get)