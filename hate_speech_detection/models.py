
from hate_speech_detection.classifier_model import ClassifierModel


models = {
    "Bert": ClassifierModel(),
    "GPT": ClassifierModel(),
    "GPT-2": ClassifierModel(),
}
