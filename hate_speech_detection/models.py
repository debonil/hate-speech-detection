
from hate_speech_detection.classifier_model import ClassifierModel
from hate_speech_detection.model_gpt3_text_infer import GPT3TextInference


models = {
    "Bert": ClassifierModel(),
    "GPT": ClassifierModel(),
    "GPT-2": ClassifierModel(),
    "GPT-3":GPT3TextInference(),
}
