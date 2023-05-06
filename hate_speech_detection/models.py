
from hate_speech_detection.classifier_model import ClassifierModel
from hate_speech_detection.model_bert import BertBasedClassifer
from hate_speech_detection.model_gpt3_text_infer import GPT3TextInference


models = {
    "Bert": BertBasedClassifer(),
    "GPT-2": ClassifierModel(),
    "GPT-3":GPT3TextInference(),
}
