
from hate_speech_detection.classifier_model import ClassifierModel
from hate_speech_detection.model_bert import BertBasedClassifer
from hate_speech_detection.model_gpt3_fine_tuned import GPT3FinedTunedInference
from hate_speech_detection.model_gpt3_text_infer import GPT3TextInference


models = {
    "Bert": BertBasedClassifer(),
    "GPT-3": GPT3TextInference(),
    "GPT-3 FineTuned": GPT3FinedTunedInference(),
}
