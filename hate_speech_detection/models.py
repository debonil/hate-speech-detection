
from hate_speech_detection.classifier_model import ClassifierModel
from hate_speech_detection.model_bert import BertBasedClassifer
from hate_speech_detection.model_bert_adv import BertAdvancedClassifer
from hate_speech_detection.model_gpt3_fine_tuned import GPT3FinedTunedInference
from hate_speech_detection.model_gpt3_text_infer import GPT3TextInference


models = {
    "GPT-3 Zero Shot": GPT3TextInference(),
    "XLM-roberta Hi En finetuned": BertBasedClassifer(),
    "BanglaBert finetuned-sc": BertAdvancedClassifer(model_type="BERT", model_name="ka05ar/banglabert-sentiment", label_maper={'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'}),
    "Hinglish finetuned-sc": BertAdvancedClassifer(model_type="BERT", model_name="ganeshkharad/gk-hinglish-sentiment", label_maper={'LABEL_0': 'neutral', 'LABEL_1': 'negative', 'LABEL_2': 'positive'}),
    "MuRIL finetuned-sc": BertAdvancedClassifer(model_type="BERT", model_name="IIIT-L/muril-base-cased-finetuned-TRAC-DS", label_maper={'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}),
    "Hing-roberta finetuned-code-mixed": BertAdvancedClassifer(model_type="ROBERTA", model_name="IIIT-L/hing-roberta-finetuned-code-mixed-DS", label_maper={'LABEL_0': 'neutral', 'LABEL_1': 'positive', 'LABEL_2': 'negative'}),
    "Bangla XLM-R finetuned-Bangla": BertAdvancedClassifer(model_type="ROBERTA", model_name="Arunavaonly/Bangla_multiclass_sentiment_analysis_model", label_maper={'LABEL_0': 'neutral', 'LABEL_1': 'positive', 'LABEL_2': 'negative'}),
    # "GPT-3 FineTuned": GPT3FinedTunedInference(),
}
