
import numpy as np
from hate_speech_detection.classifier_model import ClassifierModel
from happytransformer import HappyTextClassification


class BertBasedClassifer(ClassifierModel):
    def __init__(self):
        super(BertBasedClassifer, self).__init__()

        self.happy_tc = HappyTextClassification(
            model_type="ROBERTA", model_name="AryPratap/XLM-roberta-HIEN-Sentiment-Analysis", num_labels=3)

    def detect_hate_speech(self, sentence):
        result = self.happy_tc.classify_text(sentence)
        #print(result)
        return result.label
        if 'negative' == result.label:
            return True
        else:
            return False

    def detect_social_bias(self, sentence):
        # not implemented
        return False

    def predict(self, inputs):
        if np.isscalar(inputs):
            return self.detect_hate_speech(inputs), self.detect_social_bias(inputs)
        else:
            return [self.detect_hate_speech(inp) for inp in inputs], [self.detect_social_bias(inp) for inp in inputs]
