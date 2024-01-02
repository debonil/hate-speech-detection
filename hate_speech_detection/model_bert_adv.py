
import numpy as np
from hate_speech_detection.classifier_model import ClassifierModel
from happytransformer import HappyTextClassification
# Use a pipeline as a high-level helper
from transformers import pipeline


class BertAdvancedClassifer(ClassifierModel):
    def __init__(self,model_type,model_name, label_maper):
        super(BertAdvancedClassifer, self).__init__()
        self.label_maper=label_maper
        self.happy_tc = HappyTextClassification(model_type=model_type, model_name=model_name,
                                                num_labels=3, use_auth_token="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")

    def detect_hate_speech(self, sentence):
        result = self.happy_tc.classify_text(sentence)
        #print(result)
        return result.label.lower() if self.label_maper == None else self.label_maper[result.label]
        if 'NEGATIVE' == result.label:
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
