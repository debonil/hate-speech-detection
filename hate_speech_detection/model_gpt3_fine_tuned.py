import openai
import numpy as np
from hate_speech_detection.classifier_model import ClassifierModel


from openai import OpenAI


class GPT3FinedTunedInference(ClassifierModel):
    def __init__(self):
        super(GPT3FinedTunedInference, self).__init__()
        self.ft_model = 'ft:davinci-002:personal::8PaksLJY'
        self.client = OpenAI()
    # Define the function to detect hate speech

    def detect_hate_speech(self, sentence):
        # Classify the sentence as either hate speech or not hate speech using GPT-3
        prompt = f"{sentence}\n\n###\n\n"
        print(f'prompt = {prompt}')
        response = self.client.completions.create(
            model=self.ft_model,
            prompt=prompt,
            temperature=0,
            max_tokens=4,  # Increase max_tokens to include the entire classification
        )
        print(f'response={response}')
        classification = str(response.choices[0].text.strip())
        print(f'classification={classification}')
        if '0' in classification:
            return True
        else:
            return False

    def detect_social_bias(self, sentence):
        return False

    def predict(self, inputs):
        if np.isscalar(inputs):
            return self.detect_hate_speech(inputs), self.detect_social_bias(inputs)
        else:
            return [self.detect_hate_speech(inp) for inp in inputs], [self.detect_social_bias(inp) for inp in inputs]
