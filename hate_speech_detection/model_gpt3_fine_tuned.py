import openai
import numpy as np
from hate_speech_detection.classifier_model import ClassifierModel


class GPT3FinedTunedInference(ClassifierModel):
    def __init__(self):
        super(GPT3FinedTunedInference, self).__init__()
        self.ft_model = 'davinci:ft-personal-2023-05-07-12-34-28'
    # Define the function to detect hate speech

    def detect_hate_speech(self, sentence):
        # Classify the sentence as either hate speech or not hate speech using GPT-3
        prompt = f"{sentence}\n\n###\n\n"
        print(f'prompt = {prompt}')
        response = openai.Completion.create(
            model=self.ft_model,
            prompt=prompt,
            temperature=0,
            max_tokens=1,  # Increase max_tokens to include the entire classification
        )
        # print(f'response={response}')
        classification = str(response.choices[0].text.strip())
        print(f'classification={classification}')
        if '1' in classification:
            return True
        else:
            return False

    def detect_social_bias(self, sentence):
        return False
        # Classify the sentence as either hate speech or not hate speech using GPT-3
        prompt = f"Is following sentence targeting any social group, nation, race, ethnicity, gender, religion ? reply in yes or no :\n{sentence}\n"
        print(f'prompt = {prompt}')
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=0,
            max_tokens=5,  # Increase max_tokens to include the entire classification
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        # print(f'response={response}')
        classification = str(response.choices[0].text.strip())
        print(f'classification={classification}')
        if 'yes' in classification.lower():
            return True
        else:
            return False

    def predict(self, inputs):
        if np.isscalar(inputs):
            return self.detect_hate_speech(inputs), self.detect_social_bias(inputs)
        else:
            return [self.detect_hate_speech(inp) for inp in inputs], [self.detect_social_bias(inp) for inp in inputs]
