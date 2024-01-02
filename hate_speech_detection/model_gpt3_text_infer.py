import openai
import numpy as np
from hate_speech_detection.classifier_model import ClassifierModel

from openai import OpenAI
import json


class GPT3TextInference(ClassifierModel):
    def __init__(self):
        super(GPT3TextInference, self).__init__()
        self.client = OpenAI()

    def detect_feedback_sentiment(self, sentence):
        # Classify the sentence as either hate speech or not hate speech using GPT-3
        prompt = f"What is the sentiment of this multilingual sentence? reply in positive, negative, neutral :\n{sentence}\n"
        # print(f'prompt = {prompt}')
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,  # Increase max_tokens to include the entire classification
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        classification = str(response.choices[0].message.content.strip())
        # print(classification)
        return classification if classification in ['positive', 'negative'] else 'neutral'
        if 'yes' in classification.lower():
            return True
        else:
            return False

    def get_aspect_sentiment(self, review):
        return False
        prompt = f"Analyze the provided review delimited by <> and extract product-related aspects, sentiments (positive/negative/neutral), and justifications. Format the result as a JSON object with the following structure: a key named 'aspects' containing a nested object with keys for each aspect, and each aspect key having an object with 'sentiment' and 'justification' keys. Exclude non-product discussions. If no sentiment is expressed for an aspect, return 'neutral' as the sentiment and 'No information provided' as the justification. If information isn't present, use 'unknown' as the value. If information can't be retrieved, return an empty JSON object. Review: <{review}>"
        # print(f'prompt = {prompt}')
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result = json.loads(response.choices[0].message.content)
        print(result)
        return result

    def predict(self, inputs):
        if np.isscalar(inputs):
            return self.detect_feedback_sentiment(inputs), self.get_aspect_sentiment(inputs)
        else:
            return [self.detect_feedback_sentiment(inp) for inp in inputs], [self.get_aspect_sentiment(inp) for inp in inputs]
