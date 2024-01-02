# Import necessary libraries
import streamlit as st
import os
import sys
module_path = os.path.abspath(os.path.join(''))
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from hate_speech_detection.models import models


# Streamlit app


def main():
    st.set_page_config(
        page_title="Sentiment Analysis using LLMs", layout="wide")
    st.title("Feedback Analysis using LLMs")
    # Model selection
    model_name = st.sidebar.selectbox("Select a model", list(models.keys()))

    # Input text
    user_input = st.text_area("Input text for analysis", "")

    # Process input text
    if st.button("Analyze"):
        if user_input:
            model = models[model_name]

            sentiment_result, aspect_result = model.predict(user_input)

            # Display results
            # st.write("### Hate Speech Analysis")
            st.write(
                f"### Sentiment detected: {'Negative' if sentiment_result else 'Positive'}")

            if model_name == 'GPT-3 Zero Shot':
                st.write("### Aspect Analysis")
                data = {
                    'Aspects': list(aspect_result.keys()),
                    'Age': [25, 30, 22],
                    'City': ['New York', 'San Francisco', 'Seattle']
                }
                st.table(aspect_result)
            # st.write(
            #    f"Social bias detected: {'Yes' if social_bias_result else 'No'}")
        else:
            st.error("Please provide input text for analysis.")


if __name__ == "__main__":
    main()
