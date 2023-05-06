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
        page_title="Hate Speech and Social Bias Detection", layout="wide")
    st.title("Hate Speech and Social Bias Detection")

    # Model selection
    model_name = st.sidebar.selectbox("Select a model", list(models.keys()))

    # Input text
    user_input = st.text_area("Input text for analysis", "")

    # Process input text
    if st.button("Analyze"):
        if user_input:
            model = models[model_name]

            hate_speech_result, social_bias_result = model.predict(user_input)

            # Display results
            st.write("### Hate Speech Analysis")
            st.write(
                f"Hate speech detected: {'Yes' if hate_speech_result else 'No'}")

            st.write("### Social Bias Analysis")
            st.write(
                f"Social bias detected: {'Yes' if social_bias_result else 'No'}")
        else:
            st.error("Please provide input text for analysis.")


if __name__ == "__main__":
    main()
