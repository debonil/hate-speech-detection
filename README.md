
# Hate Speech and Social Bias Detection using GPT3

CSL7030 | DLOps 22-23 Sem II | Term Project

**Paper:**

Ke-Li Chiu, Annie Collins, Rohan Alexander, ‘Detecting Hate Speech with GPT-3’, arXiv [cs.CL]. 2022.

Reference : [https://arxiv.org/abs/2103.12407]

**Brief Description:**

Sophisticated language models such as OpenAI's GPT-3 can generate hateful text that targets marginalized groups. Given this capacity, we are interested in whether large language models can be used to identify hate speech and classify text as sexist or racist. We use GPT-3 to identify sexist and racist text passages with zero-, one-, and few-shot learning. We find that with zero- and one-shot learning, GPT-3 can identify sexist or racist text with an average accuracy between 55 per cent and 67 per cent, depending on the category of text and type of learning. With few-shot learning, the model's accuracy can be as high as 85 per cent. Large language models have a role to play in hate speech detection, and with further development they could eventually be used to counter hate speech.

In addition to the paper implementation, we ahave added one **GPT3 finetuned davinchi** model for Hate Speech **Classification** and **Social Bias** Detection

## Authors

- [Debonil Ghosh [M21AIE225] ](<https://www.github.com/debonil>)
- [Saurav Chowdhury [M21AIE256] ](<https://www.github.com/sauraviitj>)
- [Ravi Shankar Kumar [M21AIE247]](<https://www.github.com/rsk-iitj>)

## Usage

    1. Clone repository

    2. pip install -r requirement.txt

    3. run notebooks/data_prepare.ipynb to finetune GPT model and link it to GPT3FineTune.py

    4. streamlit run streamlit_app/app.py

## Results of GPT3 Fine Tuned model

Accuracy: **89.362%**

F1 Score: **89.383%**

**Classwise Accuracy Score:**
|     Hate |   Non-Hate |
|---------:|-----------:|
| 0.903846 |   0.885496 |

![confusion_matrix](results/confusion_mat_GPT3%20Fine%20tune.png)

## Comaprison of three models

| Model           |   Accuracy |   F1 Score |   Precision |   Recall |   ROC AUC Score |
|:----------------|-----------:|-----------:|------------:|---------:|----------------:|
| Bert            |   0.702128 |   0.701038 |    0.722628 | 0.755725 |        0.69517  |
| GPT-3 Zero Shot |   0.608511 |   0.580911 |    0.854545 | 0.358779 |        0.640928 |
| GPT-3 FineTuned |   0.889362 |   0.889616 |    0.92     | 0.877863 |        0.890854 |

## Demo

![gpt3_finetuned_english_churchil](results/streamlit_gpt3_finetuned_english_churchil.PNG?raw=true)

![gpt3_finetuned_english_netaji](results/streamlit_gpt3_finetuned_english_netaji.PNG?raw=true)

![gpt3_finetuned_hindi](results/streamlit_gpt3_finetuned_hindi.PNG?raw=true)

![gpt3_finetuned_french](results/streamlit_gpt3_finetuned_french.PNG?raw=true)

## Weights and Biases reports

![wandb hyper param tuning](results/wandb_hyper_tuning.PNG?raw=true)

![wandb reports](results/wandb_reports.PNG?raw=true)

## Conclusion

GPT3 Fine Tuned Model worked best
