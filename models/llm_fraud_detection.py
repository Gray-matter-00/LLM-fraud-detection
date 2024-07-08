import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load data
df = pd.read_csv("../data/sample_transactions.csv")

# Initialize the pipeline
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Example function to detect fraud using LLM
def detect_fraud(transaction_description):
    result = nlp(transaction_description)
    return result

# Apply the function to transaction descriptions
df['fraud_probability'] = df['description'].apply(lambda x: detect_fraud(x)[0]['score'])

# Save results
df.to_csv("../data/llm_fraud_results.csv", index=False)
