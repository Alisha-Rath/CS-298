from flask import Flask, request, render_template, jsonify
import pickle
import joblib
import os
import shap
from transformers_interpret import SequenceClassificationExplainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
import torch
from peft import PeftModel, PeftConfig
from xgboost import DMatrix
import re
from shap import KernelExplainer
from sklearn.preprocessing import LabelEncoder
# Initialize the Flask app
app = Flask(__name__)
matplotlib.use('Agg')  # Non-interactive backend
# Define a global variable to map model outputs to labels (replace this with your actual mapping)
id2label = {0: "First Party Wins", 1: "First Party Loses"}

# Initialize global model and tokenizer
model = None
tokenizer = None

# # Load the pre-trained model
# MODEL_PATH = "models/model3.pkl"
# with open(MODEL_PATH, "rb") as model_file:
#     model = pickle.load(model_file)

def preprocess_text(text):
    # Defines a function to preprocess text by removing unnecessary HTML tags.

    # remove <p> tag
    text = text.replace('<p>', '')
    text = text.replace('</p>', '')
    # Removes occurrences of the HTML `<p>` tag from the input text.

    return text
    # Returns the cleaned text.

def model_wrapper(input_ids):
    # Convert the list of strings to numerical token IDs using the tokenizer
    # input_ids = tokenizer.convert_tokens_to_ids(input_ids) # Removed this line
    # Modified to handle multiple sequences for SHAP explainer
    input_ids = [tokenizer.encode(text) for text in input_ids]  # Encode each text in the input_ids list
    max_len = max(len(ids) for ids in input_ids)  # Determine the maximum length of the tokenized inputs
    # Pad each sequence to the maximum length to ensure uniform input dimensions
    padded_input_ids = [ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids in input_ids]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = torch.tensor(padded_input_ids, dtype=torch.long).to(device)  # Convert to torch tensor and move to GPU
    logits = model(input_ids).logits  # Get the logits from the model
    return logits.cpu().detach().numpy()  # Return logits as a NumPy array

def text_masker(text, mask):
    tokens = tokenizer.tokenize(text)  # Tokenize the input text
    # Create masked tokens by replacing tokens with [MASK] where mask is False
    masked_tokens = [tokens[i] if mask[i] else "[MASK]" for i in range(len(tokens))]
    masked_text = tokenizer.convert_tokens_to_string(masked_tokens)  # Convert tokens back to string
    return masked_text

def get_attention_weights(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(text, return_tensors="pt").to(device)  # Tokenize the input text and move to GPU
    outputs = model(**inputs, output_attentions=True)  # Get the model outputs including attention weights
    return outputs.attentions[-1]  # Return the last layer's attention weights

def explain_prediction_shap(text, predicted_label):
    masker = shap.maskers.Text(tokenizer, mask_token="[MASK]")
    explainer = shap.Explainer(model_wrapper, masker=masker, output_names=list(id2label.values()))

    shap_values = explainer([text])
    shap_html = shap.plots.text(shap_values[0], display=False)  # Save as HTML if needed
    # Extract word impacts and SHAP values
    # words = shap_values.data[0]  # Assuming single instance prediction
    # shap_values_list = shap_values.values[0][:, 1]  # Get SHAP values for the positive class

    # # Combine words and SHAP values
    # word_impacts = [(word, value) for word, value in zip(words, shap_values_list)]

    # # Sort by absolute impact (optional)
    # word_impacts_sorted = sorted(word_impacts, key=lambda x: abs(x[1]), reverse=True)

    # # Create a bar plot
    # words, impacts = zip(*word_impacts_sorted)  # Unpack words and impacts
    # fig, ax = plt.subplots(figsize=(12, 6))  # Create figure and axes

    # # Create the bar plot
    # ax.barh(words, impacts, color=['red' if impact > 0 else 'blue' for impact in impacts]) 
    # ax.set_title('Word Impacts')
    # ax.set_xlabel('SHAP Value')

    # # Save the figure to a buffer and encode as base64
    # buffer = io.BytesIO()
    # fig.savefig(buffer, format='png', bbox_inches='tight')
    # buffer.seek(0)
    # image_data = base64.b64encode(buffer.read()).decode('utf-8')
    # plt.close(fig)  # Close the figure to release resources

    # # Create HTML with embedded image
    # shap_html = f'<img src="data:image/png;base64,{image_data}" alt="SHAP Bar Plot">'

    shap_data = display_shap_word_impact(text, shap_values, tokenizer)

    # Return all relevant data as a dictionary
    return {
        "shap_values": shap_values,
        "formatted_impacts": shap_data['word_impacts_sorted'],
        "avg_positive": shap_data['avg_positive'],
        "avg_negative": shap_data['avg_negative'],
        "shap_plot": shap_html  # Include the SHAP plot HTML
    }
    
def explain_prediction_shap_classifier(text, model, vectorizer):
    """Explains the prediction using SHAP for XGBoost."""
    # Transform the input text into a vector using the TF-IDF vectorizer
    input_vector = vectorizer.transform([text])

    # Create a TreeExplainer for the XGBoost model
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values for the input vector
    shap_values = explainer.shap_values(input_vector)
    # Get feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()
    word_shap_values = {}
    for i, feature_name in enumerate(feature_names):
        word_shap_values[feature_name] = shap_values[0][i]  # Assuming single instance prediction

    feature_to_token_mapping = {}
    for i, feature_name in enumerate(feature_names):
       # Example: assuming feature names are single words
       feature_to_token_mapping[feature_name] = feature_name 

    # Get nonzero indices of features present in the input text
    nonzero_indices = input_vector.nonzero()[1]  # Get indices of nonzero TF-IDF values
    present_features = feature_names[nonzero_indices]  # Words present in input text
    present_shap_values = shap_values[0][nonzero_indices]  # Only SHAP values for those words
    present_tfidf_values = input_vector.toarray()[0][nonzero_indices]  # Their TF-IDF values
    
    # Create a SHAP Explanation object (Only for words in text)
    explanation = shap.Explanation(
        values=present_shap_values,
        base_values=explainer.expected_value,
        data=present_tfidf_values,  # Provide TF-IDF values for the words
        feature_names=present_features
    )

    # Generate SHAP bar plot
    shap.plots.bar(explanation, show=False)  # Only plots selected words

    # Get the current figure
    fig = plt.gcf()

    # Save the figure to a buffer and encode as base64
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_data = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)  # Close the figure to release resources

    # Create HTML with embedded image
    shap_html = f'<img src="data:image/png;base64,{image_data}" alt="SHAP Summary Plot">'

    shap_data = display_shap_word_impact_classifier(shap_values, text, vectorizer)

    # Return all relevant data as a dictionary
    return {
        "shap_values": shap_values,
        "formatted_impacts": shap_data['word_impacts_sorted'],
        "avg_positive": shap_data['avg_positive'],
        "avg_negative": shap_data['avg_negative'],
        "shap_plot": shap_html  # Optionally include the SHAP plot
    }    
    

def display_shap_word_impact(input_text, shap_values, tokenizer):
    """
    Extracts and displays the words and their SHAP values.

    Args:
        shap_values (shap.Explanation): SHAP values object.

    Returns:
        None: Prints words and their impacts.
    """
    # Split the input text by spaces, trim whitespace, and remove special symbols using regex
    words = [re.sub(r'[^A-Za-z0-9]+', '', word.strip()) for word in input_text.split()]

    print("--------- words ------")
    print(words)
    # Trim each token in shap_values.data[0] and remove special symbols
    shap_words = [re.sub(r'[^A-Za-z0-9]+', '', word.strip()) for word in shap_values.data[0]]
    
    print("--------- shap_words ------")
    print(shap_words)
    # Extract SHAP values for the words (assuming one SHAP value per word)
    shap_values_list = shap_values.values[0][:, 1]  # Corresponding SHAP values for the words

    # Ensure we only pick words that appear in both the input text and the shap_values_list
    word_impacts = [
        (words[i], shap_values_list[i], "positively" if shap_values_list[i] > 0 else "negatively")
        for i in range(len(words))
        if words[i] in shap_words  # Check if the word is in the shap_values list
    ]

    # Sort words by their absolute SHAP value (impact)
    word_impacts_sorted = sorted(word_impacts, key=lambda x: abs(x[1]), reverse=True)

    # Return structured data (word impacts with direction)
    return {
        "word_impacts_sorted": word_impacts_sorted,
        "avg_positive": calculate_word_impact_summary(shap_values, tokenizer)[0],
        "avg_negative": calculate_word_impact_summary(shap_values, tokenizer)[1]
    }

def display_shap_word_impact_classifier(shap_values, text, vectorizer):
    """
    Extracts and displays the words and their SHAP values.

    Args:
        shap_values (shap.Explanation): SHAP values object.
        text (str): Original text for extracting words.
        vectorizer (object): TF-IDF vectorizer object.

    Returns:
        dict: Structured data containing word impacts, average positive, and average negative impacts.
    """
    import re

    # Extract feature names from the vectorizer
    feature_names = vectorizer.get_feature_names_out()

    # Create a dictionary mapping words to their SHAP values
    word_shap_values = {}
    for i, feature_name in enumerate(feature_names):
        word_shap_values[feature_name] = shap_values[0][i]

    # Split the input text by spaces, trim whitespace, and remove special symbols using regex
    words = [re.sub(r'[^A-Za-z0-9]+', '', word.strip()) for word in text.split()]

    # Calculate SHAP values for words in the input text
    word_impacts = [
        (word, word_shap_values.get(word, 0.0), "positively" if word_shap_values.get(word, 0.0) > 0 else "negatively")
        for word in words
    ]

    # Sort words by their absolute SHAP value (impact)
    word_impacts_sorted = sorted(word_impacts, key=lambda x: abs(x[1]), reverse=True)

    # Call calculate_word_impact_summary to get average impacts
    avg_positive, avg_negative = calculate_word_impact_summary_classifier(shap_values, text, vectorizer)

    # Display the summary
    print("\nSummary of Word Impacts:")
    print(f"Average Positive Impact: {avg_positive:.4f}")
    print(f"Average Negative Impact: {avg_negative:.4f}")

    # Return structured data (word impacts with direction)
    return {
        "word_impacts_sorted": word_impacts_sorted,
        "avg_positive": avg_positive,
        "avg_negative": avg_negative
    }

def calculate_word_impact_summary(shap_values, tokenizer):
    """Calculates the average impact of positive and negative words.
    Args:
        shap_values: shap.Explanation object
    Returns:
        tuple: (avg_positive_impact, avg_negative_impact)
    """

    positive_impacts = []
    negative_impacts = []

    for word, value in zip(shap_values.data[0], shap_values.values[0][:, 1]):
        if word not in tokenizer.all_special_tokens and word != tokenizer.pad_token:
          if value > 0:
              positive_impacts.append(value)
          else:
              negative_impacts.append(value)

    avg_positive_impact = sum(positive_impacts) / len(positive_impacts) if positive_impacts else 0
    avg_negative_impact = sum(negative_impacts) / len(negative_impacts) if negative_impacts else 0

    return avg_positive_impact, avg_negative_impact

def calculate_word_impact_summary_classifier(shap_values, text, vectorizer):
    """Calculates the average impact of positive and negative words.

    Args:
        shap_values: shap.Explanation object
        text (str): Original text for extracting words.
        vectorizer (object): TF-IDF vectorizer object.

    Returns:
        tuple: (avg_positive_impact, avg_negative_impact)
    """
    feature_names = vectorizer.get_feature_names_out()

    # Create a dictionary mapping words to their SHAP values
    word_shap_values = {}
    for i, feature_name in enumerate(feature_names):
        word_shap_values[feature_name] = shap_values[0][i]

    positive_impacts = []
    negative_impacts = []

    for word in text.split():
        shap_value = word_shap_values.get(word, 0.0)
        if shap_value > 0:
            positive_impacts.append(shap_value)
        else:
            negative_impacts.append(shap_value)

    avg_positive_impact = sum(positive_impacts) / len(positive_impacts) if positive_impacts else 0
    avg_negative_impact = sum(negative_impacts) / len(negative_impacts) if negative_impacts else 0

    return avg_positive_impact, avg_negative_impact
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    global model, tokenizer  # Declare model and tokenizer as global variables
    if request.method == 'GET':
        return "This route expects a POST request."
    if request.method == "POST":
        print("Received POST request with data:", request.form)
        # Get input data from the form
        name = request.form["name"]
        first_party = request.form["first_party"]
        second_party = request.form["second_party"]
        majority_vote = int(request.form["majority_vote"])
        minority_vote = int(request.form["minority_vote"])
        decision_type = request.form["decision_type"]
        disposition = request.form["disposition"]
        issue_area = request.form["issue_area"]
        chief_justice = request.form["chief_justice"]
        facts = request.form["facts"]
        model_type = request.form["model_type"]
        llm = request.form.get("llm", None)
        peft = request.form.get("peft", None)
        classifier = request.form.get("classifier", None)

        if not all([first_party, second_party]):
            return "Error: All input fields must be filled."
        # Process based on model type
        if model_type == 'LLM and PEFT':
            try:
                model_path = f"models/{llm}_{peft}"
                config = PeftConfig.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=2)
                model = PeftModel.from_pretrained(model, model_path)
                tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = model.to(device)

                # Create the input text for the model
                input_text = f"{name} {first_party} {second_party} {majority_vote} to {minority_vote} {decision_type} {disposition}  {issue_area} {facts} {chief_justice}"

                #pre_process input
                input_text = preprocess_text(input_text)
                print(input_text)

                # Tokenize the input text
                inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
                # Compute logits
                with torch.no_grad():
                    logits = model(inputs).logits
                # Convert logits to label
                predictions = torch.argmax(logits)
                predicted_label = id2label[predictions.tolist()]
                print(f"Prediction: {predicted_label}")
                # Explain the prediction using SHAP and attention map
                shap_data = explain_prediction_shap(input_text, predicted_label)

                shap_html = shap_data['shap_plot']

                return render_template("result.html", 
                prediction=predicted_label,
                word_impacts=shap_data['formatted_impacts'],
                avg_positive=shap_data['avg_positive'],
                avg_negative=shap_data['avg_negative'],
                shap_html=shap_html  # Include the SHAP plot or other explanations if needed
            )

            except Exception as e:
                return f"Error loading model from {model_path}: {e}"

        else:  # Classifier
            try:
                model = joblib.load(f"models/classifier/{classifier}_model.joblib")
                tfidf_vectorizer = joblib.load("models/classifier/tfidf_vectorizer.joblib")
                input_text = f"{name} {first_party} {second_party} {majority_vote} to {minority_vote} {decision_type} {disposition}  {issue_area} {facts} {chief_justice}"
                #pre_process input
                input_text = preprocess_text(input_text)
                print(input_text)
                
                input_vector = tfidf_vectorizer.transform([input_text])
                #input_dmatrix = DMatrix(input_vector)  # Convert to DMatrix
                #prediction = model.predict(input_dmatrix)
                prediction = model.predict(input_vector)
                # Prediction label if 1 then First Party Wins else second party wins
                prediction_label = "First Party Wins" if prediction[0] == 1 else "First Party Loses"
                print(f"Prediction: {prediction_label}")  # Or render_template if needed

                shap_data = explain_prediction_shap_classifier(input_text, model, tfidf_vectorizer)
                shap_html = shap_data['shap_plot']

                return render_template("result.html", 
                prediction=prediction_label,
                word_impacts=shap_data['formatted_impacts'],
                avg_positive=shap_data['avg_positive'],
                avg_negative=shap_data['avg_negative'],
                shap_html=shap_html  # Include the SHAP plot or other explanations if needed
            )

            except Exception as e:
                return f"Error loading classifier or vectorizer: {e}"
                        
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=False, host ='0.0.0.0')
