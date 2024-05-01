# T5_FINETUNE_Electrical_ts
# Fine-Tuned T5 Model for Root Cause Analysis

# Description
This model is a fine-tuned version of the T5 (Text-to-Text Transfer Transformer) base model, specifically tailored to predict actions based on provided root causes in industrial or technical settings. The model has been trained to understand various root causes and suggest corresponding actions, facilitating faster decision-making and troubleshooting in operational environments.

# Model Details

Base Model: T5 Base

Training Data: The model was trained on a proprietary dataset consisting of documented root causes and the actions taken to resolve them in a manufacturing context.
Fine-Tuning Details: The model was fine-tuned for 3 epochs with a learning rate of 3e-4, using a batch size of 8. The fine-tuning process focused on adapting the T5 model to generate action plans based on textual descriptions of root causes.

# Usage
Installation
To use this model, you will need to install Python and the necessary Python libraries. The primary library required is transformers by Hugging Face.

-> pip install transformers torch

# Loading the Model
You can load the model using the Transformers library. Ensure you have the model and tokenizer files(final_model.zip) downloaded from the repository or Kaggle dataset.

from transformers import T5ForConditionalGeneration, T5Tokenizer

model_path = 'path_to_model_directory'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

#Making Predictions
To use the model to predict actions based on a root cause, use the following Python code:

def predict_action(root_cause):
    input_text = f"root cause: {root_cause} -> action:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output_ids = model.generate(input_ids)
    action = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return action

# Example
print(predict_action("Example of a root cause"))

# Contributing
Contributions to this model are welcome. You can contribute in the following ways:

Data: More data on root causes and actions can help to improve the model's accuracy and robustness.
Code: Enhancements in the prediction script, additional features, or performance optimizations are appreciated.
Issues: If you encounter issues while using this model, please report them in the issues section of this repository.

# License
This project is licensed under the MIT License - see the LICENSE file for details.





