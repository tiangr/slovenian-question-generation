import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset

# Load the tokenizer and model
model_name = 'doc2query/msmarco-14langs-mt5-base-v1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("Model and Tokenizer loaded successfully")

# Load and prepare the dataset
dataset_name = 'thetian11/slosquad' 
raw_dataset = load_dataset(dataset_name, split="train")
split_dataset = raw_dataset.train_test_split(test_size=0.2)
print("Dataset loaded successfully")

# Preprocess function for Seq2Seq
def preprocess_function(examples):
    model_inputs = tokenizer(examples["context"], max_length=1024, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["question"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = split_dataset.map(preprocess_function, batched=True)
print("Dataset tokenized successfully")
# Setting training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True
)

# Initialize Seq2Seq Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer
)
print("Trainer initialized successfully")
# Training the model
trainer.train()
print("Training completed successfully")

# Saving the trained model
model.save_pretrained("outputs")
print("Model saved successfully")

# Example usage of the trained model
print("Example usage: ")
context = "Normani (Norman: Nourmands; Francoščina: Normandi; Latinščina: Normanni) so bili ljudje, ki so v 10. in 11. stoletju dali ime Normandiji, regiji v Franciji. Bili so potomci nordijskih plenilcev in piratov iz Danske, Islandije in Norveške, ki so pod svojim voditeljem Rollom prisegli zvestobo kralju Karlu III. iz Zahodne Frankovske. Skozi generacije asimilacije in mešanja z domačimi frankovskimi in rimsko-gavskimi populacijami so se njihovi potomci postopoma združili s karolinškimi kulturami Zahodne Frankovske. Posebna kulturna in etnična identiteta Normanov se je sprva pojavila v prvi polovici 10. stoletja in se je razvijala v naslednjih stoletjih."
input_ids = tokenizer(context, return_tensors="pt").input_ids
outputs = model.generate(input_ids)
print("Generated Question:", tokenizer.decode(outputs[0], skip_special_tokens=True))
