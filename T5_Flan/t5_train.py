import os
import json
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer, DataCollatorWithPadding, T5ForConditionalGeneration, TrainingArguments, Trainer

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

# load the data from train.json
with open("TORQUE-dataset/data/train.json", "r") as file:
    dataset = json.load(file)

# print("Example from the dataset:")
# print(dataset[0])

tokenized_dataset = []

for example in dataset:
    passages = example["passages"]
    for passage_info in passages:
        passage = passage_info["passage"]
        question_answer_pairs = passage_info["question_answer_pairs"]

        for qa_pair in question_answer_pairs:
            question = qa_pair["question"]
            answer_info = qa_pair

            input_text = f"question: {question} context: {passage}"
            
            # Tokenize with truncation and padding directly applied
            tokenized_inputs = tokenizer(
                input_text, 
                add_special_tokens=True, 
                truncation=True, 
                padding="max_length", 
                max_length=512,
                return_tensors="pt"
            )
            
            # For T5, the labels should be the encoded answer text with special tokens
            # Here, we use the passage for simplicity, but you might need to adjust based on your task
            tokenized_labels = tokenizer(
                passage, 
                add_special_tokens=True, 
                truncation=True, 
                padding="max_length", 
                max_length=512,
                return_tensors="pt"
            )

            # Here, we assume you're constructing a dataset for training, so we don't need return_tensors
            # Adjust based on your actual needs
            tokenized_example = {
                "input_ids": tokenized_inputs["input_ids"].squeeze().tolist(),  # Remove batch dimension
                "labels": tokenized_labels["input_ids"].squeeze().tolist(),  # Adjusted to use 'labels' for clarity
                "is_default_question": answer_info["is_default_question"]
            }

            tokenized_dataset.append(tokenized_example)


print("TOKENIZED")

class TorqueDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(self.dataset[index]["input_ids"], dtype=torch.long),
            "labels": torch.tensor(self.dataset[index]["labels"], dtype=torch.long),  # Corrected key here
            "is_default_question": self.dataset[index]["is_default_question"]
        }


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_size = int(0.8 * len(tokenized_dataset))
val_size = int(0.1 * len(tokenized_dataset))

train_dataset = TorqueDataset(tokenized_dataset[:train_size])
val_dataset = TorqueDataset(tokenized_dataset[train_size:train_size+val_size])
test_dataset = TorqueDataset(tokenized_dataset[train_size+val_size:])

model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

checkpoint_directory = "./checkpoint"
training_args = TrainingArguments(
    output_dir=checkpoint_directory,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=5,
    save_steps=5,
    resume_from_checkpoint=None
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

trainer.train()
print("TRAINED")

# Save the trained model and tokenizer
model_save_path = "flan_t5_small_torque_model.pth"
tokenizer_save_path = "flan_t5_small_torque_tokenizer"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)

eval_results = trainer.evaluate(test_dataset)
print(eval_results)

# question = "What will happen in the future?"
# passage = "Mobil is cutting back its U.S. oil and gas exploration and production group by up to 15% as part of a restructuring of the business."

# input_text = f"question: {question} answer: {passage}"
# input_ids = tokenizer.encode(input_text, return_tensors="pt")

# outputs = model.generate(input_ids)
# generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(generated_answer)