import os
import json
import time
import csv
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from transformers import T5Tokenizer, DataCollatorWithPadding, T5ForQuestionAnswering, AdamW, T5ForConditionalGeneration


def initialize_device():
    if torch.cuda.is_available():
        print("CUDA is available. Using CUDA...")
        device = torch.device("cuda")
        print(f"Using device: {device}")
    elif torch.backends.mps.is_available():
        print("MPS backend is available. Using MPS...")
        device = torch.device("mps")
        print(f"Using device: {device}")
    else:
        print("Neither CUDA nor MPS backend is available. Using CPU instead.")
        device = torch.device("cpu")
        print(f"Using device: {device}")
    return device

def preprocess_data(csv_file):
    dataset = []
    with open(csv_file, "r") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            dataset.append(row)
    return dataset

def tokenize_dataset(dataset, tokenizer):
    tokenized_dataset = []
    max_answer_length = 0
    for example in dataset:
        context = example["Context"]
        question = example["Question"]
        answer = example["Answer"]

        input_text = f"question: {question} context: {context}"

        # Tokenize with truncation and padding directly applied
        tokenized_inputs = tokenizer(
            input_text,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        # Preprocess the answer string and tokenize each keyword separately
        answer_keywords = answer.strip("[]").split(", ")
        start_positions = []
        end_positions = []

        for keyword in answer_keywords:
            keyword_tokens = tokenizer.encode(keyword, add_special_tokens=False)
            keyword_len = len(keyword_tokens)

            if keyword_tokens:
                # Find the start and end positions of the keyword in the tokenized context
                context_ids = tokenized_inputs["input_ids"][0].tolist()
                try:
                    start_pos = context_ids.index(keyword_tokens[0])
                    end_pos = start_pos + keyword_len - 1

                    start_positions.append(start_pos)
                    end_positions.append(end_pos)
                except ValueError:
                    # Keyword not found in the context
                    pass

        max_answer_length = max(max_answer_length, len(start_positions))

        start_position = start_positions[0] if start_positions else 0
        end_position = end_positions[0] if end_positions else 0

        tokenized_example = {
            "input_ids": tokenized_inputs["input_ids"].squeeze().tolist(),
            "attention_mask": tokenized_inputs["attention_mask"].squeeze().tolist(),
            "start_position": start_position,
            "end_position": end_position,
            "category": example["Category"]
        }

        tokenized_dataset.append(tokenized_example)

    return tokenized_dataset, max_answer_length

class TorqueDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(self.dataset[index]["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(self.dataset[index]["attention_mask"], dtype=torch.long),
            "start_position": torch.tensor(self.dataset[index]["start_position"], dtype=torch.long),
            "end_position": torch.tensor(self.dataset[index]["end_position"], dtype=torch.long),
            "category": self.dataset[index]["category"]
        }

def create_dataloaders(tokenized_dataset, tokenizer):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_size = int(0.8 * len(tokenized_dataset))
    val_size = int(0.1 * len(tokenized_dataset))

    train_dataset = TorqueDataset(tokenized_dataset[:train_size])
    val_dataset = TorqueDataset(tokenized_dataset[train_size:train_size + val_size])
    test_dataset = TorqueDataset(tokenized_dataset[train_size + val_size:])

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=10, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=10, num_workers=0, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader

def train_model(device, train_dataloader, val_dataloader, tokenizer):
    model = T5ForQuestionAnswering.from_pretrained("google/flan-t5-small")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-4)
    num_epochs = 3
    max_grad_norm = 1.0

    checkpoint_dir = "T5_Flan/checkpoint"
    model_dir = "flan_t5_small_torque_model.pth"

    if os.path.exists(model_dir):
        print("Fully trained model found. Skipping training and loading the model.")
        model = T5ForQuestionAnswering.from_pretrained(model_dir)
        model.to(device)
        return model

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Check for the most recent checkpoint
    checkpoint_files = sorted(os.listdir(checkpoint_dir), reverse=True)
    if checkpoint_files:
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        start_epoch = checkpoint['epoch']
        start_batch_idx = checkpoint['batch_idx'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resumed from checkpoint. Starting from Epoch: {start_epoch + 1}, Batch: {start_batch_idx}")
    else:
        start_epoch = 0
        start_batch_idx = 0
        print("No checkpoint found. Starting training from the beginning.")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        start_time_epoch = time.time()  # Start time of the epoch
        print(f"Epoch {epoch + 1}/{num_epochs} started.")

        for batch_idx, batch in enumerate(train_dataloader):
            if epoch == start_epoch and batch_idx < start_batch_idx:
                continue  # Skip batches that have already been processed

            start_time_batch = time.time()  # Start time of the batch processing

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_position = batch["start_position"].to(device)
            end_position = batch["end_position"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_position,
                end_positions=end_position
            )

            loss = outputs.loss

            loss.backward()

            # gradient clipping and capturing the total norm of all gradients
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()

            optimizer.step()
            optimizer.zero_grad()

            end_time_batch = time.time()  # End time of the batch processing
            print(f"Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}, Grad Norm: {total_grad_norm:.4f}, Time per batch: {(end_time_batch - start_time_batch):.2f} seconds")

            if batch_idx % 60 == 0 and batch_idx != 0:  # checkpoint every 60 batches
                try:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}_batch_{batch_idx}.pt")
                    torch.save({
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                    }, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}.")
                except Exception as e:
                    print(f"Error occurred while saving checkpoint: {str(e)}")

        end_time_epoch = time.time()  # End time of the epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Time per epoch: {(end_time_epoch - start_time_epoch):.2f} seconds")

    model.save_pretrained(model_dir)
    return model

def evaluate_model(model, test_dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_position = batch["start_position"].to(device)
            end_position = batch["end_position"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_position,
                end_positions=end_position
            )

            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(test_dataloader)
    print(f"Test Loss: {avg_loss:.4f}")

def inference(model, tokenizer, question, context):
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        answer_start_index = torch.argmax(start_logits)
        answer_end_index = torch.argmax(end_logits)

        input_ids = input_ids.squeeze()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        answer = tokenizer.decode(input_ids[answer_start_index:answer_end_index+1])

    return answer

def untrained_inference(tokenizer, question, context):
    model = T5ForQuestionAnswering.from_pretrained("google/flan-t5-small")
    model.to(device)
    model.eval()

    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        answer_start_index = torch.argmax(start_logits)
        answer_end_index = torch.argmax(end_logits)

        input_ids = input_ids.squeeze()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        answer = tokenizer.decode(input_ids[answer_start_index:answer_end_index+1])

    return answer

if __name__ == '__main__':
    mp.set_start_method('spawn')

    print("---------- INITIALIZING ----------")
    device = initialize_device()
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    tokenizer_dir = "flan_t5_small_torque_tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)

    print("---------- PREPROCESSING ----------")
    dataset = preprocess_data("temporal_qa_data.csv")

    print("---------- TOKENIZING ----------")
    tokenized_dataset, max_answer_length = tokenize_dataset(dataset, tokenizer)

    print("---------- DATALOADER ----------")
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(tokenized_dataset, tokenizer)

    print("---------- TRAINING ----------")
    model = train_model(device, train_dataloader, val_dataloader, tokenizer)
    model_dir = "flan_t5_small_torque_model.pth"
    model.save_pretrained(model_dir)

    print("---------- EVALUATION ----------")
    evaluate_model(model, test_dataloader, device)

    print("---------- INFERENCE (Trained Model) ----------")
    question = "What happened after an event?"
    context = "After developing the World Wide Web, Tim Berners-Lee founded the World Wide Web Consortium (W3C) at MIT in 1994."
    answer = inference(model, tokenizer, question, context)
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Answer: {answer}")

    print("---------- INFERENCE (Untrained Model) ----------")
    untrained_answer = untrained_inference(tokenizer, question, context)
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Answer: {untrained_answer}")