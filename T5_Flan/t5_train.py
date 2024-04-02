import os
import json
import time
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, DataCollatorWithPadding, T5ForConditionalGeneration, AdamW

if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    print("---------- INITIALIZING ----------")

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

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")

    print("---------- PREPROCESSING ----------")
    # Load the data from temporal_qa_data.csv
    dataset = []
    with open("temporal_qa_data.csv", "r") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            dataset.append(row)

    tokenized_dataset = []

    print("---------- TOKENIZING ----------")
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

        # Tokenize the answer text as the label
        tokenized_labels = tokenizer(
            answer,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        tokenized_example = {
            "input_ids": tokenized_inputs["input_ids"].squeeze().tolist(),
            "labels": tokenized_labels["input_ids"].squeeze().tolist(),
            "category": example["Category"]
        }

        tokenized_dataset.append(tokenized_example)

    print("---------- DATALOADER ----------")

    class TorqueDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            return {
                "input_ids": torch.tensor(self.dataset[index]["input_ids"], dtype=torch.long),
                "labels": torch.tensor(self.dataset[index]["labels"], dtype=torch.long),
                "category": self.dataset[index]["category"]
            }

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_size = int(0.8 * len(tokenized_dataset))
    val_size = int(0.1 * len(tokenized_dataset))

    train_dataset = TorqueDataset(tokenized_dataset[:train_size])
    val_dataset = TorqueDataset(tokenized_dataset[train_size:train_size + val_size])
    test_dataset = TorqueDataset(tokenized_dataset[train_size + val_size:])

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=10, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=10, num_workers=0, pin_memory=True)

    print("---------- TRAINING ----------")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    model.to(device)

    # optimizer and learning rate
    optimizer = AdamW(model.parameters(), lr=1e-4)

    #  number of training epochs
    num_epochs = 3

    # maximum gradient norm for gradient clipping
    max_grad_norm = 1.0

    checkpoint_dir = "T5_Flan/checkpoint"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Check for the most recent checkpoint
    checkpoint_files = sorted(os.listdir(checkpoint_dir), reverse=True)
    if checkpoint_files:
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
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
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            loss.backward()

            # gradient clipping and capturing the total norm of all gradients
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()

            optimizer.step()
            optimizer.zero_grad()

            end_time_batch = time.time()  # End time of the batch processing
            print(f"Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}, Grad Norm: {total_grad_norm:.4f}, Time per batch: {(end_time_batch - start_time_batch):.2f} seconds")

            if (batch_idx + 1) % 10 == 0:  # Log every 10 batches
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}_batch_{batch_idx}.pt")
                torch.save({
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}.")

        end_time_epoch = time.time()  # End time of the epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Time per epoch: {(end_time_epoch - start_time_epoch):.2f} seconds")


    print("---------- MODEL SAVING ----------")

    # save trained model and tokenizer
    model_save_path = "flan_t5_small_torque_model.pth"
    tokenizer_save_path = "flan_t5_small_torque_tokenizer"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)

    print("---------- EVALUATION ----------")
    # evaluate on the test dataset
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(test_dataloader)
    print(f"Test Loss: {avg_loss:.4f}")
