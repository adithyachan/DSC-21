from transformers import BertTokenizerFast, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
import torch
import pandas as pd
import os


def preprocess_for_answer_prediction(df, tokenizer):
    input_ids = []
    attention_masks = []
    labels = []  # Will store original token IDs for masked tokens and -100 elsewhere

    for index, row in df.iterrows():
        context_question = f"{row['Question']} {tokenizer.sep_token} {row['Context']}"
        tokenized_input = tokenizer(context_question, max_length=512, truncation=True, padding="max_length",
                                    return_tensors="pt", return_offsets_mapping=True)
        input_ids_seq = tokenized_input["input_ids"].squeeze().tolist()
        attention_mask_seq = tokenized_input["attention_mask"].squeeze().tolist()
        offset_mapping = tokenized_input["offset_mapping"].squeeze().tolist()
        tokenized_input.pop("offset_mapping")

        # Initialize labels with -100
        label_seq = [-100] * len(input_ids_seq)

        if row['Answer Indices'] != "[]":  # Check if there are answers
            for ans_indices in eval(row['Answer Indices']):  # Processing each set of answer indices
                start_char, end_char = eval(ans_indices)  # Convert string to tuple

                # Find token indexes corresponding to start_char and end_char
                start_token = next((idx for idx, offset in enumerate(offset_mapping) if offset[0] == start_char), None)
                end_token = next((idx for idx, offset in enumerate(offset_mapping) if offset[1] == end_char), None)

                if start_token is not None and end_token is not None:
                    # Mask answer tokens
                    for idx in range(start_token, end_token + 1):
                        input_ids_seq[idx] = tokenizer.mask_token_id
                        # Set label for masked token
                        correct_token_id = tokenizer.encode(row['Context'][start_char:end_char],
                                                            add_special_tokens=False)
                        label_seq[idx] = correct_token_id[0] if correct_token_id else -100  # Handle potential mismatch

        input_ids.append(input_ids_seq)
        attention_masks.append(attention_mask_seq)
        labels.append(label_seq)

    return TensorDataset(torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels))


df = pd.read_csv("temporal_qa_data.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
dataset = preprocess_for_answer_prediction(df, tokenizer)
model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)

print(df.head())

# Split dataset and create dataloaders
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=10)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=10)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=10)

# Save the test dataset for later evaluation
torch.save(test_dataset, 'test_dataset.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

checkpoint_dir = "model_checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

epochs = 100

start_epoch = 0
start_step = 0
latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")

if os.path.exists(latest_checkpoint_path):
    checkpoint = torch.load(latest_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    start_step = checkpoint.get('step', 0)  # For resuming training correctly
    print(f"Resuming training from epoch {start_epoch}, step {start_step}.")
else:
    print("No checkpoint found. Starting training from scratch.")

total_steps = len(train_dataloader) * (epochs - start_epoch)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

model.train()
for epoch in range(start_epoch, epochs):
    for step, (input_ids, attention_mask, labels) in enumerate(train_dataloader, start=start_step):
        if step == start_step:
            start_step = 0  # Reset start_step for subsequent epochs

        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 20 == 0:
            print(f"Epoch: {epoch}, Step: {step + 1}, Loss: {loss.item()}")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'step': step + 1,
            }
            torch.save(checkpoint, latest_checkpoint_path)

# Save the final model
model.save_pretrained('./finetuned_bert')
