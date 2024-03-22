from transformers import BertTokenizer, BertForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch
import pandas as pd
import os

# Function to find answer positions (unchanged)
def find_answer_positions(contexts, answers):
    start_positions = []
    end_positions = []
    for context, answer in zip(contexts, answers):
        start_position = context.find(answer)
        end_position = start_position + len(answer) - 1
        start_positions.append(start_position)
        end_positions.append(end_position)
    return start_positions, end_positions

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Load dataset (unchanged)
df = pd.read_csv("temporal_qa_data.csv")
df['Context'] = df['Context'].astype(str)
df['Answer'] = df['Answer'].astype(str)
df['start_positions'], df['end_positions'] = find_answer_positions(df['Context'], df['Answer'])

# Tokenize and create dataset (unchanged)
inputs = tokenizer(df['Context'].tolist(), df['Question'].tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512, return_token_type_ids=True, return_attention_mask=True)
inputs['start_positions'] = torch.tensor(df['start_positions'].tolist())
inputs['end_positions'] = torch.tensor(df['end_positions'].tolist())
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'], inputs['start_positions'], inputs['end_positions'])

# Split dataset and create dataloaders (unchanged)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=10)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=10)

# Device setup (unchanged)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

checkpoint_dir = "./model_checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Define epochs
epochs = 100

# Load checkpoint if exists
start_epoch = 0
start_step = 0  # New variable to keep track of step
latest_checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")

if os.path.exists(latest_checkpoint_path):
    checkpoint = torch.load(latest_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    start_step = checkpoint.get('step', 0)  # Ensure backward compatibility
    print(f"Resuming training from epoch {start_epoch}, step {start_step}.")
else:
    print("No checkpoint found. Starting training from scratch.")

# Calculate total steps for the scheduler considering continuation from checkpoint
total_steps = len(train_dataloader) * (epochs - start_epoch)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Adjusted training loop
for epoch in range(start_epoch, epochs):
    # Adjust enumeration of train_dataloader to start from `start_step` if resuming
    iterable_dataloader = enumerate(train_dataloader)
    if epoch == start_epoch and start_step > 0:
        # Fast-forward to start_step if resuming in the middle of an epoch
        for _ in range(start_step):
            next(iterable_dataloader)

    for step, batch in iterable_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'start_positions': batch[3],
            'end_positions': batch[4]
        }

        model.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Adjusted print and checkpoint logic
        if (step + 1) % 20 == 0 or (step + 1) == len(train_dataloader):
            print(f"Epoch: {epoch}, Step: {step + 1}, Loss: {loss.item()}")
            checkpoint = {
                'epoch': epoch,
                'step': step + 1,  # Save the next step to resume from
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(checkpoint, latest_checkpoint_path)

# Final model save
model.save_pretrained('./finetuned_bert')
