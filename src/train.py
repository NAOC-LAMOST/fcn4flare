import torch
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from src.models.fcn4flare import FCN4FlareConfig, FCN4FlareModel
import numpy as np
from typing import Dict, Union, Any

def collate_fn(examples):
    """
    Collate function to process batch data.
    """
    input_features = torch.stack([torch.tensor(example['features']) for example in examples])
    labels = torch.stack([torch.tensor(example['labels']) for example in examples])
    
    return {
        'input_features': input_features,
        'labels': labels
    }

def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    """
    predictions, labels = eval_pred
    predictions = torch.sigmoid(torch.from_numpy(predictions)).numpy()
    predictions = (predictions > 0.5).astype(np.float32)
    
    # Calculate Dice score
    smooth = 1e-8
    intersection = np.sum(predictions * labels, axis=1)
    dice = np.mean((2. * intersection + smooth) / 
                   (np.sum(predictions, axis=1) + np.sum(labels, axis=1) + smooth))
    
    return {
        'dice_score': dice
    }

def main():
    # Load dataset
    dataset = load_dataset("your-username/your-dataset-name")
    
    # Model config
    config = FCN4FlareConfig(
        input_dim=3,
        hidden_dim=64,
        output_dim=1,
        depth=4,
        dilation=[1, 2, 4, 8],
        maskdice_threshold=0.5,
        dropout_rate=0.1,
        kernel_size=3
    )
    
    # Initialize model
    model = FCN4FlareModel(config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="dice_score",
        greater_is_better=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    
    # Train model
    trainer.train()
    
    # Save final model
    trainer.save_model("./final_model")

if __name__ == "__main__":
    main()
