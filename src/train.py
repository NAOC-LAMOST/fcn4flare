import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.trainer_callback import EarlyStoppingCallback

from data.data_collator import DataCollatorForFlareDetection
from models import FCN4FlareModel, FCN4FlareConfig


AutoConfig.register("fcn4flare", FCN4FlareConfig)
AutoModel.register(FCN4FlareConfig, FCN4FlareModel)

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    input_features_column_name: Optional[str] = field(
        default="flux_norm",
        metadata={"help": "The name of the column containing the input features."}
    )
    labels_column_name: Optional[str] = field(
        default="label",
        metadata={"help": "The name of the column containing the labels."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of test examples to this "
                "value if set."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically "
                "when batching to the maximum length in the batch."
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None:
            raise ValueError("dataset_name is required")    

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Maxwell-Jia/fcn4flare",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name_or_path: str = field(
        default="Maxwell-Jia/fcn4flare",
        metadata={"help": "Path to local config file or model identifier from huggingface.co/models"},
    )
    cache_dir: str = field(
        default=None,
        metadata={"help": "Path to cache directory"},
    )
    token: str = field(
        default=None,
        metadata={"help": "Hugging Face token"},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code"},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    early_stopping_patience: int = field(
        default=5,
        metadata={"help": "Stop training when the metric worsens for N epochs."}
    )
    early_stopping_threshold: float = field(
        default=0.0001,
        metadata={"help": "Denotes how much the specified metric must improve to satisfy early stopping conditions."}
    )


def compute_metrics(eval_preds):
    """
    Compute multiple metrics for binary segmentation evaluation.
    Only computes metrics on non-padded portions of sequences.
    """        
    predictions = eval_preds.predictions[0]  # Get logits with shape [num_samples, max_seq_len, 1]
    labels = eval_preds.label_ids  # Shape [num_samples, max_seq_len]
    
    # Remove last dimension to match labels shape
    predictions = predictions.squeeze(-1)  # Now shape is [num_samples, max_seq_len]
    predictions_sigmoid = 1 / (1 + np.exp(-predictions))
    
    # Remove padded values (using -100 in labels to indicate padding)
    true_predictions = []
    true_labels = []
    
    for pred, label in zip(predictions_sigmoid, labels):
        valid_mask = (label != -100)  # 使用-100标识padding
        true_predictions.append(pred[valid_mask])
        true_labels.append(label[valid_mask])
    
    # Flatten predictions and labels
    true_predictions = np.concatenate(true_predictions)
    true_labels = np.concatenate(true_labels)
    
    # Convert to binary predictions
    binary_predictions = (true_predictions > 0.5).astype(np.int64)

    precision = precision_score(
        y_true=true_labels, 
        y_pred=binary_predictions, 
        average="binary", 
        zero_division=0
    )
    recall = recall_score(
        y_true=true_labels, 
        y_pred=binary_predictions, 
        average="binary", 
        zero_division=0
    )
    f1 = f1_score(
        y_true=true_labels, 
        y_pred=binary_predictions, 
        average="binary"
    )
    
    # Calculate additional metrics
    intersection = np.sum(binary_predictions * true_labels)
    union = np.sum(binary_predictions) + np.sum(true_labels) - intersection
    iou = intersection / (union + 1e-8)
    dice = 2 * intersection / (np.sum(binary_predictions) + np.sum(true_labels) + 1e-8)
    
    try:
        auc_roc = roc_auc_score(true_labels, true_predictions)
        avg_precision = average_precision_score(true_labels, true_predictions)
    except ValueError:
        auc_roc = 0.0
        avg_precision = 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
        "auc_roc": auc_roc,
        "average_precision": avg_precision
    }


def main():
    # 1. Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    os.makedirs(training_args.output_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("./train.log")
        ],
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 4. Set seed before initializing model
    set_seed(training_args.seed)

    # 5. Load dataset
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    # 6. Split train/validation if needed
    if "validation" not in dataset and isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # 7. Load pretrained model and config
    config = (AutoConfig.from_pretrained(model_args.config_name_or_path)
             if os.path.isfile(model_args.config_name_or_path)
             else AutoConfig.from_pretrained(model_args.config_name_or_path))
    
    model = (AutoModel.from_config(config)
            if os.path.isfile(model_args.config_name_or_path)
            else AutoModel.from_pretrained(model_args.model_name_or_path, config=config))

    # 8. Prepare datasets for trainer
    def preprocess_function(examples):
        """Pad sequences and create sequence mask"""
        input_features = examples[data_args.input_features_column_name]
        labels = examples[data_args.labels_column_name]
        
        # Convert boolean labels to int
        labels = [int(l) for l in labels]
        
        if data_args.max_seq_length is not None:
            # Pad or truncate sequences
            if len(input_features) > data_args.max_seq_length:
                input_features = input_features[:data_args.max_seq_length]
                labels = labels[:data_args.max_seq_length]
                sequence_mask = [1] * data_args.max_seq_length
            else:
                pad_length = data_args.max_seq_length - len(input_features)
                sequence_mask = [1] * len(input_features) + [0] * pad_length
                input_features.extend([float('nan')] * pad_length)
                labels.extend([-100] * pad_length)
        else:
            sequence_mask = [1] * len(input_features)
        
        return {
            "input_features": input_features,
            "labels": labels,
            "sequence_mask": sequence_mask
        }

    train_dataset = None
    eval_dataset = None
    predict_dataset = None

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                desc="Padding train dataset",
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=train_dataset.column_names,
                features=datasets.Features({
                    "input_features": datasets.Sequence(datasets.Value("float32")),
                    "labels": datasets.Sequence(datasets.Value("int64")),
                    "sequence_mask": datasets.Sequence(datasets.Value("int64"))
                })
            )

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="eval dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                desc="Padding validation dataset",
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=eval_dataset.column_names,
                features=datasets.Features({
                    "input_features": datasets.Sequence(datasets.Value("float32")),
                    "labels": datasets.Sequence(datasets.Value("int64")),
                    "sequence_mask": datasets.Sequence(datasets.Value("int64"))
                })
            )

    if training_args.do_predict:
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = dataset["test"]
        if data_args.max_predict_samples:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="predict dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                desc="Padding test dataset",
                load_from_cache_file=not data_args.overwrite_cache,
                remove_columns=predict_dataset.column_names,
                features=datasets.Features({
                    "input_features": datasets.Sequence(datasets.Value("float32")),
                    "labels": datasets.Sequence(datasets.Value("int64")),
                    "sequence_mask": datasets.Sequence(datasets.Value("int64"))
                })
            )

    # 9. Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForFlareDetection(
            pad_to_max_length=data_args.pad_to_max_length,
            max_length=data_args.max_seq_length,
            pad_to_multiple_of=8 if training_args.fp16 else None
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience,
                                       early_stopping_threshold=training_args.early_stopping_threshold)]
    )

    # 10. Training
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 11. Evaluation
    if training_args.do_eval:
        logger.info("\n*** Evaluate ***\n")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 12. Prediction
    if training_args.do_predict:
        logger.info("\n*** Predict ***\n")
        predict_results = trainer.predict(predict_dataset)
        metrics = predict_results.metrics
        metrics["predict_samples"] = len(predict_dataset)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        
        # Get logits with shape [num_samples, max_seq_len, 1]
        predictions = predict_results.predictions[0]
        # Remove last dimension to get shape [num_samples, max_seq_len]
        predictions = predictions.squeeze(-1)
        # Apply sigmoid to get probabilities
        predictions_sigmoid = 1 / (1 + np.exp(-predictions))
        # Convert to binary predictions
        binary_predictions = (predictions_sigmoid > 0.5).astype(np.int64)
        
        # Save predictions
        output_predict_file = os.path.join(training_args.output_dir, "predictions.npz")
        np.savez(
            output_predict_file,
            logits=predictions,
            probabilities=predictions_sigmoid,
            predictions=binary_predictions,
            labels=predict_results.label_ids if predict_results.label_ids is not None else None,
            metrics=predict_results.metrics if predict_results.metrics is not None else None
        )
        logger.info(f"Saved predictions to {output_predict_file}")

if __name__ == "__main__":
    main()