from dataclasses import dataclass, field
import logging
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments
import sys
import os
import transformers
from transformers.trainer_utils import get_last_checkpoint
from models import FCN4FlareModel, FCN4FlareConfig
from datasets import load_dataset   
from transformers import Trainer
import evaluate
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from data.data_collator import DataCollatorForFlareDetection

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """

    dataset_name: Optional[str] = field(
        default="Maxwell-Jia/kepler_flare",
        metadata={
            "help": "Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub)."
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
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

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Maxwell-Jia/fcn4flare",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name_or_path: str = field(
        default="Maxwell-Jia/fcn4flare",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )


def compute_metrics(eval_preds):
    """
    Compute multiple metrics for binary segmentation evaluation.
    Only computes metrics on non-padded (non-NaN) portions of sequences.
    """        
    logits, labels = eval_preds
    
    # Create mask for non-NaN values
    valid_mask = ~np.isnan(labels)
    
    # Apply sigmoid and threshold for binary predictions
    predictions_sigmoid = 1 / (1 + np.exp(-logits))
    predictions = (predictions_sigmoid > 0.5).astype(np.int64)
    
    # Filter out padded values using the mask
    predictions_flat = predictions[valid_mask]
    predictions_sigmoid_flat = predictions_sigmoid[valid_mask]
    labels_flat = labels[valid_mask].astype(np.int64)
    
    # Load metrics with zero_division parameter
    metric_precision = evaluate.load("precision")
    metric_recall = evaluate.load("recall")
    metric_f1 = evaluate.load("f1")
    
    # Calculate metrics with zero_division=0
    precision = metric_precision.compute(
        predictions=predictions_flat, 
        references=labels_flat,
        average="binary",
        zero_division=0
    )["precision"]
    
    recall = metric_recall.compute(
        predictions=predictions_flat, 
        references=labels_flat,
        average="binary",
        zero_division=0
    )["recall"]
    
    f1 = metric_f1.compute(
        predictions=predictions_flat, 
        references=labels_flat,
    )["f1"]
    
    # Calculate IoU (Intersection over Union)
    intersection = np.sum(predictions_flat * labels_flat)
    union = np.sum(predictions_flat) + np.sum(labels_flat) - intersection
    iou = intersection / (union + 1e-8)
    
    # Calculate Dice coefficient
    dice = 2 * intersection / (np.sum(predictions_flat) + np.sum(labels_flat) + 1e-8)
    
    # Calculate AUC-ROC and Average Precision
    try:
        auc_roc = roc_auc_score(labels_flat, predictions_sigmoid_flat)
        avg_precision = average_precision_score(labels_flat, predictions_sigmoid_flat)
    except ValueError:  # Handle cases where there's only one class
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
    # See all possible arguments in transformers.training_args
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
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

    # Load dataset
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # TODO support datasets from local folders
    dataset = load_dataset(data_args.dataset_name, trust_remote_code=True)

    # Rename column names to standardized names (only "input_features" and "labels" need to be present)
    for split in dataset.keys():
        if "flux_norm" in dataset[split].column_names:
            dataset[split] = dataset[split].rename_column("flux_norm", "input_features")
        if "label" in dataset[split].column_names:
            dataset[split] = dataset[split].rename_column("label", "labels")

    # If we don't have a validation split, split off a percentage of train as validation.
    data_args.train_val_split = None if "validation" in dataset.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset["train"].train_test_split(data_args.train_val_split)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Data collator
    data_collator = DataCollatorForFlareDetection(pad_value=float('nan'))

    # Load model
    if model_args.config_name_or_path and os.path.isfile(model_args.config_name_or_path):
        # If config file is provided, initialize model from config
        config = FCN4FlareConfig.from_json_file(model_args.config_name_or_path)
        model = FCN4FlareModel(config)
        logger.info(f"Model initialized from config file: {model_args.config_name_or_path}")
    else:
        # Load pretrained model/config
        config = FCN4FlareConfig.from_pretrained(
            model_args.config_name_or_path if model_args.config_name_or_path else model_args.model_name_or_path
        )
        model = FCN4FlareModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
        )
        logger.info(f"Model loaded from pretrained: {model_args.model_name_or_path}")

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
            )

    if training_args.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            dataset["validation"] = (
                dataset["validation"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if training_args.do_train else None,
        eval_dataset=dataset["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()