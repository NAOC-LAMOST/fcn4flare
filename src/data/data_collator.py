from dataclasses import dataclass
from typing import Dict, List, Union

import torch
from transformers.data.data_collator import DataCollatorMixin


@dataclass
class DataCollatorForFlareDetection(DataCollatorMixin):
    """
    Data collator for flare detection. Handles variable length sequences by padding.
    
    Args:
        pad_to_max_length: Whether to pad all samples to max_length. False will pad to longest in batch.
        max_length: Maximum length to pad to if pad_to_max_length is True.
        pad_to_multiple_of: If set, will pad the sequence length to be multiple of this value.
        pad_value: The value to use for padding (default: nan)
    """
    pad_to_max_length: bool = False
    max_length: int = None
    pad_to_multiple_of: int = None
    pad_value: float = float('nan')
    return_tensors: str = "pt"

    def torch_call(
        self, 
        features: List[Dict[str, Union[List[float], List[int]]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate function to handle variable length sequences.
        
        Args:
            features: List of dictionaries containing the features
            
        Returns:
            batch: Dictionary with padded tensors
        """
        # Find max length in the batch
        batch_max_length = max(len(f['input_features']) for f in features)
        
        # If pad_to_max_length is True and max_length is set, use max_length
        if self.pad_to_max_length and self.max_length is not None:
            max_length = self.max_length
        else:
            max_length = batch_max_length

        # If pad_to_multiple_of is set, pad length to multiple of that value
        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )
        
        batch = {}
        
        for key in features[0].keys():
            if key == 'input_features':
                # Initialize padded tensor with nans for input features
                padded = torch.full(
                    (len(features), max_length, 1),
                    self.pad_value,
                    dtype=torch.float32
                )
                
                # Fill in the actual values
                for i, feature in enumerate(features):
                    length = len(feature[key])
                    padded[i, :length, 0] = torch.tensor(feature[key])
                    
                batch[key] = padded
                
            elif key == 'labels':
                # Initialize padded tensor with -100 for labels
                padded = torch.full(
                    (len(features), max_length),
                    -100,
                    dtype=torch.long
                )
                
                # Fill in the actual values
                for i, feature in enumerate(features):
                    length = len(feature[key])
                    padded[i, :length] = torch.tensor(feature[key], dtype=torch.long)
                    
                batch[key] = padded
                
            elif key == 'sequence_mask':
                # Convert sequence_mask to tensor with integer type
                batch[key] = torch.tensor(
                    [feature[key] for feature in features],
                    dtype=torch.long
                )
                
        return batch
