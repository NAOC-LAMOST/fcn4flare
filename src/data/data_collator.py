from dataclasses import dataclass
from typing import Dict, List, Union
import torch
from transformers.data.data_collator import DataCollatorMixin


@dataclass
class DataCollatorForFlareDetection(DataCollatorMixin):
    """
    Data collator for flare detection. Handles variable length sequences by padding.
    
    Args:
        pad_value: The value to use for padding (default: nan)
    """
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
            batch: Dictionary with padded tensors and sequence masks
        """
        # Find max length in the batch
        max_length = max(len(f['input_features']) for f in features)
        
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
                # Initialize padded tensor with nans for labels
                padded = torch.full(
                    (len(features), max_length),
                    self.pad_value,
                    dtype=torch.float32
                )
                
                # Fill in the actual values
                for i, feature in enumerate(features):
                    length = len(feature[key])
                    padded[i, :length] = torch.tensor(feature[key])
                    
                batch[key] = padded
                
        return batch
