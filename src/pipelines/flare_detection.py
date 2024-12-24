from typing import Dict, List, Union
import numpy as np
import torch
from transformers import Pipeline
from astropy.io import fits

class FlareDetectionPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_count = 0
        
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        
        # Add parameters that need to be passed to specific steps
        return preprocess_kwargs, {}, postprocess_kwargs

    def preprocess(self, light_curve: Union[np.ndarray, str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        """Preprocess the input light curve from FITS files.
        
        Args:
            light_curve: Single FITS file path, list of FITS file paths, or numpy array
        """
        # Convert single path to list
        if isinstance(light_curve, str):
            light_curve = [light_curve]
        
        # Handle list of FITS file paths
        if isinstance(light_curve, list) and isinstance(light_curve[0], str):
            # Read data from all FITS files
            flux_data = []
            times_data = []
            lengths = []  # Store lengths of each light curve
            
            # First pass: get max length and collect data
            max_length = 0
            for fits_path in light_curve:
                with fits.open(fits_path) as hdul:
                    time = hdul[1].data['TIME'].astype(np.float32)
                    flux = hdul[1].data['PDCSAP_FLUX'].astype(np.float32)
                    # Normalize flux
                    flux = flux / np.nanmedian(flux)
                    
                    max_length = max(max_length, len(flux))
                    lengths.append(len(flux))
                    flux_data.append(flux)
                    times_data.append(time)
            
            # Second pass: pad sequences
            padded_flux = []
            padded_times = []
            sequence_mask = []
            
            for flux, time, length in zip(flux_data, times_data, lengths):
                # Create padding
                pad_length = max_length - length
                
                # Pad flux and time arrays
                padded_f = np.pad(flux, (0, pad_length), mode='constant', constant_values=np.nan)
                padded_t = np.pad(time, (0, pad_length), mode='constant', constant_values=np.nan)
                
                # Create mask (1 for real values, 0 for padding)
                mask = np.ones(length)
                mask = np.pad(mask, (0, pad_length), mode='constant', constant_values=0)
                
                padded_flux.append(padded_f)
                padded_times.append(padded_t)
                sequence_mask.append(mask)
            
            # Store time data as attribute for use in postprocessing
            self.time_series = np.array(padded_times)
            # Convert to arrays
            flux_array = np.array(padded_flux)
            sequence_mask = np.array(sequence_mask)
            
            # Add channel dimension
            flux_array = flux_array.reshape(flux_array.shape[0], flux_array.shape[1], 1)
            
            # Convert to torch tensors
            inputs = torch.tensor(flux_array, dtype=torch.float32)
            mask = torch.tensor(sequence_mask, dtype=torch.float32)
            
        return {
            "input_features": inputs,
            "sequence_mask": mask
        }

    def _forward(self, model_inputs, **forward_params):
        """Forward pass through the model.
        
        Args:
            model_inputs: Dictionary containing input tensors
            forward_params: Additional parameters for the forward pass
        """
        if model_inputs is None:
            raise ValueError("model_inputs cannot be None. Check if preprocess method is returning correct dictionary.")
        
        if "input_features" not in model_inputs:
            raise KeyError("model_inputs must contain 'input_features' key.")
        
        # Save input_features for use in postprocessing
        self.input_features = model_inputs["input_features"]
        
        # Ensure input_features is properly passed to the model
        return self.model(
            input_features=model_inputs["input_features"],
            sequence_mask=model_inputs.get("sequence_mask", None),
            return_dict=True
        )

    def postprocess(self, model_outputs, **kwargs):
        """
        Postprocess the model outputs to detect flare events.
        Returns a list of dictionaries containing flare events information.
        """
        logits = model_outputs.logits
        predictions = torch.sigmoid(logits).squeeze(-1)
        binary_predictions = (predictions > 0.5).long()
        
        # Convert to numpy for processing
        predictions_np = binary_predictions.cpu().numpy()
        flux_data = self.input_features.cpu().numpy()
        
        flare_events = []
        
        def is_valid_flare(flux, start_idx, end_idx, peak_idx):
            """Helper function to validate flare events
            
            Args:
                flux: Array of flux values
                start_idx: Start index of potential flare
                end_idx: End index of potential flare
                peak_idx: Peak index of potential flare
                
            Returns:
                bool: True if the event is a valid flare, False otherwise
            """
            # Duration of a flare should be longer than 3 cadences
            if end_idx - start_idx < 2:
                return False
            
            try:
                # If start time is the peak time, flux[start] must be greater than flux[start-1]
                if peak_idx == start_idx and flux[peak_idx] <= flux[peak_idx - 1]:
                    return False
                    
                # Time for flux to decrease should be longer than that to increase
                if end_idx - peak_idx <= peak_idx - start_idx:
                    return False
                    
                # Check flux level consistency before and after flare
                alter = (flux[peak_idx] - flux[start_idx - 2]) / (flux[peak_idx] - flux[end_idx + 2] + 1e-8)
                # Flux level should be similar before and after flare
                if alter < 0.5 or alter > 2 or np.isnan(alter):
                    return False
                    
                # Check if the slope before peak is too steep
                # if np.abs(flux[peak_idx] - flux[peak_idx-1]) < 1.2 * np.abs(flux[peak_idx-1] - flux[peak_idx-2]):
                #     return False
                    
            except (IndexError, ValueError):
                return False
            
            return True
        
        for i in range(predictions_np.shape[0]):
            pred = predictions_np[i]
            flux = flux_data[i, :, 0]  # Get flux data
            flare_idx = np.where(pred == 1)[0]
            
            if len(flare_idx) == 0:
                continue
            
            # Find continuous segments
            splits = np.where(np.diff(flare_idx) > 1)[0] + 1
            segments = np.split(flare_idx, splits)
            
            for segment in segments:
                # Skip short segments early
                if len(segment) < 3:
                    continue
                
                start_idx = segment[0]
                end_idx = segment[-1]
                
                # Find peak within segment
                segment_flux = flux[start_idx:end_idx+1]
                peak_idx = np.argmax(segment_flux) + start_idx
                
                # Validate flare characteristics
                if not is_valid_flare(flux, start_idx, end_idx, peak_idx):
                    continue
                
                # Valid flare event found
                start_time = float(self.time_series[i][start_idx])
                end_time = float(self.time_series[i][end_idx])
                duration = end_time - start_time
                event = {
                    "start_idx": int(start_idx),
                    "peak_idx": int(peak_idx),
                    "end_idx": int(end_idx),
                    "start_time": start_time,
                    "peak_time": float(self.time_series[i][peak_idx]),
                    "end_time": end_time,
                    "duration": duration,
                    "confidence": float(predictions[i, segment].mean()),
                }
                flare_events.append(event)
        
        return flare_events

def load_flare_detection_pipeline(
    model_name: str = "Maxwell-Jia/fcn4flare",
    device: int = -1,
    **kwargs
) -> FlareDetectionPipeline:
    """
    Load a flare detection pipeline.
    
    Args:
        model_name (str): The model name or path to load
        device (int): Device to use (-1 for CPU, GPU number otherwise)
        **kwargs: Additional arguments to pass to the pipeline
        
    Returns:
        FlareDetectionPipeline: A pipeline for flare detection
    """
    return FlareDetectionPipeline(
        model=model_name,
        device=device,
        **kwargs
    )