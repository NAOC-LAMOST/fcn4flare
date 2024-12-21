import datasets
from datasets import DatasetInfo
import pandas as pd
from typing import List, Dict, Any, Generator
import os
import numpy as np
from sklearn.model_selection import train_test_split
from astropy.io import fits


KEPLER_LIGHTCURVES_ROOT = "/mnt/lamostgpu/data/Yang_lightcurves/"
FLARE_EVENTS_FILE = "datasets/flare_events_yang2019.csv"

CITATION = """
@article{yang2019flare,
  title={The flare catalog and the flare activity in the Kepler mission},
  author={Yang, Huiqin and Liu, Jifeng},
  journal={The Astrophysical Journal Supplement Series},
  volume={241},
  number={2},
  pages={29},
  year={2019},
  publisher={IOP Publishing}
}

@article{jia2024fcn4flare,
  title={FCN4Flare: Fully Convolution Neural Networks for Flare Detection},
  author={Jia, Ming-Hui and Luo, A-Li and Qiu, Bo},
  journal={arXiv preprint arXiv:2407.21240},
  year={2024}
}
"""

DESCRIPTION = """
The Kepler Flare Dataset is a comprehensive collection of stellar flare events observed by the Kepler Space 
Telescope. This dataset is constructed based on the flare event catalog presented in Yang & Liu (2019), which 
provides a systematic study of stellar flares in the Kepler mission.

Dataset Content:
- Light curves (flux measurements) from the Kepler Space Telescope
- Binary labels for each time step indicating flare/non-flare events

Key Features:
- Normalized flux values for each light curve
- Binary classification labels (0: no flare, 1: flare event)
- Time series data with variable lengths
- High-quality photometric measurements from Kepler

The dataset is particularly useful for:
- Stellar flare detection and analysis
- Time series classification
- Astronomical event detection
- Deep learning applications in astronomical time series

The flare events are identified and validated according to the methodology described in Yang & Liu (2019). For 
machine learning applications using this dataset, please refer to Jia et al. (2024), which presents FCN4Flare, 
a fully convolutional neural network approach for flare detection.

Note: The flux values are normalized by dividing by their median values to facilitate machine learning 
applications.
"""

DATASET_URL = "TODO"
VERSION = "1.0.0"
LICENSE = "MIT"

TRAIN_SIZE = 0.8
VAL_SIZE = 0.05
TEST_SIZE = 0.15


class KeplerFlareDataset(datasets.GeneratorBasedBuilder):
    """Kepler Flare Dataset."""
    
    VERSION = "1.0.0"
    
    def _info(self) -> DatasetInfo:
        features = datasets.Features({
            'flux_norm': datasets.Sequence(datasets.Value('float32')),
            'label': datasets.Sequence(datasets.Value('bool')),
            'metadata': {
                'kic_id': datasets.Value('string'),
                'quarter': datasets.Value('string'),
                'file_name': datasets.Value('string'),
                'flux_median': datasets.Value('float32'),
            }
        })

        return datasets.DatasetInfo(
            description=DESCRIPTION,
            citation=CITATION,
            homepage=DATASET_URL,
            license=LICENSE,
            features=features,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        # Read flare events data
        events_df = pd.read_csv(FLARE_EVENTS_FILE)
        
        # Get unique file list
        unique_files = events_df['file'].unique()
        
        # First split: separate test set (15%)
        train_val_files, test_files = train_test_split(
            unique_files,
            test_size=TEST_SIZE,  # 15% for test set
            random_state=42
        )
        
        # Second split: separate validation set from remaining files (5%/85% ≈ 0.059)
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=VAL_SIZE/(1-TEST_SIZE),  # 5% of total
            random_state=42
        )
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"file_list": train_files, "events_df": events_df}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"file_list": val_files, "events_df": events_df}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"file_list": test_files, "events_df": events_df}
            ),
        ]

    def _get_file_path(self, file_name: str) -> str:
        """Construct the full file path from the file name.
        
        Args:
            file_name: File name in format 'kplrXXXXXXXXX-YYYYYYYYYYY_llc.fits'
            
        Returns:
            Full path to the file following the directory structure:
            root/XXXX/XXXXXXXXX/kplrXXXXXXXXX-YYYYYYYYYYY_llc.fits
        """
        # Extract KIC ID from filename (e.g., '008935655' from 'kplr008935655-...')
        kic_id = file_name[4:13]  # Extract all 9 digits
        subdir1 = kic_id[:4]      # First 4 digits for first subdirectory
        
        return os.path.join(KEPLER_LIGHTCURVES_ROOT, subdir1, kic_id, file_name)

    def _generate_examples(self, file_list: np.ndarray, events_df: pd.DataFrame):
        """Yields examples as (key, example) tuples.
        
        Args:
            file_list: List of light curve files to process
            events_df: DataFrame containing flare events information
            
        Yields:
            Tuple containing:
                - index (int): Example index
                - example (dict): Dictionary with normalized flux, labels, and metadata
        """
        for idx, file_name in enumerate(file_list):
            try:
                # Get full file path
                file_path = self._get_file_path(file_name)
                
                # Get metadata from events DataFrame (take first matching row since all metadata is identical)
                file_events = events_df[events_df['file'] == file_name]
                file_meta = file_events.iloc[0]
                kic_id = str(file_meta['id'])
                quarter = str(file_meta['quarter'])

                with fits.open(file_path) as f:
                    time = f[1].data['TIME'].astype(np.float32)
                    # round time to 4 decimal places to match the time in the flare events
                    time = np.round(time, 4)
                    flux = f[1].data['PDCSAP_FLUX'].astype(np.float32)
                    
                    # Calculate and store median before normalization
                    flux_median = float(np.nanmedian(flux))
                    
                    # Normalize flux by median value
                    flux = flux / flux_median
                    
                    # Initialize label sequence
                    labels = np.zeros(len(time), dtype=bool)
                    
                    # Mark time periods containing flare events
                    for _, event in file_events.iterrows():
                        event_mask = (time >= event['start']) & (time <= event['end'])
                        labels[event_mask] = True
                    
                    yield idx, {
                        'flux_norm': flux.tolist(),
                        'label': labels.tolist(),
                        'metadata': {
                            'kic_id': kic_id,
                            'quarter': quarter,
                            'file_name': file_name,
                            'flux_median': flux_median,
                        }
                    }
                    
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                # Skip this sample
                continue
