import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pipelines.flare_detection.pipelines as pipelines
from transformers import AutoModel
from transformers.pipelines import SUPPORTED_TASKS
from src.models import FCN4FlareModel
from datasets import load_dataset
import numpy as np

# Register the pipeline
SUPPORTED_TASKS["flare-detection"] = {
    "impl": pipelines.FlareDetectionPipeline,
    "tf": (),
    "pt": (FCN4FlareModel,),  # PyTorch model
    "default": {
        "model": "Maxwell-Jia/fcn4flare",
    },
}

def plot_flare_events(time, flux, flare_events, window_width=20, figsize=(8, 6), save_path=None):
    """Plot each flare event in a separate subplot.
    
    Args:
        time (np.ndarray): Time array
        flux (np.ndarray): Flux array
        flare_events (list): List of dictionaries containing flare event information
        window_width (int): Number of points to show before and after the flare
        figsize (tuple): Size of each subplot (width, height)
        save_path (str, optional): Path to save the figure. If None, display the plot
    """
    import matplotlib.pyplot as plt
    
    n_events = len(flare_events)
    if n_events == 0:
        print("No flare events to plot.")
        return
        
    # Create figure with subplots
    n_cols = 2  # Number of columns
    n_rows = (n_events + n_cols - 1) // n_cols  # Ceiling division
    fig = plt.figure(figsize=(figsize[0] * n_cols, figsize[1] * n_rows))
    
    for i, event in enumerate(flare_events, 1):
        start_idx = event['start_idx']
        end_idx = event['end_idx']
        
        # Calculate window indices
        window_start = max(0, start_idx - window_width)
        window_end = min(len(flux), end_idx + window_width)
        
        # Create subplot
        ax = fig.add_subplot(n_rows, n_cols, i)
        
        # Plot full light curve segment in window
        # Non-flare points
        ax.plot(time[window_start:window_end], 
                flux[window_start:window_end], 
                'k.',  # black dots
                markersize=3)
        ax.plot(time[window_start:window_end], 
                flux[window_start:window_end], 
                'k:',  # dotted line
                alpha=0.5,
                linewidth=1)
        
        # Flare segment
        ax.plot(time[start_idx:end_idx+1], 
               flux[start_idx:end_idx+1], 
               'r.',  # red dots
               markersize=3)
        ax.plot(time[start_idx:end_idx+1], 
               flux[start_idx:end_idx+1], 
               'r:',  # red dotted line
               alpha=0.5,
               linewidth=1)
        
        # Add event information
        duration = time[end_idx] - time[start_idx]
        amplitude = flux[event['peak_idx']] - flux[start_idx]
        ax.set_title(f'Flare {i}\nDuration: {duration:.2f} days\nAmplitude: {amplitude:.3f}')
        
        # Customize subplot
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (BJD - 2454833)')
        if i % n_cols == 1:  # Only add y-label for leftmost plots
            ax.set_ylabel('Normalized Flux')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def load_lightcurve(file):
    with fits.open(file) as hdul:
        time = hdul[1].data['TIME'].astype(np.float32)
        flux = hdul[1].data['PDCSAP_FLUX'].astype(np.float32)
        flux = flux / np.nanmedian(flux)
        return time, flux
    
dataset = load_dataset("Maxwell-Jia/kepler_flare")
print(np.where(dataset["train"][0]["label"])[0])

file = "/mnt/lamostgpu/data/Yang_lightcurves/0053/005357275/kplr005357275-2011177032512_llc.fits"
time, flux = load_lightcurve(file)

model = AutoModel.from_pretrained("Maxwell-Jia/fcn4flare", trust_remote_code=True)

flare_detection_pipeline = pipelines.FlareDetectionPipeline(model=model)

result = flare_detection_pipeline.predict([[
    "/mnt/lamostgpu/data/Yang_lightcurves/0053/005357275/kplr005357275-2011177032512_llc.fits"
]])
print(result)

plot_flare_events(time, flux, result[0])