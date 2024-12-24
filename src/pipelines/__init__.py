from transformers.pipelines import SUPPORTED_TASKS, Pipeline

from ..models import FCN4FlareModel
from .flare_detection.pipelines import FlareDetectionPipeline

# Register the pipeline
SUPPORTED_TASKS["flare-detection"] = {
    "impl": FlareDetectionPipeline,
    "tf": (),
    "pt": (FCN4FlareModel,),  # PyTorch model
    "default": {
        "model": "Maxwell-Jia/fcn4flare",
    },
}