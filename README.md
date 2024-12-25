# FCN4Flare

[![GitHub](https://img.shields.io/badge/-Homepage-181717?style=flat&logo=github)](https://github.com/NAOC-LAMOST/fcn4flare)
[![arXiv](https://img.shields.io/badge/-arXiv-B31B1B?style=flat&logo=arxiv)](https://arxiv.org/abs/2407.21240)
[![Dataset](https://img.shields.io/badge/-Dataset-FF9D00?style=flat&logo=huggingface)](https://huggingface.co/datasets/Maxwell-Jia/kepler_flare)
[![Model](https://img.shields.io/badge/-Model-7957D5?style=flat&logo=huggingface)](https://huggingface.co/Maxwell-Jia/fcn4flare)
[![W&B](https://img.shields.io/badge/-WandB-326CE5?style=flat&logo=weightsandbiases)](https://wandb.ai/maxwell-jia/keler_flare_detection/reports/FCN4Flare-Fully-Convolutional-Neural-Networks-for-Flare-Detection--VmlldzoxMDcxODg2NQ)

> **Click the buttons above to access:**
> - 📂 Source code (Homepage)
> - 📄 Research paper (arXiv)
> - 📊 Dataset (Hugging Face)
> - 🤖 Pre-trained models (Hugging Face)
> - 📈 Training logs (Weights & Biases)

This repository contains the official implementation of the paper: [FCN4Flare: Fully Convolution Neural Networks for Flare Detection](https://arxiv.org/abs/2407.21240).

> ⚠️ **Important Note:**
> - The original implementation (used in the paper) was based on PyTorch Lightning with random dataset splits, and is now available in the `lightning_version` directory.
> - We have recently restructured the project using the Hugging Face ecosystem to improve accessibility and reproducibility:
>   - Code has been refactored using Hugging Face's frameworks
>   - Dataset has been reorganized into an AI-Ready format
>   - Pre-trained models are now available through Hugging Face Hub
> - Due to these improvements, results using our pre-trained models might show slight variations from those reported in the paper, though the differences are not significant.


## **Overview**

FCN4Flare is a fully convolutional neural network designed for precise point-to-point detection of stellar flares in photometric time-series data. Stellar flares provide valuable insights into stellar magnetic activity and space weather environments, but detecting these flares is challenging due to missing data, imbalanced classes, and diverse flare morphologies. FCN4Flare addresses these challenges with:

- **NaN Mask**: A mechanism to handle missing data points effectively during training.
- **Mask Dice Loss**: A loss function tailored to mitigate class imbalance by optimizing the overlap between predicted and true flares.

FCN4Flare achieves state-of-the-art performance on the Kepler flare dataset, significantly surpassing previous methods such as Flatwrm2 and Stella. Key performance metrics are summarized in the table below:

| Metric               | FCN4Flare | Flatwrm2 | Stella |
|----------------------|-----------|----------|--------|
| Recall               | **0.67**  | 0.26     | 0.50   |
| Precision            | **0.69**  | 0.08     | 0.09   |
| F1 Score             | **0.64**  | 0.13     | 0.16   |
| Average Precision    | **0.55**  | 0.12     | 0.14   |
| Dice Coefficient     | **0.64**  | 0.12     | 0.15   |
| Intersection over Union (IoU) | **0.54**  | 0.10     | 0.13   |

These results demonstrate FCN4Flare's robustness in accurately detecting flares while maintaining computational efficiency. With its open-source implementation and publicly available dataset, FCN4Flare is a valuable tool for stellar physics and time-domain astronomy research.


## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/NAOC-LAMOST/fcn4flare.git
   cd fcn4flare
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## **Usage**

### **1. Reproduce the Results**
To reproduce the results in the paper, run:
```bash
./scripts/train.sh
```

### **2. Inference with AutoModel API**

```python
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("Maxwell-Jia/fcn4flare")

# Load your data and create required tensors
# You need to implement your own data loading logic that returns:
# 1. input_features: tensor of flux values, shape [batch_size, sequence_length, 1]
#    - Contains the actual flux measurements and padded values
# 2. sequence_mask: binary tensor, shape [batch_size, sequence_length]
#    - 1 indicates real flux values
#    - 0 indicates padded positions
input_features, sequence_mask = load_data()

# Example of expected tensor shapes and values:
# input_features = torch.tensor([
#     [1.2, 1.5, 1.1, nan, nan],  # nan are padded values
#     [1.3, 1.4, 1.6, 1.2, 1.1]   # all real values
# ])
# sequence_mask = torch.tensor([
#     [1, 1, 1, 0, 0],  # last 2 positions are padded
#     [1, 1, 1, 1, 1]   # no padding
# ])

logits = model(input_features, sequence_mask)

# Apply a threshold to get binary predictions
threshold = 0.5
predictions = (logits > threshold).float()

# Implement your own post-processing logic to reduce false positives
# The post-processing step is crucial for:
# 1. Filtering out noise and spurious detections
# 2. Merging nearby detections
# 3. Applying additional threshold or rule-based filtering
#
# Example post-processing strategies:
# - Apply minimum duration threshold
# - Merge events that are too close in time
# - Consider the amplitude of the detected events
# - Use domain knowledge to validate detections
final_results = post_process_predictions(predictions)

# Example implementation:
# def post_process_predictions(predictions):
#     # Apply minimum duration filter
#     # Remove detections shorter than X minutes
#     # Merge events within Y minutes of each other
#     # Apply additional validation rules
#     return processed_results
```

### **3. Inference with pipeline API**

```python
from transformers import pipeline

flare_detector = pipeline("flare-detection", model="Maxwell-Jia/fcn4flare")
# Only surport for Kepler/K2 light curves now.
results = flare_detector([
    "Path/to/your/lightcurve.fits",
    "Path/to/your/lightcurve.fits",
    ...
])

print(results)
```


## **Citation**

If you find this work useful, please cite our paper:
```bibtex
@article{jia2024fcn4flare,
  title={FCN4Flare: Fully Convolution Neural Networks for Flare Detection},
  author={Minghui Jia, A-Li Luo, Bo Qiu},
  journal={arXiv preprint arXiv:2407.21240},
  year={2024}
}
```


## **License**
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.


## **Dissatisfaction**

We feel compelled to share our frustrating experience with the initial publication process of this work:

This paper was first submitted to ApJ over a year ago, where we encountered an extremely disappointing review process:
- First review dragged on for nearly 6 months
- After our revision, the reviewer suddenly became "unavailable"
- Finding a new reviewer caused further unnecessary delays
- Almost a year was wasted before the final rejection

This experience highlights the significant challenges in the astronomical peer review system, particularly for interdisciplinary research combining astronomy with modern AI techniques. While the paper has now been accepted by MNRAS, such extensive delays in the review process can severely impede scientific progress, especially in rapidly evolving fields like AI.

We strongly advocate for more open-source practices in astronomical research. By making our entire project (code, data, models and even traing logs) publicly available, we hope to encourage more transparent and efficient knowledge sharing within the astronomy community. We believe this is crucial for advancing our field and fostering better collaboration between astronomy and modern AI research.