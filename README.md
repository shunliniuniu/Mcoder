# The code provides an example of fine-tuning and evaluating DeepSeekMath 7B.

## Installation

### 1. Create Conda Environment

```bash
conda create -n math python=3.10
```

### 2. Activate the environment:
```bash
conda activate math
```
### 3.Install torch from PyTorch.
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```
### 4. Install the required packages:
```bash
pip install -r train/requirements.txt
```
## train
```bash
bash train/train.sh
```
## eval
```bash
python eval/deepsat.py
python eval/sat.py
```

