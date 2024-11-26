# MORPHEUS
MORPHEUS: Modeling Role from Personalized Dialogue History by Exploring and Utilizing Latent Space(EMNLP 2024)

[Paper](https://aclanthology.org/2024.emnlp-main.437/) 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

MORPHEUS is a role modeling framework based on personalized dialogue history. The model captures role characteristics in conversations by exploring and utilizing latent space to generate more coherent and personalized responses.

## Key Features

- Encoder-decoder architecture for exploring dialogue latent space
- Contrastive learning to enhance role representation distinctiveness  

## Requirements
```
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data & Models

Download the models:
```
./models/gpt2-chinese-cluecorpussmall
./models/gpt2
./models/bert-base-uncased
./models/bert-base-chinese
```

Download the datasets:

EN: [ConvAI2](https://huggingface.co/datasets/Toyhom/MORPHEUS_Datasets)
ZH: [Baidu PersonaChat](https://huggingface.co/datasets/Toyhom/MORPHEUS_Datasets)

### 2. Run the code

```
python main.py
```

**Model Arguments**

Key arguments for training:

- `--lang`: Language choice ['en', 'zh']
- `--num_epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--lr`: Learning rate
- `--max_condition_length`: Max length for condition input
- `--max_input_length`: Max length for model input
- `--initial_method`: Initialization method
- `--codebook_num`: Number of codebook entries
- `--seq_len`: Sequence length for role modeling
- `--peft`: Parameter-efficient fine-tuning method [None, 'lora', 'p-tuning', 'prefix-tuning', 'prompt-tuning']

### 3. Evaluate the model
```
python Metrics.py
```

