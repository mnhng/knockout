# Knockout: A simple way to handle missing inputs

[Knockout](https://arxiv.org/abs/2405.20448) addresses missing data by randomly replacing input features with placeholders during training, effectively learning both conditional and marginal distributions, and offering a computationally efficient implicit marginalization strategy for robust performance.

## Usage Examples

Knockout is designed for easy integration into existing PyTorch models.
A few points to note:
- If the missing input features are missing completely at random (MCAR), mask them using NaN (`torch.nan`).
- If the missing input features are MAR or MNAR, mask them using Inf (`torch.inf`).
- Like Dropout, it is important to the model to evaluation model (i.e. `.eval()`) during testing and training mode during training (i.e. `.train()`)

**1. Continuous features:**

```python
from blocks import KnockoutContinuousUnbounded

# assuming training_data is a tensor of size [num_samples, num_features]
mean, std = training_data.nanmean(), training_data.std()
knockout_layer = KnockoutContinuousUnbounded(mean, std, gap=10)  # use the default knockout rate

model = nn.Sequential(
            knockout_layer,
            nn.Linear(in_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1))
```

**2. Categorical feature:**

```python
from blocks import KnockoutEmbedding

knockout_layer = KnockoutEmbedding(num_embeddings=15, embedding_dim=20, p=0.5)  # knockout rate = 0.5
```


## Experiments and Visualizations

### 1. Regression Experiment with Continuous Input Features

- Launch experiment using `continuous_data_imputation_comparison.py`. For example, to use `Knockout` method with `MCAR` data, run:
```python
python continuous_data_imputation_comparison.py -d MCAR -m Knockout -o result_dir_name
```

- Visualize results using `visualize_continuous_data_imputation.py`

### 2. Binary Classification Experiments

- Continuous input features: `continuous_input_binary_classification.py`
- Continuous and categorical input features: `mixed_input_binary_classification.py`


If you find this code useful, please cite
```bibtex
@article{nguyen2024knockout,
  title={Knockout: A simple way to handle missing inputs},
  author={Nguyen, Minh and Karaman, Batuhan K and Kim, Heejong and Wang, Alan Q and Liu, Fengbei and Sabuncu, Mert R},
  journal={arXiv preprint arXiv:2405.20448},
  year={2024}
}
```