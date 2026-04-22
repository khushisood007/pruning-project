# Self-Pruning Neural Network

## Features
- Automatic weight pruning during training  
- Reduced model complexity  
- Improved efficiency and smaller model size  
- Implemented using PyTorch  

## Tech Stack
- Python  
- PyTorch  
- NumPy  
- Matplotlib  

## Project Structure
pruning-project/
│── pruning_nn.ipynb
│── README.md
## How to Run

### Option 1: Run on Google Colab
Open the notebook directly in Colab:  
https://colab.research.google.com/github/khushisood007/pruning-project/blob/main/pruning_nn.ipynb  

### Option 2: Run Locally

Clone the repository:
git clone https://github.com/khushisood007/pruning-project.git

cd pruning-project
Install dependencies:
pip install torch torchvision matplotlib numpy
Launch Jupyter Notebook:

jupyter notebook
Run all cells in `pruning_nn.ipynb` sequentially to train the model and observe pruning behavior, performance metrics, and sparsity levels.

## Description
The model introduces sparsity by pruning less significant weights during training. This reduces redundancy in the network, making it more efficient without a major drop in accuracy. The approach is useful for optimizing models for deployment in resource-constrained environments.

## Future Improvements
- Improve accuracy using advanced architectures  
- Experiment with structured and dynamic pruning techniques  
- Optimize training and inference time  
- Deploy the model as an application  
