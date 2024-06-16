# CS680 A2

## Environment

- **Python**: 3.11.9
- **Torch**: 2.3.1
- **Torchvision**: 0.18.1

All the training and testing are performed using a Macbook Air M2, accelerated using MPS (Metal Performance Shaders).

## How to Use

### Running the Scripts

You can perform all the necessary training and testing by executing the `script.sh` file. This script will handle the training and testing processes for both the regular and augmented datasets.

### Training and Testing

1. **Regular Training**:
   - Models will be saved to the `./checkpoints` directory.
   - Run `script.sh` without any additional arguments for regular training.

2. **Augmented Training**:
   - Data augmentation techniques are applied during training to improve model robustness.
   - Models will be saved to the `./checkpoints_augmented` directory.
   - Specify the augmentation mode in the `script.sh` to enable augmented training.

### Graphs and Metrics

- The graphs for training and testing metrics (such as accuracy and loss) will be plotted at the end of each training or testing session.
- Ensure you have all necessary dependencies installed for plotting the graphs.

## Directory Structure

- **./checkpoints**: Directory where models trained without data augmentation are saved.
- **./checkpoints_augmented**: Directory where models trained with data augmentation are saved.

## Dependencies

Make sure to install the required dependencies using the following command:

```sh
pip install torch torchvision matplotlib tqdm
