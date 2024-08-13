# Plant Traits Prediction

Before running the project, ensure you have all the necessary Python dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```

## Feature Extraction

In this project, I apply feature extraction using [DINOv2](https://github.com/facebookresearch/dinov2/tree/main), to extract image embeddings from the original image data.

The feature can be extracted using the following command.

```bash
python feature_extraction.py --data_path ./data --model_version dinov2_vitl14_reg
```

or

```bash
sh feature_extraction.sh
```

## Training CatBoost

CatBoost is used in the training process. Running the following command can perform the whole process from data loading, model training and evaluation. The submission file will be generated at the end. Remember to have the image embeddings ready before running this.

```bash
python main.py --data_path ./data --suffix dinov2_vitl14_reg --model_type catboost
```

or

```bash
sh main.sh
```

You can view the detailed project report [here](report.pdf).
