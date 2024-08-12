import pandas as pd
import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm
from catboost import Pool, CatBoostRegressor
import argparse

class Config():
    TARGET_COLUMNS = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']
    VAL_SAMPLES = 2196
    SEED = 42
    DEVICE = 'mps' if  torch.backends.mps.is_available()  else 'cpu'
    
def seed_everything(seed):    
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)

def load_data(data_path):
    train = pd.read_csv(f'{data_path}/train.csv')
    train['file_path'] = train['id'].apply(lambda s: f'{os.path.dirname(data_path)}/train_images/{s}.jpeg')
    
    test = pd.read_csv(f'{data_path}/test.csv')
    test['file_path'] = test['id'].apply(lambda s: f'{os.path.dirname(data_path)}/test_images/{s}.jpeg')
    
    return train, test

def preprocess_data(train, test, config):
    train, val = train_test_split(train, test_size=config.VAL_SAMPLES, shuffle=True, random_state=config.SEED)
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    feature_columns = test.columns.values[1:-1]
    
    # Standard Scaler for Features
    feature_scaler = StandardScaler()
    train_features = feature_scaler.fit_transform(train[feature_columns].values.astype(np.float32))
    val_features = feature_scaler.transform(val[feature_columns].values.astype(np.float32))
    test_features = feature_scaler.transform(test[feature_columns].values.astype(np.float32))

    y_train = train[config.TARGET_COLUMNS].values
    y_val = val[config.TARGET_COLUMNS].values

    return train_features, val_features, test_features, y_train, y_val, feature_columns

def load_embeddings(suffix):
    train_image_embeddings = np.load(f'./train_{suffix}.npy')
    val_image_embeddings = np.load(f'./val_{suffix}.npy')
    test_image_embeddings = np.load(f'./test_{suffix}.npy')
    return train_image_embeddings, val_image_embeddings, test_image_embeddings

def prepare_features(train_features, val_features, test_features, train_image_embeddings, val_image_embeddings, test_image_embeddings, first_n_poly_feats):
    train_features_all = np.concatenate(
        (PolynomialFeatures(2).fit_transform(train_features)[:, :first_n_poly_feats], train_image_embeddings), axis=1
    )
    val_features_all = np.concatenate(
        (PolynomialFeatures(2).fit_transform(val_features)[:, :first_n_poly_feats], val_image_embeddings), axis=1
    )
    test_features_all = np.concatenate(
        (PolynomialFeatures(2).fit_transform(test_features)[:, :first_n_poly_feats], test_image_embeddings), axis=1
    )

    train_features_df = pd.DataFrame(train_features_all)
    train_features_df['embeddings'] = list(train_image_embeddings)

    val_features_df = pd.DataFrame(val_features_all)
    val_features_df['embeddings'] = list(val_image_embeddings)

    test_features_df = pd.DataFrame(test_features_all)
    test_features_df['embeddings'] = list(test_image_embeddings)

    return train_features_df, val_features_df, test_features_df

def train_and_evaluate(train_features_df, val_features_df, y_train, y_val, config, learning_rate):
    models = {}
    scores = {}
    for i, col in tqdm(enumerate(config.TARGET_COLUMNS), total=len(config.TARGET_COLUMNS)):
        y_curr = y_train[:, i]
        y_curr_val = y_val[:, i] 
        train_pool = Pool(train_features_df, y_curr, embedding_features=['embeddings'])
        val_pool = Pool(val_features_df, y_curr_val, embedding_features=['embeddings'])

        model = CatBoostRegressor(iterations=1000, learning_rate=learning_rate, loss_function='RMSE', verbose=0, random_state=config.SEED)
        model.fit(train_pool)
        models[col] = model
        
        y_curr_val_pred = model.predict(val_pool)
        
        r2_col = r2_score(y_curr_val, y_curr_val_pred)
        scores[col] = r2_col
        print(f'Target: {col}, R2: {r2_col:.3f}')

    print(f'Mean R2: {np.mean(list(scores.values())):.3f}')
    return models

def create_submission(models, test, test_features_df, config):
    submission = pd.DataFrame({'id': test['id']})
    submission[config.TARGET_COLUMNS] = 0
    submission.columns = submission.columns.str.replace('_mean', '')
    for i, col in enumerate(config.TARGET_COLUMNS):
        test_pool = Pool(test_features_df, embedding_features=['embeddings'])
        col_pred = models[col].predict(test_pool)
        submission[col.replace('_mean', '')] = col_pred

    submission.to_csv('submission.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description='Training script for regression model.')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the directory containing train.csv and test.csv')
    parser.add_argument('--suffix', type=str, default='dinov2_vitb14_reg', choices=['dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg'], help='Suffix for embedding files')
    parser.add_argument('--learning-rate', type=float, default=0.06, help='Learning rate for the model')
    parser.add_argument('--first-n-poly-feats', type=int, default=1000, help='Number of polynomial features to use')

    args = parser.parse_args()

    config = Config()
    seed_everything(config.SEED)
    
    print("Loading data...")
    train, test = load_data(args.data_path)
    print(f"Training data shape: {train.shape}")
    print(f"Test data shape: {test.shape}")
    print("Data loading completed!")
    
    print("Data preprocessing...")
    train_features_mask, val_features_mask, test_features, y_train_mask, y_val_mask, feature_columns = preprocess_data(train, test, config)
    print(f"Train feature shape: {train_features_mask.shape}")
    print(f"Validation feature shape: {val_features_mask.shape}")
    print(f"Test feature shape: {test_features.shape}")
    print("Data preprocessing completed!")
    
    print(f"Loading embeddings with suffix '{args.suffix}'...")
    train_image_embeddings, val_image_embeddings, test_image_embeddings = load_embeddings(args.suffix)
    print(f"Train embeddings shape: {train_image_embeddings.shape}")
    print(f"Validation embeddings shape: {val_image_embeddings.shape}")
    print(f"Test embeddings shape: {test_image_embeddings.shape}")
    print("Embeddings loading completed!")
    
    print("Preparing features...")
    train_features_mask_df, val_features_mask_df, test_features_mask_df = prepare_features(
        train_features_mask, val_features_mask, test_features,
        train_image_embeddings, val_image_embeddings, test_image_embeddings,
        args.first_n_poly_feats
    )
    print(f"Train features after preparation shape: {train_features_mask_df.shape}")
    print(f"Validation features after preparation shape: {val_features_mask_df.shape}")
    print(f"Test features after preparation shape: {test_features_mask_df.shape}")
    print("Feature preparation completed!")

    print("Training and evaluating models...")
    models = train_and_evaluate(train_features_mask_df, val_features_mask_df, y_train_mask, y_val_mask, config, args.learning_rate)
    print("Model training and evaluation completed!")
    
    print("Creating submission file...")
    create_submission(models, test, test_features_mask_df, config)
    print("Submission file created successfully!")

if __name__ == '__main__':
    main()
