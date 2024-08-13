import pandas as pd
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
import os

def get_image_embeddings(model, preprocess, batch_size, df, device):
    image_embeddings = []
    for i in tqdm(range(0, len(df), batch_size)):
        paths = df['file_path'][i:i + batch_size]
        image_tensor = torch.stack([preprocess(Image.open(path)) for path in paths]).to(device)
        with torch.no_grad():
            curr_image_embeddings = model(image_tensor)
        image_embeddings.extend(curr_image_embeddings.cpu().numpy())
    return image_embeddings

def main():
    # Argument parser for command-line options
    parser = argparse.ArgumentParser(description='Feature extraction using pretrained model')
    parser.add_argument('--model_version', type=str, choices=['dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg'],
                        default='dinov2_vits14_reg', help='Specify the model version to use')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the directory containing train.csv and test.csv')
    parser.add_argument('--test_size', type=int, default=2196)
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Construct paths to the train and test CSV files
    train_csv_path = os.path.join(args.data_path, 'train.csv')
    test_csv_path = os.path.join(args.data_path, 'test.csv')

    # Load the train and test data
    train = pd.read_csv(train_csv_path)
    train['file_path'] = train['id'].apply(lambda s: f'{args.data_path}/train_images/{s}.jpeg')

    test = pd.read_csv(test_csv_path)
    test['file_path'] = test['id'].apply(lambda s: f'{args.data_path}/test_images/{s}.jpeg')

    train, val = train_test_split(train, test_size=args.test_size, shuffle=True, random_state=42)
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    # Load the specified model
    model = torch.hub.load('facebookresearch/dinov2', args.model_version).to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Extract embeddings for train, validation, and test datasets
    train_image_embeddings = get_image_embeddings(model, preprocess, args.batch_size, train, device)
    np.save(f'train_{args.model_version}', np.array(train_image_embeddings))
    
    val_image_embeddings = get_image_embeddings(model, preprocess, args.batch_size, val, device)
    np.save(f'val_{args.model_version}', np.array(val_image_embeddings))
    
    test_image_embeddings = get_image_embeddings(model, preprocess, args.batch_size, test, device)
    np.save(f'test_{args.model_version}', np.array(test_image_embeddings))

if __name__ == "__main__":
    main()
