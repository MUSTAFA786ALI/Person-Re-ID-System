import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage import io
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, filename='reid.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    """Configuration class for hyperparameters and paths"""
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 0.00005
    INPUT_SIZE = (128, 128)
    MARGIN = 1.0
    EMBEDDING_DIM = 128
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = min(4, os.cpu_count() or 1)

    # Test thresholds
    POS_DIST_THRESHOLD = 0.5
    NEG_DIST_THRESHOLD = 1.0
    NOISE_DIST_THRESHOLD = 0.6
    OCCL_DIST_THRESHOLD = 0.7
    TRIPLET_LOSS_THRESHOLD = 0.2
    TOP5_ACCURACY_THRESHOLD = 0.8

def setup_paths(data_dir=None, csv_path=None):
    """Setup data paths with flexible input and validation"""
    if data_dir and csv_path and os.path.exists(data_dir) and os.path.exists(csv_path):
        # Validate image files in CSV
        df = pd.read_csv(csv_path)
        required_columns = ['Anchor', 'Positive', 'Negative']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV must contain 'Anchor', 'Positive', 'Negative' columns")
        for col in required_columns:
            for img_path in df[col]:
                full_path = os.path.join(data_dir, img_path)
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"Image not found: {full_path}")
        return data_dir, csv_path

    possible_base_dirs = [
        os.getcwd(),
        os.path.join(os.getcwd(), "Person-Re-Id-Dataset"),
    ]

    for base_dir in possible_base_dirs:
        potential_data_dir = os.path.join(base_dir, "train")
        potential_csv_path = os.path.join(base_dir, "train.csv")
        if os.path.exists(potential_data_dir) and os.path.exists(potential_csv_path):
            df = pd.read_csv(potential_csv_path)
            required_columns = ['Anchor', 'Positive', 'Negative']
            if not all(col in df.columns for col in required_columns):
                continue
            for col in required_columns:
                for img_path in df[col]:
                    full_path = os.path.join(potential_data_dir, img_path)
                    if not os.path.exists(full_path):
                        logging.warning(f"Image not found: {full_path}")
                        continue
            return potential_data_dir, potential_csv_path

    raise FileNotFoundError(
        "Could not find train directory and CSV file.\n"
        "Please specify paths or ensure data is in the correct structure."
    )

class APN_Dataset(Dataset):
    """Anchor-Positive-Negative triplet dataset with augmentation"""

    def __init__(self, df, data_dir, input_size=(128, 128), augment=True, triplet_mode=True):
        self.df = df
        self.data_dir = data_dir
        self.augment = augment
        self.input_size = input_size
        self.triplet_mode = triplet_mode

        if not triplet_mode:
            self.image_list = df['image'].tolist()
            self.labels = df['person_id'].tolist()
            self.unique_labels = list(set(self.labels))

        base_transforms = [
            transforms.ToPILImage(),
            transforms.Resize(input_size),
        ]

        if augment:
            base_transforms.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.Lambda(self.random_occlusion),
            ])

        base_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.transform = transforms.Compose(base_transforms)

    def random_occlusion(self, img):
        """Apply random occlusion augmentation"""
        if np.random.rand() > 0.5:
            img = np.array(img)
            h, w = img.shape[:2]
            occl_h, occl_w = int(h * np.sqrt(0.2)), int(w * np.sqrt(0.2))
            x = np.random.randint(0, h - occl_h)
            y = np.random.randint(0, w - occl_w)
            img[x:x+occl_h, y:y+occl_w] = 0
            return img
        return img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.triplet_mode:
            return self._get_triplet_item(idx)
        else:
            return self._generate_triplet(idx)

    def _get_triplet_item(self, idx):
        try:
            row = self.df.iloc[idx]
            A_path = os.path.join(self.data_dir, row['Anchor'])
            P_path = os.path.join(self.data_dir, row['Positive'])
            N_path = os.path.join(self.data_dir, row['Negative'])

            if not all(os.path.exists(p) for p in [A_path, P_path, N_path]):
                raise FileNotFoundError(f"Missing image at index {idx}")

            A_img = io.imread(A_path)
            P_img = io.imread(P_path)
            N_img = io.imread(N_path)

            A_img = self.transform(A_img)
            P_img = self.transform(P_img)
            N_img = self.transform(N_img)

            return A_img, P_img, N_img
        except Exception as e:
            logging.error(f"Error loading item {idx}: {str(e)}")
            return None

    def _generate_triplet(self, idx):
        try:
            anchor_img = io.imread(os.path.join(self.data_dir, self.image_list[idx]))
            anchor_label = self.labels[idx]

            positive_indices = [i for i, label in enumerate(self.labels) if label == anchor_label and i != idx]
            if not positive_indices:
                logging.warning(f"No positive sample for index {idx}")
                return None
            positive_idx = np.random.choice(positive_indices)
            positive_img = io.imread(os.path.join(self.data_dir, self.image_list[positive_idx]))

            negative_label = np.random.choice([l for l in self.unique_labels if l != anchor_label])
            negative_indices = [i for i, label in enumerate(self.labels) if label == negative_label]
            if not negative_indices:
                logging.warning(f"No negative sample for index {idx}")
                return None
            negative_idx = np.random.choice(negative_indices)
            negative_img = io.imread(os.path.join(self.data_dir, self.image_list[negative_idx]))

            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

            return anchor_img, positive_img, negative_img
        except Exception as e:
            logging.error(f"Error generating triplet for index {idx}: {str(e)}")
            return None

def custom_collate(batch):
    """Custom collate function to skip None samples"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

class APN_Model(nn.Module):
    """CNN model for generating embeddings"""

    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Conv2d(3, 64, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.fc1 = nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, embedding_dim)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        x = F.normalize(x, p=2, dim=1)
        return x

class TripletLoss(nn.Module):
    """Triplet loss implementation"""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return torch.mean(loss)

class PersonReIDTrainer:
    """Training and evaluation class"""

    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE

    def train_epoch(self, model, dataloader, optimizer, criterion):
        """Training function for one epoch"""
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training"):
            if batch is None:
                continue
            try:
                A_img, P_img, N_img = [x.to(self.device) for x in batch]
                optimizer.zero_grad()

                A_emb = model(A_img)
                P_emb = model(P_img)
                N_emb = model(N_img)

                loss = criterion(A_emb, P_emb, N_emb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            except Exception as e:
                logging.error(f"Training batch error: {str(e)}")
                continue
        return total_loss / max(1, len(dataloader))

    def evaluate(self, model, dataloader, criterion):
        """Evaluation function"""
        model.eval()
        total_loss = 0
        all_pos_dists = []
        all_neg_dists = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                if batch is None:
                    continue
                try:
                    A_img, P_img, N_img = [x.to(self.device) for x in batch]
                    A_emb = model(A_img)
                    P_emb = model(P_img)
                    N_emb = model(N_img)

                    loss = criterion(A_emb, P_emb, N_emb)
                    total_loss += loss.item()

                    pos_dist = torch.sum((A_emb - P_emb) ** 2, dim=1).cpu().numpy()
                    neg_dist = torch.sum((A_emb - N_emb) ** 2, dim=1).cpu().numpy()
                    all_pos_dists.extend(pos_dist)
                    all_neg_dists.extend(neg_dist)
                except Exception as e:
                    logging.error(f"Validation batch error: {str(e)}")
                    continue

        avg_loss = total_loss / max(1, len(dataloader))
        pos_mean = np.mean(all_pos_dists) if all_pos_dists else float('inf')
        neg_mean = np.mean(all_neg_dists) if all_neg_dists else 0
        return avg_loss, pos_mean, neg_mean, all_pos_dists, all_neg_dists

    def train(self, model, train_loader, valid_loader, criterion, optimizer, epochs, progress_callback=None):
        """Full training loop"""
        train_losses = []
        valid_losses = []
        best_valid_loss = float('inf')

        for epoch in range(epochs):
            try:
                print(f"\nEpoch {epoch+1}/{epochs}")
                train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
                valid_loss, pos_mean, neg_mean, _, _ = self.evaluate(model, valid_loader, criterion)

                train_losses.append(train_loss)
                valid_losses.append(valid_loss)

                print(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
                print(f"Avg Positive Distance: {pos_mean:.4f}, Avg Negative Distance: {neg_mean:.4f}")

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), 'best_model.pt')
                    print("Best model saved!")

                if progress_callback:
                    progress_callback(epoch + 1, epochs, train_loss, valid_loss)

            except Exception as e:
                logging.error(f"Error in epoch {epoch+1}: {str(e)}")
                break

        return train_losses, valid_losses

class PersonReIDTester:
    """Test case implementation for TC01-TC06"""

    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE

    def plot_test_images(self, img1, img2, title1, title2, distance=None, save_prefix="test"):
        """Utility function to plot test case images"""
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img1)
        axes[0].set_title(title1)
        axes[0].axis("off")
        axes[1].imshow(img2)
        axes[1].set_title(title2)
        axes[1].axis("off")
        if distance is not None:
            fig.suptitle(f"Distance: {distance:.4f}")
        plt.savefig(f"{save_prefix}.png")
        plt.close()

    def plot_triplet_images(self, anchor, positive, negative, save_prefix="tc05_triplet"):
        """Utility function to plot triplet for TC05"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(anchor)
        axes[0].set_title("Anchor")
        axes[0].axis("off")
        axes[1].imshow(positive)
        axes[1].set_title("Positive")
        axes[1].axis("off")
        axes[2].imshow(negative)
        axes[2].set_title("Negative")
        axes[2].axis("off")
        plt.savefig(f"{save_prefix}.png")
        plt.close()

    def plot_closest_matches(self, query_img, closest_imgs, distances, save_prefix="tc06_matches"):
        """Plot query image with closest matches"""
        n_matches = len(closest_imgs)
        fig, axes = plt.subplots(1, n_matches + 1, figsize=(3 * (n_matches + 1), 5))

        axes[0].imshow(query_img)
        axes[0].set_title("Query Image")
        axes[0].axis("off")

        for i, (img, dist) in enumerate(zip(closest_imgs, distances)):
            axes[i + 1].imshow(img)
            axes[i + 1].set_title(f"Match {i+1}\nDist: {dist:.3f}")
            axes[i + 1].axis("off")

        plt.tight_layout()
        plt.savefig(f"{save_prefix}.png")
        plt.close()

    def test_distance_thresholds(self, model, valid_loader, data_dir, valid_df):
        """Test TC01 and TC02: Distance thresholds"""
        model.eval()
        pos_dists = []
        neg_dists = []

        with torch.no_grad():
            for i, batch in enumerate(valid_loader):
                if i >= 1 or batch is None:
                    break
                A_img, P_img, N_img = [x.to(self.device) for x in batch]
                A_emb = model(A_img)
                P_emb = model(P_img)
                N_emb = model(N_img)

                pos_dist = torch.sum((A_emb - P_emb) ** 2, dim=1).cpu().numpy()
                neg_dist = torch.sum((A_emb - N_emb) ** 2, dim=1).cpu().numpy()
                pos_dists.extend(pos_dist)
                neg_dists.extend(neg_dist)

                row = valid_df.iloc[i]
                anchor_img = io.imread(os.path.join(data_dir, row['Anchor']))
                positive_img = io.imread(os.path.join(data_dir, row['Positive']))
                negative_img = io.imread(os.path.join(data_dir, row['Negative']))

                self.plot_test_images(anchor_img, positive_img, "Anchor", "Positive",
                                    distance=pos_dist[0], save_prefix="tc01_pos_dist")
                self.plot_test_images(anchor_img, negative_img, "Anchor", "Negative",
                                    distance=neg_dist[0], save_prefix="tc02_neg_dist")

        avg_pos_dist = np.mean(pos_dists) if pos_dists else float('inf')
        avg_neg_dist = np.mean(neg_dists) if neg_dists else 0

        print(f"TC01 - Avg Positive Distance: {avg_pos_dist:.4f} (threshold: {self.config.POS_DIST_THRESHOLD})")
        print(f"TC02 - Avg Negative Distance: {avg_neg_dist:.4f} (threshold: {self.config.NEG_DIST_THRESHOLD})")

        tc01_pass = avg_pos_dist < self.config.POS_DIST_THRESHOLD
        tc02_pass = avg_neg_dist > self.config.NEG_DIST_THRESHOLD

        return avg_pos_dist, avg_neg_dist, tc01_pass, tc02_pass

    def test_noise_robustness(self, model, img_path, sigma=0.1):
        """Test TC03: Noise robustness"""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.config.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img = io.imread(img_path)
        noisy_img = img + np.random.normal(0, sigma, img.shape)
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

        img_tensor = transform(img).to(self.device)
        noisy_tensor = transform(noisy_img).to(self.device)

        model.eval()
        with torch.no_grad():
            img_emb = model(img_tensor.unsqueeze(0))
            noisy_emb = model(noisy_tensor.unsqueeze(0))
            dist = torch.sum((img_emb - noisy_emb) ** 2).cpu().numpy()

        self.plot_test_images(img, noisy_img, "Original", f"Noisy (σ={sigma})",
                            distance=dist, save_prefix="tc03_noise_test")

        print(f"TC03 - Noise Test Distance: {dist:.4f} (threshold: {self.config.NOISE_DIST_THRESHOLD})")
        tc03_pass = dist < self.config.NOISE_DIST_THRESHOLD
        return dist, tc03_pass

    def test_occlusion_robustness(self, model, img_path, occlusion_ratio=0.2):
        """Test TC04: Occlusion robustness"""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.config.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img = io.imread(img_path)
        h, w = img.shape[:2]
        occl_h, occl_w = int(h * np.sqrt(occlusion_ratio)), int(w * np.sqrt(occlusion_ratio))
        occl_img = img.copy()
        start_h, start_w = (h - occl_h) // 2, (w - occl_w) // 2
        occl_img[start_h:start_h + occl_h, start_w:start_w + occl_w] = 0

        img_tensor = transform(img).to(self.device)
        occl_tensor = transform(occl_img).to(self.device)

        model.eval()
        with torch.no_grad():
            img_emb = model(img_tensor.unsqueeze(0))
            occl_emb = model(occl_tensor.unsqueeze(0))
            dist = torch.sum((img_emb - occl_emb) ** 2).cpu().numpy()

        self.plot_test_images(img, occl_img, "Original", f"Occluded ({int(occlusion_ratio*100)}%)",
                            distance=dist, save_prefix="tc04_occlusion_test")

        print(f"TC04 - Occlusion Test Distance: {dist:.4f} (threshold: {self.config.OCCL_DIST_THRESHOLD})")
        tc04_pass = dist < self.config.OCCL_DIST_THRESHOLD
        return dist, tc04_pass

    def test_triplet_loss_threshold(self, final_loss, valid_loader, data_dir, valid_df):
        """Test TC05: Triplet loss threshold"""
        for i, batch in enumerate(valid_loader):
            if i >= 1 or batch is None:
                break
            row = valid_df.iloc[i]
            anchor_img = io.imread(os.path.join(data_dir, row['Anchor']))
            positive_img = io.imread(os.path.join(data_dir, row['Positive']))
            negative_img = io.imread(os.path.join(data_dir, row['Negative']))

            self.plot_triplet_images(anchor_img, positive_img, negative_img, "tc05_triplet_example")
            break

        print(f"TC05 - Final Triplet Loss: {final_loss:.4f} (threshold: {self.config.TRIPLET_LOSS_THRESHOLD})")
        tc05_pass = final_loss < self.config.TRIPLET_LOSS_THRESHOLD
        return tc05_pass

    def euclidean_distance(self, emb1, emb2):
        """Calculate euclidean distance between embeddings"""
        return np.sqrt(np.sum((emb1 - emb2) ** 2))

    def get_embeddings(self, model, img_names, data_dir, batch_size=32):
        """Generate embeddings for a list of images"""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.config.INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        embeddings = []
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(img_names), batch_size), desc="Generating embeddings"):
                batch_names = img_names[i:i + batch_size]
                batch_imgs = []
                for img_name in batch_names:
                    img_path = os.path.join(data_dir, img_name)
                    if not os.path.exists(img_path):
                        logging.warning(f"Image not found: {img_path}")
                        continue
                    img = io.imread(img_path)
                    img_tensor = transform(img).to(self.device)
                    batch_imgs.append(img_tensor)

                if not batch_imgs:
                    continue
                batch_imgs = torch.stack(batch_imgs)
                batch_emb = model(batch_imgs)
                embeddings.extend(batch_emb.cpu().numpy())

        return np.array(embeddings)

    def test_top5_accuracy(self, model, test_df, data_dir, n_samples=50):
        """Test TC06: Top-5 accuracy for person retrieval"""
        print("Generating database embeddings...")
        db_img_names = test_df['Anchor'].values
        db_embeddings = self.get_embeddings(model, db_img_names, data_dir)

        correct_top5 = 0
        total = 0
        query_results = []

        query_indices = np.random.choice(len(test_df), min(n_samples, len(test_df)), replace=False)

        for i, query_idx in enumerate(tqdm(query_indices, desc="Testing Top-5 accuracy")):
            query_row = test_df.iloc[query_idx]
            query_img_name = query_row['Anchor']
            query_person_id = query_img_name.split('_')[0]

            query_emb = db_embeddings[query_idx:query_idx+1]

            distances = []
            for db_emb in db_embeddings:
                dist = self.euclidean_distance(query_emb.squeeze(), db_emb)
                distances.append(dist)

            sorted_indices = np.argsort(distances)
            top5_indices = []
            for idx in sorted_indices:
                if idx != query_idx:
                    top5_indices.append(idx)
                    if len(top5_indices) == 5:
                        break

            top5_person_ids = [db_img_names[idx].split('_')[0] for idx in top5_indices]
            if query_person_id in top5_person_ids:
                correct_top5 += 1

            total += 1

            if i < 3:
                query_img = io.imread(os.path.join(data_dir, query_img_name))
                top5_imgs = []
                top5_dists = []
                for idx in top5_indices:
                    img = io.imread(os.path.join(data_dir, db_img_names[idx]))
                    top5_imgs.append(img)
                    top5_dists.append(distances[idx])

                self.plot_closest_matches(query_img, top5_imgs, top5_dists,
                                        f"tc06_top5_query_{i+1}")

                query_results.append({
                    'query_id': query_person_id,
                    'top5_ids': top5_person_ids,
                    'match_found': query_person_id in top5_person_ids
                })

        accuracy = correct_top5 / total if total > 0 else 0
        print(f"TC06 - Top-5 Accuracy: {accuracy:.4f} (threshold: {self.config.TOP5_ACCURACY_THRESHOLD})")
        print(f"Correct matches: {correct_top5}/{total}")

        for i, result in enumerate(query_results):
            match_status = "✓" if result['match_found'] else "✗"
            print(f"Query {i+1} - Person {result['query_id']}: {match_status}")
            print(f"  Top-5 IDs: {result['top5_ids']}")

        tc06_pass = accuracy > self.config.TOP5_ACCURACY_THRESHOLD
        return accuracy, tc06_pass

    def plot_tsne_visualization(self, model, df, data_dir, n_samples=100):
        """Generate t-SNE visualization of embeddings"""
        print("Generating t-SNE visualization...")

        sample_indices = np.random.choice(len(df), min(n_samples, len(df)), replace=False)
        sample_df = df.iloc[sample_indices]

        img_names = sample_df['Anchor'].values
        embeddings = self.get_embeddings(model, img_names, data_dir)
        person_ids = [name.split('_')[0] for name in img_names]

        if len(embeddings) < 2:
            print("Not enough valid embeddings for t-SNE")
            return

        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        tsne_embeddings = tsne.fit_transform(embeddings)

        plt.figure(figsize=(12, 8))
        unique_ids = list(set(person_ids))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_ids)))

        for i, person_id in enumerate(unique_ids):
            indices = [j for j, pid in enumerate(person_ids) if pid == person_id]
            if len(indices) > 1:
                plt.scatter(tsne_embeddings[indices, 0], tsne_embeddings[indices, 1],
                          c=[colors[i]], label=f'Person {person_id}', s=50, alpha=0.7)

        plt.title('t-SNE Visualization of Person Embeddings')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('tsne_embeddings.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run_all_tests(self, model, valid_loader, data_dir, valid_df, sample_img_path, final_loss):
        """Run all test cases TC01-TC06"""
        print("\n" + "="*50)
        print("RUNNING ALL TEST CASES (TC01-TC06)")
        print("="*50)

        avg_pos_dist, avg_neg_dist, tc01_pass, tc02_pass = self.test_distance_thresholds(
            model, valid_loader, data_dir, valid_df)

        noise_dist, tc03_pass = self.test_noise_robustness(model, sample_img_path)

        occl_dist, tc04_pass = self.test_occlusion_robustness(model, sample_img_path)

        tc05_pass = self.test_triplet_loss_threshold(final_loss, valid_loader, data_dir, valid_df)

        top5_accuracy, tc06_pass = self.test_top5_accuracy(model, valid_df, data_dir)

        self.plot_tsne_visualization(model, valid_df, data_dir)

        print("\n" + "="*60)
        print("COMPREHENSIVE TEST RESULTS SUMMARY")
        print("="*60)
        print(f"TC01 - Positive Distance    : {avg_pos_dist:.4f} < {self.config.POS_DIST_THRESHOLD}     : {'✓ PASS' if tc01_pass else '✗ FAIL'}")
        print(f"TC02 - Negative Distance    : {avg_neg_dist:.4f} > {self.config.NEG_DIST_THRESHOLD}     : {'✓ PASS' if tc02_pass else '✗ FAIL'}")
        print(f"TC03 - Noise Robustness     : {noise_dist:.4f} < {self.config.NOISE_DIST_THRESHOLD}     : {'✓ PASS' if tc03_pass else '✗ FAIL'}")
        print(f"TC04 - Occlusion Robustness : {occl_dist:.4f} < {self.config.OCCL_DIST_THRESHOLD}     : {'✓ PASS' if tc04_pass else '✗ FAIL'}")
        print(f"TC05 - Triplet Loss         : {final_loss:.4f} < {self.config.TRIPLET_LOSS_THRESHOLD}     : {'✓ PASS' if tc05_pass else '✗ FAIL'}")
        print(f"TC06 - Top-5 Accuracy       : {top5_accuracy:.4f} > {self.config.TOP5_ACCURACY_THRESHOLD}     : {'✓ PASS' if tc06_pass else '✗ FAIL'}")

        all_tests = [tc01_pass, tc02_pass, tc03_pass, tc04_pass, tc05_pass, tc06_pass]
        pass_count = sum(all_tests)
        print(f"\nOverall Test Results: {pass_count}/6 tests passed ({pass_count/6*100:.1f}%)")

        return {
            'tc01': {'value': avg_pos_dist, 'pass': tc01_pass, 'description': 'Positive Distance'},
            'tc02': {'value': avg_neg_dist, 'pass': tc02_pass, 'description': 'Negative Distance'},
            'tc03': {'value': noise_dist, 'pass': tc03_pass, 'description': 'Noise Robustness'},
            'tc04': {'value': occl_dist, 'pass': tc04_pass, 'description': 'Occlusion Robustness'},
            'tc05': {'value': final_loss, 'pass': tc05_pass, 'description': 'Triplet Loss'},
            'tc06': {'value': top5_accuracy, 'pass': tc06_pass, 'description': 'Top-5 Accuracy'},
            'overall': {'pass_count': pass_count, 'total': 6, 'pass_rate': pass_count/6}
        }

def main(data_dir=None, csv_path=None, config=None):
    """Main execution function"""
    if config is None:
        config = Config()
    print(f"Using device: {config.DEVICE}")

    try:
        DATA_DIR, CSV_PATH = setup_paths(data_dir, csv_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded dataset with {len(df)} samples")

    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train size: {len(train_df)}, Validation size: {len(valid_df)}")

    trainset = APN_Dataset(train_df, DATA_DIR, config.INPUT_SIZE, augment=True)
    validset = APN_Dataset(valid_df, DATA_DIR, config.INPUT_SIZE, augment=False)
    trainloader = DataLoader(trainset, batch_size=config.BATCH_SIZE, shuffle=True,
                           num_workers=config.NUM_WORKERS, collate_fn=custom_collate)
    validloader = DataLoader(validset, batch_size=config.BATCH_SIZE, shuffle=False,
                           num_workers=config.NUM_WORKERS, collate_fn=custom_collate)

    model = APN_Model(config.EMBEDDING_DIM).to(config.DEVICE)
    criterion = TripletLoss(margin=config.MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    trainer = PersonReIDTrainer(config)
    print("\nStarting training...")
    train_losses, valid_losses = trainer.train(
        model, trainloader, validloader, criterion, optimizer, config.EPOCHS)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()

    model.load_state_dict(torch.load('best_model.pt'))

    tester = PersonReIDTester(config)
    sample_img_path = os.path.join(DATA_DIR, df.iloc[0]['Anchor'])
    final_loss = valid_losses[-1] if valid_losses else 0.0
    test_results = tester.run_all_tests(model, validloader, DATA_DIR, valid_df, sample_img_path, final_loss)

    print("\nTraining and testing completed!")
    return model, test_results

if __name__ == "__main__":
    model, results = main()