import os
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import torchvision
from torchvision.models import ResNet18_Weights
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import DBSCAN
import config
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

class NoveltyDetector:
    def __init__(self, train_dir, val_dir, test_dir):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = config.batch_size
        self.embedding_dim = config.embedding_dim
        self.learning_rate = config.learning_rate
        self.epochs = config.epochs
        self.transform = config.transform

        self.model = self._load_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.train_dataset = datasets.ImageFolder(train_dir, transform=self.transform, loader=self.custom_loader)
        self.val_dataset = self._get_image_folder_with_paths(val_dir)
        self.test_dataset = self._get_image_folder_with_paths(test_dir)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        self.prototypes = {}
        self.novelty_threshold = None

    def custom_loader(self, path):
        img = Image.open(path)
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode == "RGBA":
            img = img.convert("RGB")
        return img

    def _load_model(self):
        model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        return model.to(self.device)

    def _get_image_folder_with_paths(self, directory):
        class ImageFolderWithPaths(datasets.ImageFolder):
            def __getitem__(self, index):
                original_tuple = super().__getitem__(index)
                path = self.imgs[index][0]
                return original_tuple + (path,)
        return ImageFolderWithPaths(directory, transform=self.transform, loader=self.custom_loader)

    def compute_embeddings(self, loader):
        embeddings = {}
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs).cpu().numpy()
                # Normalize embeddings for cosine similarity
                outputs = outputs / np.linalg.norm(outputs, axis=1, keepdims=True)
                for i, target in enumerate(targets):
                    class_name = self.train_dataset.classes[target]
                    if class_name not in embeddings:
                        embeddings[class_name] = []
                    embeddings[class_name].append(outputs[i])
        for key in embeddings:
            embeddings[key] = np.vstack(embeddings[key])
        return embeddings

    def compute_prototypes(self, embeddings):
        # Compute normalized prototypes for cosine similarity
        prototypes = {class_name: np.mean(class_embeddings, axis=0) for class_name, class_embeddings in embeddings.items()}
        return {key: value / np.linalg.norm(value) for key, value in prototypes.items()}

    def calculate_threshold(self, prototypes, embeddings):
        distances = []
        for class_name, prototype in prototypes.items():
            class_embeddings = embeddings[class_name]
            # Compute cosine similarity
            similarities = cosine_similarity(class_embeddings, prototype.reshape(1, -1))
            # Convert cosine similarity to cosine distance (1 - similarity)
            distances.extend(1 - similarities.flatten())
        return np.mean(distances) + 2 * np.std(distances)

    def refine_with_validation(self):
        print("Refining the model using validation data...")
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            for inputs, targets, _ in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                distances = torch.stack([
                    torch.norm(outputs - torch.tensor(self.prototypes[class_name]).to(self.device), dim=1)
                    for class_name in self.train_dataset.classes
                ], dim=1)
                loss = self.criterion(-distances, targets)
                total_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                predicted = distances.argmin(dim=1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss:.4f}, Accuracy: {correct / total:.4f}")
        self.model.eval()
        print("Model refinement with validation data completed.")

    def detect_novelty(self, embedding, prototypes, threshold):
        # Normalize embedding for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        distances = {class_name: 1 - np.dot(embedding, prototype) for class_name, prototype in prototypes.items()}
        closest_class = min(distances, key=distances.get)
        closest_distance = distances[closest_class]
        print(f"Closest class: {closest_class}, Distance: {closest_distance:.2f}")
        if closest_distance > threshold:
            return "Novel", closest_distance
        else:
            return closest_class, closest_distance

    # Test the model for novelty detection and group novel classes
    def test_and_group_novel_classes(self,test_loader, prototypes, novelty_threshold):
        print("Testing the model for novelty detection and grouping novel classes...")
        results = []
        embeddings = []
        labels = []
        novel_groups = []  # Store embeddings of novel samples
        files = []
        previous_novelgroups = []

        with torch.no_grad():
            for inputs, targets, paths in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs).cpu().numpy()
                embeddings.extend(outputs)

                
                for i, output in enumerate(outputs):
                    if len(novel_groups) >= 3 and previous_novelgroups != novel_groups:
                        prototypes, novel_groups = self.dynamic_clustering_and_prototype(novel_groups, prototypes)
                    class_name, distance = self.detect_novelty(output, prototypes, novelty_threshold)
                    true_label = "Novel" if targets[i] == -1 else self.train_dataset.classes[targets[i]]
                    file_name = os.path.basename(paths[i])
                    labels.append(class_name)
                    files.append(file_name)
                    results.append({
                        "File": file_name,
                        "True Label": true_label,
                        "Predicted": class_name,
                        "Distance": distance
                    })
                    print(f"File: {file_name}, True Label: {true_label}, Predicted: {class_name}, Distance: {distance:.2f}")
                    previous_novelgroups = novel_groups.copy()
                    if class_name == "Novel":
                        print(f"Novel sample detected with distance: {distance:.2f}")
                        novel_groups.append(output)
                        print(f"Novel groups: {len(novel_groups)}")
                        
                if len(novel_groups) >= 3:
                        # prototypes_afternovel = combineNovality(novel_groups, grouped_prototypes, prototypes)
                        prototypes, novel_groups =self.dynamic_clustering_and_prototype(novel_groups, prototypes)
            
            return results, np.array(embeddings), labels, prototypes, files



    def dynamic_clustering_and_prototype(self, novel_groups, prototypes, similarity_threshold=0.7, min_samples=3):
        """
        Dynamically clusters novel samples using cosine similarity and constructs prototypes for valid clusters.

        Args:
            novel_groups (list): List of embeddings for novel samples.
            prototypes (dict): Existing prototypes dictionary.
            similarity_threshold (float): Minimum cosine similarity to consider points in the same cluster.
            min_samples (int): Minimum number of samples required to form a cluster.

        Returns:
            dict: Updated prototypes with new novel classes added.
            list: Updated list of ungrouped novelties.
        """

        # Normalize novel groups for cosine similarity
        novel_groups = np.array(novel_groups)
        novel_groups = novel_groups / np.linalg.norm(novel_groups, axis=1, keepdims=True)

        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(novel_groups)

        # Convert similarity to distance (1 - similarity)
        distance_matrix = 1 - similarity_matrix

        # Ensure no negative values in the distance matrix (clamp to 0)
        distance_matrix = np.maximum(distance_matrix, 0)

        # Apply DBSCAN with the cosine distance
        clustering = DBSCAN(eps=1 - similarity_threshold, min_samples=min_samples, metric="precomputed").fit(distance_matrix)
        cluster_labels = clustering.labels_

        print(f"Cluster labels: {cluster_labels}")  # -1 indicates noise (outliers)

        # Group embeddings based on cluster labels
        grouped_novels = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            if label != -1:  # Ignore noise points
                grouped_novels[label].append(novel_groups[idx])

        # Assign names to valid clusters and construct prototypes
        for cluster_id, group_embeddings in grouped_novels.items():
            if len(group_embeddings) >= min_samples:
                print(f"Novel cluster {cluster_id} detected with {len(group_embeddings)} samples.")
                print(f"Please name the novel cluster {cluster_id}:")
                new_class_name = input(f"Enter name for the novel class (e.g., 'new_class'): ")

                # Compute the prototype for the new class
                new_prototype = np.mean(group_embeddings, axis=0)
                prototypes[new_class_name] = new_prototype / np.linalg.norm(new_prototype)  # Normalize the prototype
                print(f"New class '{new_class_name}' added with prototype.")

        # Filter remaining ungrouped novelties
        ungrouped_indices = [idx for idx, label in enumerate(cluster_labels) if label == -1]
        remaining_novelties = novel_groups[ungrouped_indices]

        print(f"Remaining ungrouped novelties: {len(remaining_novelties)}")
        return prototypes, remaining_novelties.tolist()
   
        


    def plot_embeddings(self, embeddings, labels, prototypes, filenames, title="Embedding Visualization"):
        num_samples = len(embeddings)
        perplexity = min(30, num_samples - 1)

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        for key in prototypes:
            prototypes[key] = prototypes[key] / np.linalg.norm(prototypes[key])

        all_embeddings = np.vstack([embeddings, np.array(list(prototypes.values()))])
        reduced_embeddings = tsne.fit_transform(all_embeddings)

        plt.figure(figsize=(12, 10))
        unique_labels = set(labels)
        for label in unique_labels:
            indices = [i for i, lbl in enumerate(labels) if lbl == label]
            plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=label, alpha=0.7)
            for i in indices:
                plt.annotate(filenames[i], (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8, alpha=0.6)

        prototype_labels = list(prototypes.keys())
        prototype_indices = range(len(embeddings), len(all_embeddings))
        plt.scatter(reduced_embeddings[prototype_indices, 0], reduced_embeddings[prototype_indices, 1], c="red",
                    marker="X", label="Prototypes", s=100)
        for i, label in enumerate(prototype_labels):
            plt.annotate(label, (reduced_embeddings[prototype_indices[i], 0], reduced_embeddings[prototype_indices[i], 1]),
                         fontsize=10, color="black")

        plt.title(title)
        plt.legend()
        plt.show()

    def run(self):
        print("Computing embeddings for training data...")
        train_embeddings = self.compute_embeddings(self.train_loader)
        self.prototypes = self.compute_prototypes(train_embeddings)

        self.refine_with_validation()
        print("Recomputing prototypes after refinement...")
        train_embeddings = self.compute_embeddings(self.train_loader)
        self.prototypes = self.compute_prototypes(train_embeddings)

        np.save(f"{current_dir}/train_embeddings.npy", train_embeddings)
        np.save(f"{current_dir}/prototypes.npy", self.prototypes)

        original = train_embeddings['apple']
        loaded = np.load(config.train_embeddings, allow_pickle=True).item()['apple']
        print(f"Max difference in prototype: {np.max(np.abs(original - loaded))}")


        self.novelty_threshold = self.calculate_threshold(self.prototypes, train_embeddings)
        print(f"Dynamic novelty threshold: {self.novelty_threshold}")

        results, test_embeddings, test_labels, updated_prototypes, file_names = self.test_and_group_novel_classes(self.test_loader, self.prototypes, self.novelty_threshold)
        self.plot_embeddings(test_embeddings, test_labels, updated_prototypes, file_names, title="t-SNE Visualization")
        

        entry_model_path = os.path.join(current_dir, "entry2_model.pth")
        torch.save(self.model, entry_model_path)  # Save the entire model (architecture + weights)
        print(f"Entry model saved to: {entry_model_path}")


# Example usage
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    detector = NoveltyDetector(config.train_dir, config.val_dir, config.test_dir)
    detector.run()


