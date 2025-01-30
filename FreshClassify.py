from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import DBSCAN
from PIL import Image
import numpy as np
import tempfile
import shutil
import torch
import config
import sys
import cv2
import os

class NoveltyDetector:
    def __init__(self, train_dir, tempo_test_dir, test_dir):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.train_dir = train_dir
        self.test_dir = test_dir

        self.batch_size = config.batch_size
        self.embedding_dim = config.embedding_dim
        self.learning_rate = config.learning_rate
        self.epochs = config.epochs

        self.transform = config.transform
        self.model = self._load_model()

        self.train_dataset = datasets.ImageFolder(train_dir, transform=self.transform, loader=self.custom_loader)
        self.test_dataset = self._get_image_folder_with_paths(tempo_test_dir)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        

        self.prototypes = {}
        self.novelty_threshold = None
        self.output_file_path = sys.argv[2]
        open(self.output_file_path, 'w').close()

    def custom_loader(self, path):
        img = Image.open(path)
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode == "RGBA":
            img = img.convert("RGB")
        return img

    def _load_model(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Get current directory
        entry_model_path = os.path.join(current_dir, "entry_model.pth")

        if os.path.exists(entry_model_path):
            model = torch.load(entry_model_path, weights_only=False)  # Load the entire model (architecture + weights)
            # print(f"Entry model loaded from: {entry_model_path}")
        else:
            raise FileNotFoundError(f"Entry model not found at {entry_model_path}. Please save it in the first code.")

        return model.to(self.device)  # Move the model to the correct device

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

    def detect_novelty(self, embedding, prototypes, threshold):
        # Normalize embedding for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        distances = {class_name: 1 - np.dot(embedding, prototype) for class_name, prototype in prototypes.items()}
        closest_class = min(distances, key=distances.get)
        closest_distance = distances[closest_class]
        # print(f"Closest class: {closest_class}, Distance: {closest_distance:.2f}")
        if closest_distance > threshold:
            return "Novel", closest_distance
        else:
            return closest_class, closest_distance

    def test_and_group_novel_classes(self, test_loader, prototypes, novelty_threshold):
        # print("Testing the model for novelty detection and grouping novel classes...")
        results = []
        embeddings = []
        labels = []
        novel_groups = []  # Store embeddings of novel samples
        novel_file_names = []  # Store filenames for novel samples
        files = []
        previous_novelgroups = []

        with open(self.output_file_path, 'a') as output_file:
            with torch.no_grad():
                for inputs, targets, paths in test_loader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs).cpu().numpy()
                    embeddings.extend(outputs)
                    for i, output in enumerate(outputs):
                        # Dynamic clustering if novel groups reach a threshold
                        if len(novel_groups) >= 3 and previous_novelgroups != novel_groups:
                            prototypes, novel_groups, updated_results, novel_file_names = self.dynamic_clustering_and_prototype(
                                novel_groups, prototypes, novel_file_names
                            )
                            if updated_results != []:
                                # print(f'updated results: {updated_results}')
                                self.update_output_file(self.output_file_path, updated_results)

                        # Detect novelty for the current output
                        class_name, distance = self.detect_novelty(output, prototypes, novelty_threshold)
                        true_label = "Novel" if targets[i] == -1 else self.train_dataset.classes[targets[i]]
                        file_name = os.path.basename(paths[i])
                        labels.append(class_name)
                        files.append(file_name)
                        results.append({
                            "File": file_name,
                            "Predicted": class_name,
                            "Distance": distance
                        })

                        # Write the class name to the output file in the same order as input
                        output_file.write(f"{file_name}, {class_name}\n")
                        output_file.flush()

                        # Print detailed information
                        # print(f"File: {file_name}, Predicted: {class_name}, Distance: {distance:.2f}")

                        previous_novelgroups = novel_groups.copy()

                        # Append novel embeddings and filenames to novel_groups
                        if class_name == "Novel":
                            # print(f"Novel sample detected with distance: {distance:.2f}")
                            novel_groups.append(output)
                            novel_file_names.append(file_name)
                            
          
                    if len(novel_groups) >= 3:
                        prototypes, novel_groups, updated_results, novel_file_names = self.dynamic_clustering_and_prototype(
                            novel_groups, prototypes, novel_file_names
                        )

                        # Update results and output file for grouped novelties
                        self.update_output_file(self.output_file_path, updated_results)
                        if updated_results != []:
                                # print(f'updated results: {updated_results}')
                                self.update_output_file(self.output_file_path, updated_results)

                    output_file.flush()

        return results, np.array(embeddings), labels, prototypes, files


    def dynamic_clustering_and_prototype(self, novel_groups, prototypes, novel_file_names, similarity_threshold=0.7, min_samples=3):
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

        # print(f"Cluster labels: {cluster_labels}")  # -1 indicates noise (outliers)

        # Group embeddings based on cluster labels
        grouped_novels = defaultdict(list)
        grouped_file_names = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            if label != -1:  # Ignore noise points
                grouped_novels[label].append(novel_groups[idx])
                grouped_file_names[label].append(novel_file_names[idx])
        # print(f'grouped file names {grouped_file_names}')


        updated_results = []
        # Assign names to valid clusters and construct prototypes
        for cluster_id, group_embeddings in grouped_novels.items():
            if len(group_embeddings) >= min_samples:
                # print(f"Novel cluster {cluster_id} detected with {len(group_embeddings)} samples.")
                # print(f"Please name the novel cluster {cluster_id}:")

                # Display images of the grouped novelties
                file_paths = grouped_file_names[cluster_id][:3]  # Display up to 3 images
                fig, axes = plt.subplots(1, len(file_paths), figsize=(15, 5))
                if len(file_paths) == 1:
                    axes = [axes]  # Ensure axes is iterable

                for ax, file_path in zip(axes, file_paths):
                    image = cv2.imread(f'{config.fruitCrash_level2_original}/{file_path}')
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    ax.imshow(image)
                    ax.axis("off")

                plt.show()
                new_class_name = input(f"Enter name and corresponding score for the novel class: ")

                # Compute the prototype for the new class
                new_prototype = np.mean(group_embeddings, axis=0)
                prototypes[new_class_name] = new_prototype / np.linalg.norm(new_prototype)  # Normalize the prototype
                print(f"New class '{new_class_name}' added with prototype.")

                # Update results and filenames for grouped samples
                for file_name in grouped_file_names[cluster_id]:
                    updated_results.append({"File": file_name, "Predicted": new_class_name})

        # Filter remaining ungrouped novelties
        ungrouped_indices = [idx for idx, label in enumerate(cluster_labels) if label == -1]
        remaining_novelties = novel_groups[ungrouped_indices]
        remaining_file_names = [novel_file_names[idx] for idx in ungrouped_indices]
        # print(f"Remaining ungrouped novelties: {len(remaining_novelties)}")
        return prototypes, remaining_novelties.tolist(), updated_results, remaining_file_names
    
    def update_output_file(self, output_file_path, updated_results):
        update_dict = {item['File']: item['Predicted'] for item in updated_results}
        
        # Read entire file into memory
        with open(output_file_path, 'r') as file:
            lines = file.readlines()

        # Update lines in memory
        for i, line in enumerate(lines):
            filename, _ = line.strip().split(', ')
            if filename in update_dict:
                lines[i] = f"{filename}, {update_dict[filename]}\n"

        # Write updated content back to the file
        with open(output_file_path, 'w') as file:
            file.writelines(lines)

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

        self.prototypes = np.load(config.prototypes, allow_pickle=True).item()
        train_embeddings = np.load(config.train_embeddings, allow_pickle=True).item()
        self.novelty_threshold = self.calculate_threshold(self.prototypes, train_embeddings)
        results, test_embeddings, test_labels, updated_prototypes, file_names = self.test_and_group_novel_classes(self.test_loader, self.prototypes, self.novelty_threshold)
        self.plot_embeddings(test_embeddings, test_labels, updated_prototypes, file_names, title="t-SNE Visualization")



if __name__ == "__main__":

    # Ensure the correct number of arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python3 noveltyDetection_fruitCrash.py <input_file> <output_file>")
        sys.exit(1)

    # Get input and output file paths from command-line arguments
    input_file = sys.argv[1]  # Path to 'selected_files.txt'
    output_file = sys.argv[2]  # Path to 'ann_output.txt'
    test_dir = sys.argv[3]

    # Ensure the input file exists
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        sys.exit(1)

    # Load the input file (list of selected file names)
    with open(input_file, 'r') as f:
        selected_files = [line.strip() for line in f.readlines()]

    print(f"Processing {len(selected_files)} files...")

    

    with tempfile.TemporaryDirectory() as temp_test_dir:
        # Create a dummy class folder inside the temporary directory
        class_folder = os.path.join(temp_test_dir, "unknown")
        os.makedirs(class_folder, exist_ok=True)

        skipped_files = 0
        for file_name in selected_files:
            source_path = os.path.join(test_dir, file_name)
            if not os.path.exists(source_path):
                print(f"File {file_name} not found in {config.fruitCrash_test_dir}. Skipping.")
                skipped_files += 1
                continue
            dest_path = os.path.join(class_folder, file_name)
            shutil.copy(source_path, dest_path)

        # Initialize the detector with the dynamic test directory
        detector = NoveltyDetector(config.train_dir, temp_test_dir, test_dir)
        print("Running the classification algorithm...")
        detector.run()

