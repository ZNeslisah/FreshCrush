import os
from torchvision import transforms
# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))

train_dir = os.path.join(current_dir, "dataset", "train")
val_dir = os.path.join(current_dir, "dataset", "val")
test_dir = os.path.join(current_dir, "dataset", "test")


fruitCrash_level1 = os.path.join(current_dir, "game_dataset", "level1")
fruitCrash_level2 = os.path.join(current_dir, "game_dataset", "level2")

fruitCrash_level1_original = os.path.join(current_dir, "game_dataset", "level1_original")
fruitCrash_level2_original = os.path.join(current_dir, "game_dataset", "level2_original")


# fruitCrash_level2 = os.path.join(current_dir, "game_dataset", "level_zorla")
# fruitCrash_level2_original = os.path.join(current_dir, "game_dataset", "level_zorla_original")


train_embeddings = f'{current_dir}/train_embeddings.npy'
prototypes = f'{current_dir}/prototypes.npy'

# Hyperparameters
batch_size = 128
embedding_dim = 512
learning_rate = 1e-4
epochs = 3


# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])