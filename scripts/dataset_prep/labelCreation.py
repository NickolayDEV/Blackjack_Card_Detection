import os
from pathlib import Path

# Path to the original dataset
base_dir = Path("../../datasets/unsortedDataset") 
output_dir = Path("../../datasets/yolo_dataset") 

# Get the mapping "class name" → class_id
def get_class_map(data_split):
    class_dirs = sorted((base_dir / data_split).glob("*"))
    return {cls.name: i for i, cls in enumerate(class_dirs)}

# Generate annotations and copy images
def generate_labels(data_split):
    class_map = get_class_map(data_split)
    print(f"[{data_split.upper()}] Classes:", class_map)
    print(f'[{class_map.keys()}]')
    label_dir = output_dir / "labels" / data_split
    image_dir = output_dir / "images" / data_split
    label_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    counter = 0  # global counter for new file names

    for class_name, class_id in class_map.items():
        img_dir = base_dir / data_split / class_name

        for img_path in img_dir.glob("*.jpg"):
            new_stem = f"{data_split}_{counter}"
            new_img_path = image_dir / f"{new_stem}.jpg"
            new_txt_path = label_dir / f"{new_stem}.txt"

            # Copy the image
            new_img_path.write_bytes(img_path.read_bytes())

            # Create the annotation
            with open(new_txt_path, "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

            counter += 1

# Run for all splits
for split in ["train", "val", "test"]:
    generate_labels(split)
