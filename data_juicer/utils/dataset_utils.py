import os
import io
import zipfile
from tqdm import tqdm

def format_huggingface_image_dataset(
    dataset, image_key, label_key, label_mapping, filename, save_dir
):
    """Convert a Huggingface dataset to Cleanlab Studio format.

    dataset: datasets.Dataset
        HuggingFace image dataset
    image_key: str
        column name for image in dataset
    label_key: str
        column name for label in dataset
    label_mapping: Dict[str, int]
        id to label str mapping
        If labels are already strings, set label_mapping to None
    filename: str
        filename for the zip file
    save_dir: str
        directory to save the zip file

    """

    def image_data_generator():
        """Generator to yield image data and its path in the zip file."""
        for idx, data in enumerate(dataset):
            image = data[image_key]
            label = data[label_key]
            class_dir = label_mapping[label] if label_mapping else label

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_data = buf.getvalue()

            yield f"hf_dataset/{class_dir}/image_{idx}.png", image_data

    zip_path = os.path.join(save_dir, f"{filename}.zip")

    with zipfile.ZipFile(zip_path, "w") as zf:
        for path, data in tqdm(image_data_generator(), total=len(dataset)):
            zf.writestr(path, data)

    print(f"Saved zip file to: {zip_path}")
    

if __name__ == '__main':
    from datasets import load_dataset, concatenate_datasets

    cifar10_hf = load_dataset("/mnt/c/Users/Administrator/Desktop/cifar10")["train"]
    # print(cifar10_dict)
    # cifar10_hf = concatenate_datasets(cifar10_dict.values())

    # construct mapping from id to label str
    label_str_list = cifar10_hf.features["label"].names
    label_mapping = {i: name for i, name in enumerate(label_str_list)}
    print(label_mapping)


    format_huggingface_image_dataset(
        dataset=cifar10_hf,
        image_key="img",
        label_key="label",
        label_mapping=label_mapping,
        filename="cifar10_hf",
        save_dir="./",
    )
