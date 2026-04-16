##SCRIPT TO DOWNLOAD THE DATA & VIEW IF THEY ARE DOWNLOADED

# import tensorflow_datasets as tfds

# ds, info = tfds.load(
#     'food101',
#     data_dir='/workspace/food101-dl-project/dataset',
#     with_info=True,
#     as_supervised=True  # returns (image, label) tuples directly
# )

# print(info)

ds, info = tfds.load(
    'food101',
    data_dir='/workspace/food101-dl-project/dataset',
    with_info=True,
    as_supervised=True,
    download=False
)

print(f"Classes: {info.features['label'].num_classes}")
print(f"Train: {info.splits['train'].num_examples}")
print(f"Val: {info.splits['validation'].num_examples}")