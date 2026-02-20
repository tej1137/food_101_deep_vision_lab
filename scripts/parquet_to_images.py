import pandas as pd
from PIL import Image
import io
from pathlib import Path

# Folder containing the parquet files
data_dir = Path(r"C:\Users\acer\OneDrive\Desktop\University\Year 3\Deep Learning\food101\data")

# Root folder where you want images saved
out_root = Path("images")  # or full path if you prefer
out_root.mkdir(exist_ok=True)

for parquet_file in data_dir.glob("*.parquet"):
    print("Processing", parquet_file)
    df = pd.read_parquet(parquet_file, engine="fastparquet")

    for _, row in df.iterrows():
        img_bytes = row["image.bytes"]        # bytes column
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        label = row["label"]                  # numeric class id
        fname = row["image.path"]             # e.g. '2885220.jpg'

        # Optional: create class subfolders like Food-101 layout
        out_dir = out_root / str(label)
        out_dir.mkdir(parents=True, exist_ok=True)

        img.save(out_dir / fname)
