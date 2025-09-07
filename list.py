import os

# Root path (change if needed)
# root = "/Users/mayursatre/Downloads/Calwest/Images"
root = "/Users/mayursatre/Downloads/CapitalCoreRecycling/Images"

# Image extensions to count
image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

# Dictionary to store counts
folder_counts = {}

for dirpath, dirnames, filenames in os.walk(root):
    count = sum(1 for f in filenames if os.path.splitext(f)[1].lower() in image_exts)
    folder_counts[dirpath] = count

# Print results
for folder, count in sorted(folder_counts.items(), key=lambda x: x[0]):
    print(f"{folder}: {count} images")
