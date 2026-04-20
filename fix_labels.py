import os

labels_dir = r"E:\PBL\smart_grocery_list\datasets\grocery\labels"

for root, _, files in os.walk(labels_dir):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            with open(file_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    parts[0] = str(int(float(parts[0])))  # Convert "0.0" → "0"
                    new_lines.append(" ".join(parts))

            with open(file_path, "w") as f:
                f.write("\n".join(new_lines))
print("✅ All label files fixed.")
