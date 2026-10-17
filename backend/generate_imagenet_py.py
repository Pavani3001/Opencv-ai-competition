import json

with open("models/imagenet_labels.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

with open("imagenet_classes.py", "w", encoding="utf-8") as f:
    f.write('"""ImageNet-1000 class names for MobileNetV2 complementary classifier."""\n\n')
    f.write(f"IMAGENET_CLASSES = {repr(labels)}\n")

print(f"Generated imagenet_classes.py with {len(labels)} classes.")
