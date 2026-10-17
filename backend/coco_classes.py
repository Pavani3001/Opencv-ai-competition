"""
COCO dataset class names used by YOLOv8 pretrained model.

The YOLOv8n model is trained on the COCO dataset and can detect exactly
these 80 object categories. Objects outside this list CANNOT be detected.

License note: YOLOv8 weights are AGPL-3.0 licensed.
This is compatible with open-source competition submissions.
"""

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Grouped categories for scene summary generation
CATEGORY_GROUPS = {
    "People": ["person"],
    "Animals": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                "bear", "zebra", "giraffe"],
    "Vehicles": ["bicycle", "car", "motorcycle", "airplane", "bus", "train",
                 "truck", "boat"],
    "Furniture": ["chair", "couch", "potted plant", "bed", "dining table",
                  "toilet", "bench"],
    "Electronics": ["tv", "laptop", "mouse", "remote", "keyboard",
                    "cell phone"],
    "Kitchen & Food": ["bottle", "wine glass", "cup", "fork", "knife",
                       "spoon", "bowl", "banana", "apple", "sandwich",
                       "orange", "broccoli", "carrot", "hot dog", "pizza",
                       "donut", "cake", "microwave", "oven", "toaster",
                       "sink", "refrigerator"],
    "Sports & Recreation": ["frisbee", "skis", "snowboard", "sports ball",
                            "kite", "baseball bat", "baseball glove",
                            "skateboard", "surfboard", "tennis racket"],
    "Accessories": ["backpack", "umbrella", "handbag", "tie", "suitcase"],
    "Outdoor": ["traffic light", "fire hydrant", "stop sign", "parking meter"],
    "Other Items": ["book", "clock", "vase", "scissors", "teddy bear",
                    "hair drier", "toothbrush"],
}
