import os
from collections import Counter

train_dir = '../../../output/train'
class_counts = {cls: len(os.listdir(os.path.join(train_dir, cls))) for cls in os.listdir(train_dir)}
print(class_counts)
