import os

classes = [0, 1]
labels = []
for sign in classes:
    a = os.listdir(os.path.join(os.getcwd(), "small_test_dataset", "Test", str(sign)))
    labels.extend([sign]*len(a))

print(labels)