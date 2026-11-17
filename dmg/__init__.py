import os
# this pages for count images


# show image size
def img_dataset():
    train_size = count_classes()
    test_size = count_test_set()
    percentage(train_size, test_size)

# count train images
def count_classes():
    train_dir = "gtsrb/Train"
    total_train = 0

    print("Class\tCount")
    print("-" * 20)
    for class_id in sorted(os.listdir(train_dir), key=int):

        class_path = os.path.join(train_dir, class_id)

        if os.path.isdir(class_path):

            count = len([
                f for f in os.listdir(class_path)
                if f.endswith((".png", ".jpg", ".jpeg"))
            ])

            total_train += count

            print(f"{class_id}\t{count}")

    print("-" * 20)
    print(f"Total Train Images: {total_train}")
    return total_train

# count test images
def count_test_set():
    import pandas as pd
    test_csv = "gtsrb/Test.csv"
    df = pd.read_csv(test_csv)

    print("Total Test Images:", len(df))
    return len(df)

# Calculate the percentage of the training set and test set in the entire dataset.
def percentage(train_num, test_num):
    total = train_num + test_num

    train_percent = train_num / total * 100
    test_percent = test_num / total * 100

    print(f"Train: {train_percent:.2f}%")
    print(f"Test : {test_percent:.2f}%")