import pickle 
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 


with open("./ASL.pickle", "rb") as f:
    dataset = pickle.load(f)

count = 0
# Iterate through each item in the dataset's "dataset" list
for i in dataset["dataset"]:
    count += 1
    # Check if the length of the current item is not equal to 42 (expected length for hand landmarks)
    if len(i) != 42:
        print(len(i))
        # Find the index of the current item in the dataset
        index = dataset["dataset"].index(i)
        
        # Remove the item from both the "dataset" and "labels" lists at the found index
        dataset["dataset"].pop(index)
        dataset["labels"].pop(index)

        print(len(i))

data = np.asarray(dataset["dataset"])
labels = np.asarray(dataset["labels"])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = accuracy_score(y_pred, y_test)

with open("./ASL_model.p", "wb") as f:
    pickle.dump({"model":model}, f)