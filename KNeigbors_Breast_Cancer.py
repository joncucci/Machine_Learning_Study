import codecademylib3_seaborn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

breast_cancer_data = load_breast_cancer()
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100) 

k_list = range(1, 101)
accuracies = []
for k in k_list:
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  accuracies.append(classifier.score(validation_data, validation_labels))

plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.plot(k_list, accuracies)
plt.show()

#print(len(training_data), len(training_labels))
#print(breast_cancer_data.feature_names)
#print(breast_cancer_data.data[0])
#print(breast_cancer_data.target)
#print(breast_cancer_data.target_names)
