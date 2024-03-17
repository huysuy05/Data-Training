import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

'''
Step 1: Clean Data
'''
df = pd.read_csv("ObesityDataSet.csv")
newdf = df.dropna()
male_filtered = newdf.query('Gender == "Male"') #Filtering data with certain values
male_filtered = male_filtered.sort_values(by="Age")
female_filtered = newdf.query('Gender == "Female"')
female_filtered = female_filtered.sort_values(by="Age")

'''
Mini Step 2: Summarize information from male_filtered and female_filtered
'''
male_summary = male_filtered.describe()
male_summary  = male_summary.iloc[1:]
female_summary = female_filtered.describe()
female_summary = female_summary.iloc[1:]
# print(list(male_summary))
ax = male_filtered.plot(kind="scatter", x="Weight", y="Height", color = "skyblue")
female_filtered.plot(kind="scatter", x="Weight", y="Height", color = "pink", ax=ax)
ax.set_xlabel("Weight Comparison for Male and Female")
ax.set_ylabel("Height comparison for Male and Female")
ax.text(0.9, 0.1, "Male", fontsize=12, color='skyblue', transform=ax.transAxes, ha='center')
ax.text(0.9, 0.05, "Female", fontsize=12, color='pink', transform=ax.transAxes, ha='center')
plt.show()


'''
Step 2: Visualizations
'''
#Male Analysis
plt.subplot(1, 2, 1)
plt.hist(male_filtered["NObeyesdad"], bins = 50, color="pink", alpha = 0.7)
plt.title("Male Obesity Level")
plt.xlabel("Body Type")
plt.ylabel("Frequency")

#Female Analysis
plt.subplot(1, 2, 2)
plt.hist(female_filtered["NObeyesdad"], bins = 50, color = "black", alpha = 0.7)
plt.title("Female Obesity Level")
plt.xlabel("Body Type")
plt.ylabel("Frequency")

plt.show(block=False)
plt.pause(1)
plt.close()

plt.tight_layout()
print("Conclusion: Both the majority of males and females in the dataset appear to be suffering from Obesity Level I \n")


'''
Step 3: Train the model
'''
# Create arrays of X and y
X = newdf[["Weight",
           "Age"
           ]].values
# print(X)
y = newdf[["family_history_with_overweight"]]
Le = LabelEncoder()
y = Le.fit_transform(y)


# Train according to the KNN model
X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.1)
svm_classifier = svm.SVC(kernel="linear")
svm_classifier.fit(X_train, y_train)
y_pred = svm_classifier.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy}")
for i in range(5):
    print(f"Actual class: {y_test[i]}, Predicted class: {y_pred[i]}")