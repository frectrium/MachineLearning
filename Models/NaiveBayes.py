import pandas as pd
import functions_for_naive_Bayes

# to import the file and load the data
# Load data
df = pd.read_csv('pima-indians-diabetes.csv') #if the csv file is in same directory as this python file, otherwise add relative path
data = df.values.tolist()

# Encode classes and convert attributes to float
data = functions_for_naive_Bayes.encode_class(data) # don't complain about the class name being this big, I didn't know how to include functions from different file before this so I just named it this big, too lazy to change it now 🤪
for i in range(len(data)):
    for j in range(len(data[i]) - 1):
        data[i][j] = float(data[i][j])


# splitting the data into training and test

ratio = 0.8
train_data, test_data = functions_for_naive_Bayes.splitting(data, ratio)

# Train the model
info = functions_for_naive_Bayes.MeanAndStdDevForClass(train_data)

# Test the model
predictions = functions_for_naive_Bayes.getPredictions(info, test_data)
accuracy = functions_for_naive_Bayes.accuracy_rate(test_data, predictions)
print('Accuracy of the model:', accuracy)




