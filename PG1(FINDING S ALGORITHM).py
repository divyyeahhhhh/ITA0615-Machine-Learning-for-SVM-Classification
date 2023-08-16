import csv

def find_s_algorithm(training_data):
    hypothesis = ['0'] * len(training_data[0][0])
    for example, label in training_data:
        if label == 'Y':
            for i, val in enumerate(example):
                if hypothesis[i] == '0':
                    hypothesis[i] = val
                elif hypothesis[i] != val:
                    hypothesis[i] = '?'
    return hypothesis

def read_csv_file(file_path):
    dataset = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dataset.append((row[:-1], row[-1]))
    return dataset

file_path = 'C:/Users/student/Desktop/dataset/enjoysport (1).csv'
training_data = read_csv_file(file_path)

hypothesis = find_s_algorithm(training_data)
print("Final hypothesis:", hypothesis)
