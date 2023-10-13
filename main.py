import csv
import sys
import calendar
import math

TEST_SIZE = 0.4

def main() -> object:
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    model = train_model(X_train, y_train)
    predictions = predict(X_train, y_train, X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def load_data(filename):
    months = {month: index-1 for index, month in enumerate(calendar.month_abbr) if index}
    months['June'] = months.pop('Jun')

    evidence = []
    labels = []

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            evidence.append([
                int(row['Administrative']),
                float(row['Administrative_Duration']),
                int(row['Informational']),
                float(row['Informational_Duration']),
                int(row['ProductRelated']),
                float(row['ProductRelated_Duration']),
                float(row['BounceRates']),
                float(row['ExitRates']),
                float(row['PageValues']),
                float(row['SpecialDay']),
                months[row['Month']],
                int(row['OperatingSystems']),
                int(row['Browser']),
                int(row['Region']),
                int(row['TrafficType']),
                1 if row['VisitorType'] == 'Returning_Visitor' else 0,
                1 if row['Weekend'] == 'TRUE' else 0
            ])
            labels.append(1 if row['Revenue'] == 'TRUE' else 0)

    return (evidence, labels)

def train_model(evidence, labels):
    return evidence, labels

def predict(X_train, y_train, X_test):
    predictions = []

    for test_instance in X_test:
        nearest_neighbor = find_nearest_neighbor(test_instance, X_train, y_train)
        predictions.append(nearest_neighbor[1])

    return predictions

def find_nearest_neighbor(test_instance, X_train, y_train):
    min_distance = float('inf')
    nearest_neighbor = None

    for i, train_instance in enumerate(X_train):
        distance = calculate_distance(test_instance, train_instance)
        if distance < min_distance:
            min_distance = distance
            nearest_neighbor = (y_train[i], min_distance)

    return nearest_neighbor

def calculate_distance(instance1, instance2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(instance1, instance2)))

def evaluate(labels, predictions):
    sensitivity = float(0)
    specificity = float(0)

    total_positive = float(0)
    total_negative = float(0)

    for label, prediction in zip(labels, predictions):
        if label == 1:
            total_positive += 1
            if label == prediction:
                sensitivity += 1
        if label == 0:
            total_negative += 1
            if label == prediction:
                specificity += 1

    sensitivity /= total_positive
    specificity /= total_negative

    return sensitivity, specificity

if __name__ == "__main__":
    main()
