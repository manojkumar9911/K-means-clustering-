from sklearn.metrics import precision_score, recall_score

# Assuming you have true cluster assignments for your documents
true_labels = [   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  ]

# Assuming you have predicted cluster assignments (replace with your actual predicted labels)
predicted_labels = [ 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1 ]

try:
    # Calculate precision and recall
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')

    # Print precision and recall
    print("Precision:", precision)
    print("Recall:", recall)

except ValueError as e:
    print(f"An error occurred: {e}")