from sklearn.metrics import accuracy_score, f1_score

# Assuming you have true cluster assignments for your documents
true_labels = [ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ,1 , 1, 1, 1, 1,1,1,1,1,1,1 , 1, 1, 1, 1,1,1,1,1,1 ]

# Assuming you have predicted cluster assignments (replace with your actual predicted labels)
predicted_labels =   [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0 ]
try:
    # Calculate accuracy and F1-score
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    # Print accuracy and F1-score
    #print("Accuracy:", accuracy)
    print("F1-score:", f1)

except ValueError as e:
    print(f"An error occurred: {e}")