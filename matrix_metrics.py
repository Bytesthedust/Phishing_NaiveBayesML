def confusion_matrix_metrics(tp, fp, tn, fn, bagging=False):
  """Calculates confusion matrix metrics
    Parameters:
    - tp (int): true positive value
    - fp (int): false positive value
    - tn (int): true negative value
    - fn (int): false negative value
    - bagging (bool): flag for whether matrix is from bagging
    Returns:
    (string): accuracy
    (string): precision
    (string): recall
    (string): f1
    (string): spec """
  accuracy = (tp+tn)/(tp+fp+tn+fn)*100
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1 = (2*precision * recall)/(precision+recall)
  spec = tn/(fp+tn)

  if bagging == False:
    print("Accuracy: ", accuracy)
  print("Precision: ", precision)
  print("Recall: ", recall)
  print("F1-score: ", f1)
  print("Specificity: ", spec)
  print("FPR: ", 1-spec)
