def confusion_matrix_metrics(tp, fp, tn, fn, bagging=False):
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
