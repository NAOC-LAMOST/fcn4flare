import evaluate

metric = evaluate.load("precision")
print(metric.compute(predictions=[1, 1, 0, 1], references=[1, 0, 0, 1]))    