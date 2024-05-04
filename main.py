from model.model import Model

model = Model("data.csv")
model.train()
print(model.theta)
print("From linalg")
model.displayResultWithLinalg()
#model.plotData()
