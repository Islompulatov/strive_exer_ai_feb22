from train import Normal
from train import Models
import data_handler as dh

x_train, x_test, y_train, y_test = dh.get_data("Chapter 02/07. Gradient Boosting and Encoding/insurance.csv")

test_model = Models(x_train, x_test, y_train, y_test)

ada = test_model.gradient_boost()
print(type(ada))