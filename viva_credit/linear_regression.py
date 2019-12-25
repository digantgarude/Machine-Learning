import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("./viva_credit.csv")
# print(len(data.index))

X_train = data.iloc[0:2499,1:-1]
X_test = data.iloc[25000:,1:-1]
# print(X.head())

Y_train = data.iloc[0:2499,-1]
Y_test = data.iloc[25000:,-1]
# print(X_test.head())
# print(Y_test.head())

reg = LinearRegression()

reg.fit(X_train,Y_train)

Y_pred = reg.predict(X_test)
print(Y_pred[:10])
Y_pred_new = np.array([1 if y > 0.5 else 0 for y in Y_pred])

print("Coefficients : ",reg.coef_)

# Mean squared error
print("Mean squared error : ",mean_squared_error(Y_test,Y_pred_new))

print("Coefficient of deteermination : ",r2_score(Y_test,Y_pred_new))


print(Y_test.shape)
print(Y_pred_new.shape)
# Verify in csv
verify1 = pd.DataFrame(Y_test)
verify1.to_csv('verify1.csv',index=False)
verify2 = pd.DataFrame(Y_pred_new)
verify2.to_csv('verify2.csv',index=False)
# Plot outputs

# plt.scatter(X_test,Y_test,color="black")
# plt.plot(X_test,Y_pred, color='blue',linewidth=3)
# plt.xticks()
# plt.yticks()
# plt.show()
