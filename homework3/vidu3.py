import numpy as np
from pandas import *
# reading CSV file
data = read_csv("D:/_4_python/Machine-learning/homework3/Admission_Predict.csv")
# converting column data to list, then convert list to array
sn = data['Serial No.'].tolist()

gre = data['GRE Score'].tolist()
X1 = np.asarray(gre)

tfl = data['TOEFL Score'].tolist()
X2 = np.asarray(tfl)

unirt = data['University Rating'].tolist()
X3 = np.asarray(unirt)

sop = data['SOP'].tolist()
X4 = np.asarray(sop)

lor1 = data['LOR '].tolist()
X5 = np.asarray(lor1)

cgpa1 = data['CGPA'].tolist()
X6 = np.asarray(cgpa1)

research_exp = data['Research'].tolist()
X7 = np.asarray(research_exp)

prob_Admit = data['Chance of Admit'].tolist()
Yt = np.asarray(prob_Admit)
X1_tran=X1[range(0,350)]
X2_tran=X2[range(0,350)]
X3_tran=X3[range(0,350)]
X4_tran=X4[range(0,350)]
X5_tran=X5[range(0,350)]
X6_tran=X6[range(0,350)]
X7_tran=X7[range(0,350)]
X1_test=X1[range(351,399)]
X2_test=X2[range(351,399)]
X3_test=X3[range(351,399)]
X4_test=X4[range(351,399)]
X5_test=X5[range(351,399)]
X6_test=X6[range(351,399)]
X7_test=X7[range(351,399)]
Yt_tran = Yt[range(0,350)]
Yt_test = Yt[range(351,399)]

def results(Y):
    y = []
    for i in Y:
        if i >= 0.75 :
            y.append(1)
        else: y.append(0)
    return y
y = results(Yt_tran)
y_test = results(Yt_test)

X = np.array([X1_tran,X2_tran,X3_tran,X4_tran,X5_tran,X6_tran,X7_tran])
Xbar = np.concatenate((np.ones((1, X.shape[1])), X), axis = 0)
def sigmoid(s):
    return 1/(1 + np.exp(-s))

def logistic_sigmoid_regression(X, y, w_init, eta, tol = 1e-4, max_count = 10000):
# method to calculate model logistic regression by Stochastic Gradient Descent method
# eta: learning rate; tol: tolerance; max_count: maximum iterates
    w = [w_init]
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
# loop of stochastic gradient descent
    while count < max_count:
# shuffle the order of data (for stochastic gradient descent).
# and put into mix_id
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta*(yi - zi)*xi
            count += 1
# stopping criteria
            if count%check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)
    return w
eta = .05
d = Xbar.shape[0]
w_init = np.random.randn(d, 1)
w = logistic_sigmoid_regression(Xbar, y, w_init, eta)
print(w[-1])
X_test = np.array([X1_test,X2_test,X3_test,X4_test,X5_test,X6_test,X7_test])
Xbar_test = np.concatenate((np.ones((1, X_test.shape[1])), X_test), axis = 0)
h_teta= sigmoid(np.dot(w[-1].T, Xbar_test))
h_teta_result = results(h_teta[0])
print(h_teta_result)
compare = []
for i in range(0,len(h_teta_result)):
    if h_teta_result[i] == y_test[i]:
        compare.append(True)
    else : compare.append(False)
print(compare)

