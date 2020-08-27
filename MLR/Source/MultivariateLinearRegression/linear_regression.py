import csv
import sys
import numpy as np


def readData(data):
    X = []
    y = []
    with open(data) as f:
        r = csv.reader(f)
        # Skip header
        next(r)
        # Đọc X, y
        for line in r:
            xline = [1.0]
            # Đọc từ 2 trở về sau bỏ cột đầu là thứ tự và cột tiếp theo là y (GPA)
            for s in line[2:]:
                xline.append(float(s))
            X.append(xline)
            # Đọc y vào là cột
            y.append(float(line[1]))
    return (X, y)

def main():
    if len(sys.argv) != 2:
        raise ValueError('Error reading arguments.')
    X0, y0 = readData(sys.argv[1])
    # Chuyển đổi tất cả trong data sang mảng numpy,
    # Ngoại trừ 10 dòng cuối cùng để test
    d = len(X0) - 10

    X = np.array(X0[:d])
    y1 = np.array([y0[:d]])

    y = np.transpose(np.array([y0[:d]]))

    # Giải Beta
    Xt = np.transpose(X)
    XtX = np.dot(Xt, X)  # X.T*X
    Xty = np.dot(Xt, y)  # X.T*y
    beta = np.linalg.solve(XtX, Xty)  ## Solve (X.T*X)Beta = X.T*y)
    print("B:")
    print(beta)

    # Thử đưa dự đoán cho 10 hàng cuối trong data
    print("Test:")
    for data, actual in zip(X0[d:], y0[d:]):
        x = np.array([data])
        prediction = np.dot(x, beta)
        print('prediction = ' + str(prediction[0, 0]) + ' actual = ' + str(actual))
    k = 9  # số cột của data
    n = d  # số hàng
    mean = np.sum(y1) / n  # tính mean
    e = []  # rss sai số (chưa được giải thích bởi mô hình)
    yy = []  # ess (dao động được giải thích bởi mô hình)
    tss = [] # Tổng bình phương
    # Tính các tổng các giao động
    for data, actual in zip(X0[:n], y0[:n]):
        x = np.array([data])
        prediction = np.dot(x, beta)
        e.append((actual - (prediction[0][0])) ** 2)
        yy.append((prediction[0][0] - mean) ** 2)
        tss.append((actual - mean)**2)
    ESS = np.sum(yy)
    RSS = np.sum(e)
    TSS = np.sum(tss)
    R_Square = 1 - RSS / TSS
    Adjusted_R_Square = 1 - (n - 1) / (n - k) * (1 - R_Square)
    print("R_Square: ", R_Square)
    print("Adjusted R_Square: ", Adjusted_R_Square)

if __name__ == '__main__':
    main()