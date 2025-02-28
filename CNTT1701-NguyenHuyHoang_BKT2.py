import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Đọc file Excel chứa dữ liệu không hoàn chỉnh
df_loaded = pd.read_excel("CNTT 17-01_Nguyễn Huy Hoàng304_BKT2.xlsx")

# In ra nội dung của file
print("Nội dung file dữ liệu không hoàn chỉnh:")
print(df_loaded)

# Phát hiện các giá trị lỗi (NaN)
missing_values = df_loaded.isnull().sum()

# Xử lý dữ liệu bị thiếu:
# - Với cột "Danh mục chi tiêu", thay thế NaN bằng "Không xác định"
df_loaded['Danh mục chi tiêu'] = df_loaded['Danh mục chi tiêu'].fillna("Không xác định")

# - Với cột "Số tiền (VNĐ)", thay thế NaN bằng giá trị trung bình của cột
df_loaded['Số tiền (VNĐ)'] = pd.to_numeric(df_loaded['Số tiền (VNĐ)'], errors='coerce')
average_amount = df_loaded['Số tiền (VNĐ)'].mean()
df_loaded['Số tiền (VNĐ)'] = df_loaded['Số tiền (VNĐ)'].fillna(average_amount)


# Nhóm dữ liệu theo "Danh mục chi tiêu" và tính tổng số tiền
category_expense = df_loaded.groupby("Danh mục chi tiêu")["Số tiền (VNĐ)"].sum()

# Vẽ biểu đồ cột
plt.figure(figsize=(10, 5))
category_expense.plot(kind="bar", color="skyblue", edgecolor="black")

# Thêm tiêu đề và nhãn
plt.title("Tổng số tiền chi tiêu theo danh mục", fontsize=14)
plt.xlabel("Danh mục chi tiêu", fontsize=12)
plt.ylabel("Số tiền (VNĐ)", fontsize=12)
plt.xticks(rotation=45, ha="right")  # Xoay nhãn trục x cho dễ đọc
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Hiển thị biểu đồ
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Tạo dữ liệu giả định về danh mục chi tiêu và số tiền
data = {
    "Danh mục chi tiêu": ["Ăn uống", "Giải trí", "Mua sắm", "Đi lại", "Học tập"] * 4,
    "Số tiền (VNĐ)": np.random.randint(50000, 2000000, 20)  # Tạo số tiền ngẫu nhiên
}

# Chuyển dữ liệu thành DataFrame
df = pd.DataFrame(data)

# Nhóm dữ liệu theo danh mục và tính tổng số tiền
df_grouped = df.groupby("Danh mục chi tiêu").sum().reset_index()

# Tách biến độc lập (X) và phụ thuộc (y)
X = np.arange(len(df_grouped)).reshape(-1, 1)  # Chuyển danh mục thành chỉ số số học
y = df_grouped["Số tiền (VNĐ)"].values

# Xây dựng mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X, y)

# Dự đoán giá trị
y_pred = model.predict(X)

# Tính toán các hệ số hồi quy
beta_0 = model.intercept_  # Hệ số chặn
beta_1 = model.coef_[0]    # Hệ số dốc

# Đánh giá mô hình
r2 = r2_score(y, y_pred)  # Hệ số xác định R^2
mse = mean_squared_error(y, y_pred)  # Sai số bình phương trung bình (MSE)

# Hiển thị kết quả
print(f"Phương trình hồi quy: y = {beta_0:.2f} + {beta_1:.2f}x")
print(f"Hệ số chặn (beta_0): {beta_0}")
print(f"Hệ số dốc (beta_1): {beta_1}")
print(f"Hệ số xác định (R^2): {r2}")
print(f"Sai số bình phương trung bình (MSE): {mse}")

# Vẽ biểu đồ
plt.scatter(X, y, color="blue", label="Dữ liệu thực tế")
plt.plot(X, y_pred, color="red", label="Dự đoán (hồi quy)")
plt.xticks(ticks=np.arange(len(df_grouped)), labels=df_grouped["Danh mục chi tiêu"], rotation=45)
plt.title("Hồi quy tuyến tính: Danh mục chi tiêu và Số tiền")
plt.xlabel("Danh mục chi tiêu")
plt.ylabel("Số tiền (VNĐ)")
plt.legend()
plt.grid(True)
plt.show()