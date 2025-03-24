# House Price Prediction

## 📌 Overview
This project applies **Linear Regression** to predict house prices based on various features. The dataset used is `Housing.csv`, which contains real estate data.

## 🛠️ Technologies Used
- **Python**
- **Pandas** (for data processing)
- **NumPy** (for numerical computations)
- **Matplotlib & Seaborn** (for data visualization)
- **Scikit-Learn** (for machine learning model)

## 📂 Project Structure
```
House-Price-Prediction/
│── data/
│   ├── Housing.csv  # Dataset
├── house_price_prediction.ipynb  # Jupyter Notebook with full analysis
│── README.md  # Project documentation
│── requirements.txt  # Dependencies
```

## 🚀 Installation & Usage
### 1️⃣ Clone the repository
```bash
git clone https://github.com/touya2604/House-Price-Prediction.git
cd House-Price-Prediction
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Jupyter Notebook
```bash
jupyter notebook notebooks/house_price_prediction.ipynb
```


## 📊 Data Preprocessing
The dataset undergoes several preprocessing steps:
- Handling missing values
- Encoding categorical features
- Feature scaling using **StandardScaler**
- Splitting into training and testing sets (80% Train, 20% Test)

## 🔍 Model Training
The **Linear Regression** model is trained using **Scikit-Learn**:
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

## 📈 Model Evaluation
After training, the model is evaluated using metrics such as:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

## 📌 Results
- The trained model provides an R² score of **0.97** (replace with actual score).
- The predictions are visualized using a scatter plot comparing actual vs. predicted prices.

## 🤝 Contributing
Feel free to fork this repository, submit issues, or make pull requests to improve the project!

## 📜 License
This project is open-source and available under the **MIT License**.

---
💡 **Author:** Touya Nguyen  
📧 Contact: nguyenanh9761@gmail.com

