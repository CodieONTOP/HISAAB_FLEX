import pandas as pd
import numpy as np

np.random.seed(42)
n_customers = 400

data = {
    "CustomerID": [f"CUST-{1000+i}" for i in range(n_customers)],
    "Age": np.random.randint(18, 70, size=n_customers),
    "Annual_Income_k$": np.random.randint(15, 140, size=n_customers),
    "Spending_Score": np.random.randint(1, 100, size=n_customers),
    "Purchase_Frequency": np.random.randint(1, 50, size=n_customers),
    "Total_Spent": np.random.randint(100, 10000, size=n_customers)
}

df = pd.DataFrame(data)
df.to_csv("customer_data.csv", index=False)
print("customer_data.csv successfully created!")