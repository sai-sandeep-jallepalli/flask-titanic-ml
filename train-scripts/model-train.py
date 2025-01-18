import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
data = pd.read_csv("datasets/raw_dataset.csv")

# Drop unnecessary columns
data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')

# Drop rows with missing values
data = data.dropna()

# Handle categorical features
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Save label encoders
encoders_path = "pickle-models/label_encoders.pkl"
with open(encoders_path, 'wb') as file:
    pickle.dump(label_encoders, file)

# Split features and target
X = data.drop(columns=['Survived'])
y = data['Survived']

# Save feature column names
column_names_path = "pickle-models/column_names.pkl"
with open(column_names_path, 'wb') as file:
    pickle.dump(X.columns.tolist(), file)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
model_path = "pickle-models/train.pkl"
with open(model_path, 'wb') as file:
    pickle.dump(model, file)

print("Model training completed and saved.")
