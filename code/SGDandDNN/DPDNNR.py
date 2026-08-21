import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
import warnings

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]

data = pd.read_csv(url, names=columns, na_values="?", skipinitialspace=True)
print(f": {data.shape}")


data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
print(f"data.shape: {data.shape}")


data["income"] = (data["income"] == ">50K").astype(int)

education_map = {
    "10th": "dropout", "11th": "dropout", "12th": "dropout",
    "1st-4th": "dropout", "5th-6th": "dropout", "7th-8th": "dropout",
    "9th": "dropout", "Preschool": "dropout",
    "HS-grad": "high_school", "Bachelors": "bachelors",
    "Masters": "masters", "Doctorate": "doctorate",
    "Prof-school": "professional", "Assoc-acdm": "associate",
    "Assoc-voc": "associate", "Some-college": "some_college"
}
data["education_level"] = data["education"].map(education_map)


data["net_capital"] = data["capital-gain"] - data["capital-loss"]


data["hours-per-week_cat"] = pd.cut(
    data["hours-per-week"], bins=[0, 30, 40, 60, 100],
    labels=["part_time", "full_time", "over_time", "extreme_time"]
).astype(str)


data["is_native_us"] = (data["native-country"] == "United-States").astype(int)


data.drop(["fnlwgt", "education", "native-country"], axis=1, inplace=True)


continuous_features = [
    "age", "education-num", "capital-gain", "capital-loss",
    "hours-per-week", "net_capital"
]


categorical_features = [col for col in data.columns if col not in continuous_features + ["income"]]
data_encoded = pd.get_dummies(data, columns=categorical_features, prefix_sep='=', drop_first=True)


scaler = MinMaxScaler()
data_scaled = data_encoded.copy()
data_scaled[continuous_features] = scaler.fit_transform(data_encoded[continuous_features])
X_temp = data_scaled.drop("income", axis=1)
y_temp = data_scaled["income"]

rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf.fit(X_temp, y_temp)

importance_df = pd.DataFrame({
    'feature': X_temp.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)


weight_dict = importance_df.set_index('feature')['importance'].to_dict()


feature_weights = {}
for col in NUM_FEATURES:
    if col in weight_dict:
        feature_weights[col] = weight_dict[col]
    else:
        feature_weights[col] = np.mean(list(weight_dict.values()))


data_noisy = data_scaled.copy()

# 全局隐私预算
EPSILON_GLOBAL = 1.0
SENSITIVITY = 1.0
data_noisy = data_scaled.copy()
noise_log = {}

weight_age = feature_weights['age']
scale_age = SENSITIVITY / (EPSILON_GLOBAL * weight_age)
noise_age = np.random.laplace(loc=0.0, scale=scale_age, size=len(data_scaled))
data_noisy['age'] = data_scaled['age'] + noise_age


weight_edu = feature_weights['education-num']
scale_edu = SENSITIVITY / (EPSILON_GLOBAL * weight_edu)
noise_edu = np.random.laplace(loc=0.0, scale=scale_edu, size=len(data_scaled))
data_noisy['education-num'] = data_scaled['education-num'] + noise_edu


weight_gain = feature_weights['capital-gain']
scale_gain = SENSITIVITY / (EPSILON_GLOBAL * weight_gain)
noise_gain = np.random.laplace(loc=0.0, scale=scale_gain, size=len(data_scaled))
data_noisy['capital-gain'] = data_scaled['capital-gain'] + noise_gain


weight_loss = feature_weights['capital-loss']
scale_loss = SENSITIVITY / (EPSILON_GLOBAL * weight_loss)
noise_loss = np.random.laplace(loc=0.0, scale=scale_loss, size=len(data_scaled))
data_noisy['capital-loss'] = data_scaled['capital-loss'] + noise_loss


weight_hours = feature_weights['hours-per-week']
scale_hours = SENSITIVITY / (EPSILON_GLOBAL * weight_hours)
noise_hours = np.random.laplace(loc=0.0, scale=scale_hours, size=len(data_scaled))
data_noisy['hours-per-week'] = data_scaled['hours-per-week'] + noise_hours


weight_net = feature_weights['net_capital']
scale_net = SENSITIVITY / (EPSILON_GLOBAL * weight_net)
noise_net = np.random.laplace(loc=0.0, scale=scale_net, size=len(data_scaled))
data_noisy['net_capital'] = data_scaled['net_capital'] + noise_net


X = data_noisy.drop("income", axis=1).astype('float32').values
y = data_noisy["income"].astype('float32').values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

model = Sequential([
    Input(shape=(X_train.shape[1],), name="InputLayer"),
    Dense(512, activation='relu', name="Dense1"),
    BatchNormalization(name="BN1"),
    Dropout(0.4, name="Dropout1"),

    Dense(256, activation='relu', name="Dense2"),
    BatchNormalization(name="BN2"),
    Dropout(0.3, name="Dropout2"),

    Dense(128, activation='relu', name="Dense3"),
    BatchNormalization(name="BN3"),
    Dropout(0.3, name="Dropout3"),

    Dense(64, activation='relu', name="Dense4"),
    BatchNormalization(name="BN4"),
    Dropout(0.2, name="Dropout4"),

    Dense(32, activation='relu', name="Dense5"),
    Dropout(0.2, name="Dropout5"),

    Dense(1, activation='sigmoid', name="Output")
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

#plot_model(model, to_file='dp_dnn_model.png', show_shapes=True, show_layer_names=True)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1)
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=1024,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)


y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

print(f"accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"precision_score:     {precision_score(y_test, y_pred):.4f}")
print(f"recall:     {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:   {f1_score(y_test, y_pred):.4f}")
print(f"AUC Score:  {roc_auc_score(y_test, y_pred_prob):.4f}")

print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_prob):.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('FPR');
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend();
plt.grid(True)
plt.show()


model.save('private_income_classifier.keras')
