import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import plot_model
import warnings

warnings.filterwarnings('ignore')

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
RANDOM_STATE = 42
TEST_SIZE = 0.3
EPOCHS = 200
BATCH_SIZE = 1024
VERBOSE = 1


np.random.seed(RANDOM_STATE)
import tensorflow as tf

tf.random.set_seed(RANDOM_STATE)



def load_and_clean_data(url, columns):
    data = pd.read_csv(url, names=columns, na_values="?", skipinitialspace=True)
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    print(f"data.shape: {data.shape}")

    return data


data = load_and_clean_data(URL, COLUMNS)


def feature_engineering(df):
    df = df.copy()

    df["income"] = (df["income"] == ">50K").astype(int)

    education_map = {
        "10th": "dropout", "11th": "dropout", "12th": "dropout",
        "1st-4th": "dropout", "5th-6th": "dropout", "7th-8th": "dropout",
        "9th": "dropout", "Preschool": "dropout",
        "HS-grad": "high_school",
        "Bachelors": "bachelors",
        "Masters": "masters",
        "Doctorate": "doctorate",
        "Prof-school": "professional",
        "Assoc-acdm": "associate", "Assoc-voc": "associate",
        "Some-college": "some_college"
    }
    df["education_level"] = df["education"].map(education_map)

    df["net_capital"] = df["capital-gain"] - df["capital-loss"]

    df["hours-per-week_cat"] = pd.cut(
        df["hours-per-week"],
        bins=[0, 30, 40, 60, 100],
        labels=["part_time", "full_time", "over_time", "extreme_time"]
    ).astype(str)

    df["is_native_us"] = (df["native-country"] == "United-States").astype(int)

    df.drop(["education", "native-country", "fnlwgt"], axis=1, inplace=True)  # fnlwgt 是样本权重，通常不用

    continuous_features = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week", "net_capital"]
    categorical_features = [col for col in df.columns if col not in continuous_features + ["income"]]

    df = pd.get_dummies(df, columns=categorical_features, drop_first=True, prefix_sep='=')

    scaler = MinMaxScaler()
    df[continuous_features] = scaler.fit_transform(df[continuous_features])

    return df, continuous_features, scaler


data_processed, cont_features, scaler = feature_engineering(data)
print(f"data_processed: {data_processed.shape}")

X = data_processed.drop("income", axis=1).astype('float32').values
y = data_processed["income"].astype('float32').values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

#print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

def create_advanced_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(512, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.4),

        Dense(256, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.2),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    return model


model = create_advanced_model(X_train.shape[1])

plot_model(model, to_file='dnn_model.png', show_shapes=True, show_layer_names=True)


callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        mode='min',
        verbose=VERBOSE,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=VERBOSE
    ),
    ModelCheckpoint(
        filepath='best_model.keras',
        save_best_only=True,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        verbose=VERBOSE
    )
]


history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=VERBOSE
)


y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1 Score:          {f1:.4f}")
print(f"AUC Score:         {auc:.4f}")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()



def plot_training_history(history):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    ax[0, 0].plot(history.history['loss'], label='Train Loss')
    ax[0, 0].plot(history.history['val_loss'], label='Val Loss')
    ax[0, 0].set_title('Loss Over Epochs')
    ax[0, 0].legend()
    ax[0, 0].grid(True)

    # Accuracy
    ax[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
    ax[0, 1].set_title('Accuracy Over Epochs')
    ax[0, 1].legend()
    ax[0, 1].grid(True)

    # Precision
    ax[1, 0].plot(history.history['precision'], label='Train Precision')
    ax[1, 0].plot(history.history['val_precision'], label='Val Precision')
    ax[1, 0].set_title('Precision Over Epochs')
    ax[1, 0].legend()
    ax[1, 0].grid(True)

    # Recall
    ax[1, 1].plot(history.history['recall'], label='Train Recall')
    ax[1, 1].plot(history.history['val_recall'], label='Val Recall')
    ax[1, 1].set_title('Recall Over Epochs')
    ax[1, 1].legend()
    ax[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


plot_training_history(history)


model.save('final_adult_income_model.keras')



loaded_model = load_model('final_adult_income_model.keras')