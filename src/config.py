#paths
DATA_DIR="./data"
ZIPFILE_PATH="./data/archive (6).zip"
TRAINING_DATA="./data/casting_data/casting_data/train"
TESTING_DATA="./data/casting_data/casting_data/test"
PLOTS_PATH="./results/plots"
MODEL_PATH="./trained models/inspection.keras"
MODEL_ARCHITECTURE="./results/model_architecture.json"
MODEL_HISTORY_PATH="./results/model_history.json"

#model parameters
INPUT_SHAPE=(128,128,3)
EPOCHS=10
OPTIMIZER="adam"
LOSS="sparse_categorical_crossentropy"
METRICS=["accuracy"]
ACTIVATION="sigmoid"
BATCH_SIZE=60