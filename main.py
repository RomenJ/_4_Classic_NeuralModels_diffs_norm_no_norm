import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Load dataset
games = pd.read_csv('games_season_enriched.csv')

print("Shape", games.shape)
print(games.head(10))
print("Columnas o variables del dataset")
print(games.columns)

# Standardize the data
scaler_seed_diff = StandardScaler()
scaler_score_diff = StandardScaler()
scaler_team_1 = StandardScaler()
scaler_team_2 = StandardScaler()

games['seed_diff_scaled'] = scaler_seed_diff.fit_transform(games[['seed_diff']])
games['score_diff_scaled'] = scaler_score_diff.fit_transform(games[['score_diff']])
games['team_1_scaled'] = scaler_team_1.fit_transform(games[['team_1']])
games['team_2_scaled'] = scaler_team_2.fit_transform(games[['team_2']])

# Function to create and compile a complex neural network model
def create_complex_model():
    input_tensor = Input(shape=(1,))
    
    # Primera capa oculta con 128 neuronas, función de activación ReLU y normalización por lotes
    hidden = Dense(128, activation='relu')(input_tensor)
    hidden = BatchNormalization()(hidden)
    hidden = Dropout(0.3)(hidden)
    
    # Segunda capa oculta con 64 neuronas, función de activación ReLU y normalización por lotes
    hidden = Dense(64, activation='relu')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Dropout(0.3)(hidden)
    
    # Tercera capa oculta con 32 neuronas, función de activación ReLU y normalización por lotes
    hidden = Dense(32, activation='relu')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Dropout(0.3)(hidden)
    
    # Capa de salida
    output_tensor = Dense(1)(hidden)
    
    model = Model(input_tensor, output_tensor)
    
    # Compilar el modelo con el optimizador Adam y la pérdida de error absoluto medio
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')
    
    return model

def create_simple_model():
    input_tensor = Input(shape=(1,))
    output_tensor = Dense(1)(input_tensor)
    model = Model(input_tensor, output_tensor)
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

# Function to train the model and plot the results
def train_and_plot(model, x, y, epochs, batch_size, validation_split=0.20, title_suffix=""):
    history = model.fit(x, y, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_split=validation_split, 
                        verbose=True)
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'One Input-One Output Network Model {epochs} Epochs {title_suffix}')
    plt.savefig(f'One_Input_One_Output_Network_Model_{epochs}_Epochs_{title_suffix}.jpg')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
  

# Function to train the model and plot the results
def train_and_plot3(model, x, y, epochs, batch_size, validation_split=0.20, title_suffix=""):
    history = model.fit(x, y, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_split=validation_split, 
                        verbose=True)
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Complex : 1 Input-Dense(128)-Dense(64)-Dense(32)-1 Output)  {epochs} Epochs {title_suffix}')
    plt.savefig(f'Model_Complex_1_Input_Dense(128)_Dense(64)_Dense(32)_1_Output_{epochs}_Epochs_{title_suffix}.jpg')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Function to create and compile a neural network model with two inputs
def create_two_input_model():
    input_tensor_1 = Input(shape=(1,), name='team_1')
    input_tensor_2 = Input(shape=(1,), name='team_2')
    concat_input = Concatenate()([input_tensor_1, input_tensor_2])
    output_tensor = Dense(1)(concat_input)
    model = Model(inputs=[input_tensor_1, input_tensor_2], outputs=output_tensor)
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

# Train simple models with different epochs
set_seeds()
epochs_list = [6, 10, 50]
for epochs in epochs_list:
    # Train with original data
    model = create_simple_model()
    print(f"Training simple model for {epochs} epochs (Original Data)")
    train_and_plot(model, games['seed_diff'], games['score_diff'], epochs, batch_size=64, title_suffix="Original")

    # Train with standardized data
    model = create_simple_model()
    print(f"Training simple model for {epochs} epochs (Standardized Data)")
    train_and_plot(model, games['seed_diff_scaled'], games['score_diff_scaled'], epochs, batch_size=64, title_suffix="Standardized")

# Train two-input models with different epochs
set_seeds()
epochs_list = [4, 10, 50, 100]
for epochs in epochs_list:
    # Train with original data
    model = create_two_input_model()
    print(f"Training two-input model for {epochs} epochs (Original Data)")
    history = model.fit([games['team_1'], games['team_2']], 
                        games['score_diff'], 
                        epochs=epochs, 
                        batch_size=2048, 
                        validation_split=0.10, 
                        verbose=True)
    
    evaluation = model.evaluate([games['team_1'], games['team_2']], games['score_diff'], verbose=True)
    print(f"Model evaluation result (Original Data): {evaluation}")
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Two Input Concat Neural Network Model {epochs} Epochs (Original Data)')
    plt.savefig(f'Two_Input_Concat_Neural_Network_Model_{epochs}_Epochs_Original.jpg')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Train with standardized data
    model = create_two_input_model()
    print(f"Training two-input model for {epochs} epochs (Standardized Data)")
    history = model.fit([games['team_1_scaled'], games['team_2_scaled']], 
                        games['score_diff_scaled'], 
                        epochs=epochs, 
                        batch_size=2048, 
                        validation_split=0.10, 
                        verbose=True)
    
    evaluation = model.evaluate([games['team_1_scaled'], games['team_2_scaled']], games['score_diff_scaled'], verbose=True)
    print(f"Model evaluation result (Standardized Data): {evaluation}")
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Two Input Concat Neural Network Model {epochs} Epochs (Standardized Data)')
    plt.savefig(f'Two_Input_Concat_Neural_Network_Model_{epochs}_Epochs_Standardized.jpg')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Train complex models with different epochs
set_seeds()
epochs_list_complex = [4, 10, 50, 100]
for epochs in epochs_list_complex:
    # Train with original data
    model = create_complex_model()
    
    print(f'Model Complex : 1 Input-Dense(128)-Dense(64)-Dense(32)-1 Output)  {epochs} Epochs' )
    train_and_plot3(model, games['seed_diff'], games['score_diff'], epochs, batch_size=64, title_suffix="Original")

    # Train with standardized data
    model = create_complex_model()
    
    print(f'Model Complex : 1 Input-Dense(128)-Dense(64)-Dense(32)-1 Output)  {epochs} Epochs ')
    train_and_plot3(model, games['seed_diff_scaled'], games['score_diff_scaled'], epochs, batch_size=64, title_suffix="Standardized")

