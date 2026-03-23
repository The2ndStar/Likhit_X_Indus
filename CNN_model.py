import tensorflow as tf
from tensorflow.keras import layers, models, optimizers ,regularizers

def create_transistor_encoder(input_shape=(224, 224, 3)):
    model = models.Sequential([
        # --- ENCODER (Shrinks 224 -> 28) ---
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)), # 112x112
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)), # 56x56
        
        # layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        # layers.BatchNormalization(),
        # layers.MaxPooling2D((2, 2)), # 28x28

        layers.Flatten(),
        # THE TUNED BOTTLENECK (For PCA)
        layers.Dense(16, activation='relu', name='feature_layer', activity_regularizer=regularizers.l1(1e-4)), 
        layers.Dropout(0.3),

        # --- THE BRIDGE ---
        # Must match the 28x28x128 shape from the last Encoder layer
        layers.Dense(28 * 28 * 128, activation='relu'),

        # --- DECODER (Grows 28 -> 224) ---
        layers.Reshape((28, 28, 128)),
        
        layers.UpSampling2D((2, 2)), # 56x56
        layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
        
        layers.UpSampling2D((2, 2)), # 112x112
        layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
        
        layers.UpSampling2D((2, 2)), # 224x224 (Matches Input!)
        layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')
    ])
    
    # Using MAE to help distinguish those tiny defect differences
    my_opt = optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=my_opt, loss='mae') 
    return model