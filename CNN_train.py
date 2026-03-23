import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from CNN_model import create_transistor_encoder

# 1. Data Loading (213 Good images)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    'data/train', # Folder with 213 'good' images
    target_size=(224, 224), # Resolution: Higher = more detail, but slower
    batch_size=8, # How many images to look at before updating.
    class_mode='input',
    subset='training'
)



# 2. Initialize and Train (ปรับได้เด้อ)
model = create_transistor_encoder()
model.fit(train_gen, epochs=150, verbose=1)

# 3. Save for the team
model.save('transistor_logic_v8.keras')
print("CNN Phase Complete. Feature Extractor is ready for PCA.")