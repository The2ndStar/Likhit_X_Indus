import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 1. Load the Model
model = tf.keras.models.load_model('transistor_logic_v8.keras')

# 2. Setup Paths
test_base_path = r'C:\Desktop\Likhit_X_Indus\data\test'
# Get all subfolder names (good, bent_lead, cut_lead, etc.)
categories = [d for d in os.listdir(test_base_path) if os.path.isdir(os.path.join(test_base_path, d))]

results = {}

print("--- Starting Full Evaluation ---")

for cat in categories:
    folder_path = os.path.join(test_base_path, cat)
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        continue
        
    errors = []
    
    # Check all images in each folder
    for f in files:
        img_path = os.path.join(folder_path, f)
        
        # Process
        img = image.load_img(img_path, target_size=(224, 224))
        img_arr = image.img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_arr, axis=0)
        
        # Predict & Calculate MSE
        reconstructed = model.predict(img_batch, verbose=0)[0]
        mse = np.mean(np.square(img_arr - reconstructed))
        errors.append(mse)
    
    avg_error = np.mean(errors)
    results[cat] = avg_error
    print(f"Category: {cat:15} | Avg Error: {avg_error:.6f}")

# 3. Visualization
plt.figure(figsize=(10, 6))
names = list(results.keys())
values = list(results.values())

# Highlight the 'good' bar in green and others in red
colors = ['green' if x == 'good' else 'red' for x in names]

plt.bar(names, values, color=colors)
plt.title("Reconstruction Error by Category (Lower is Better for 'Good')")
plt.ylabel("Mean Squared Error (MSE)")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# 1. Load the Model
model = tf.keras.models.load_model('transistor_logic_v8.keras')
test_base_path = r'C:\Desktop\Likhit_X_Indus\data\test'

# Get all folders (good, bent_lead, etc.)
categories = [d for d in os.listdir(test_base_path) if os.path.isdir(os.path.join(test_base_path, d))]

# Create a grid: 2 rows (Original vs Reconstructed) x N categories
fig, axes = plt.subplots(2, len(categories), figsize=(20, 8))

for i, cat in enumerate(categories):
    folder_path = os.path.join(test_base_path, cat)
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        continue

    # 2. Process the first image in the folder
    img_path = os.path.join(folder_path, files[0])
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_arr, axis=0)
    
    # 3. AI Reconstruction
    reconstructed = model.predict(img_batch, verbose=0)[0]
    mse = np.mean(np.square(img_arr - reconstructed))
    
    # --- Row 1: Original Image ---
    axes[0, i].imshow(img_arr)
    axes[0, i].set_title(f"ORIGINAL: {cat.upper()}", fontsize=10, fontweight='bold')
    axes[0, i].axis('off')
    
    # --- Row 2: AI Reconstruction ---
    axes[1, i].imshow(reconstructed)
    # Color the text: Green for good, Red for defects
    text_color = 'green' if cat == 'good' else 'red'
    axes[1, i].set_title(f"RECONSTRUCTED\nError: {mse:.5f}", color=text_color, fontsize=10)
    axes[1, i].axis('off')

plt.tight_layout()
plt.suptitle("Transistor Reconstruction Analysis: Good vs. Defects", fontsize=16, y=1.05)
plt.show()