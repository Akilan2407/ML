import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
model = tf.keras.models.load_model("mnist_model.keras")
img = Image.open("4.png").convert("L")
img = img.resize((28, 28))
img_array = np.array(img)
img_array = 255 - img_array
img_array = img_array / 255.0
plt.imshow(img_array, cmap="gray")
plt.title("Input Digit")
plt.show()
img_array = img_array.reshape(1, 28, 28)
prediction = model.predict(img_array)
predicted_digit = np.argmax(prediction)
print("Predicted Digit:", predicted_digit)
print("\nPrediction probabilities:")
for digit, probability in enumerate(prediction[0]):
    print(f"{digit}: {probability * 100:.2f}%")