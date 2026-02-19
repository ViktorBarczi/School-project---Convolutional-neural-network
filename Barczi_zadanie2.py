import tensorflow as tf
import keras
import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow.keras.preprocessing.image as keras_image
from tensorflow.keras.models import Sequential, model_from_json, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import regularizers

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import visualkeras
import cv2

ImageDataGenerator = keras_image.ImageDataGenerator
load_img = keras_image.load_img
img_to_array = keras_image.img_to_array


print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)


# # =========================================================
# # DATASET – categorical + shuffle=False
# # =========================================================

img_size = (64, 64)
batch_size = 32

EXPECTED_CLASSES = ["dog", "elephant", "sheep", "horse"]

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'zad2/dataset/training_set',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_set = test_datagen.flow_from_directory(
    'zad2/dataset/test_set',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)



num_classes = training_set.num_classes
print("Počet tried:", num_classes)
print("Indexy tried:", training_set.class_indices)

assert num_classes == 4, f"Očakávam 4 triedy, našiel som {num_classes}"

found_classes = list(training_set.class_indices.keys())
print("Nájdené triedy:", found_classes)

for c in EXPECTED_CLASSES:
    assert c in found_classes, f"Chýba trieda: {c}"

index_to_class = {v: k for k, v in training_set.class_indices.items()}
print("index_to_class:", index_to_class)


# =========================================================
# CNN – softmax + categorical_crossentropy
# =========================================================


cnn = Sequential()

cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu',input_shape=(64, 64, 3)))
cnn.add(MaxPooling2D(pool_size=2, strides=2))

cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=2, strides=2))

cnn.add(Flatten())
cnn.add(Dense(units=128, activation='relu'))

cnn.add(Dense(units=num_classes, activation='softmax'))

cnn.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn.summary()



# =========================================================
# TRÉNING – 25 epoch
# =========================================================

hist = cnn.fit(
    x=training_set,
    validation_data=test_set,
    epochs=25
)


# =========================================================
# FIX pre visualkeras – dopočítanie output_shape
# =========================================================

cnn.build((None, 64, 64, 3))

input_shape = cnn.input_shape
for layer in cnn.layers:
    if not hasattr(layer, "output_shape") or layer.output_shape is None:
        try:
            out_shape = layer.compute_output_shape(input_shape)
            layer.output_shape = out_shape
            input_shape = out_shape
        except Exception as e:
            print(f"Nemôžem vypočítať output_shape pre vrstvu {layer.name}: {e}")


# =========================================================
# Vizualizácia architektúry
# =========================================================

try:
    visualkeras.layered_view(
        cnn,
        to_file='cnn_architecture.png',
        legend=True
    )
    print("Architecture saved to cnn_architecture.png")
except Exception as e:
    print("Visualkeras warning:", e)


# =========================================================
# Graf accuracy
# =========================================================

plt.figure()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Model_accuracy.jpg', dpi=500)
plt.show()


# =========================================================
# Graf loss
# =========================================================

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('Model_loss.jpg', dpi=500)
plt.show()


# =========================================================
# PREDIKCIE NA TESTOVACEJ SADA
# =========================================================

print(test_set.class_indices)

predictions = cnn.predict(test_set)
print(predictions)

predictions = np.argmax(predictions, axis=1)
print(predictions)



# =========================================================
# PRESNOSŤ + CONFUSION MATRIX + REPORT
# =========================================================

results_validation = cnn.evaluate(test_set, batch_size=batch_size)
print("test loss, test acc:", results_validation)

acc = accuracy_score(test_set.classes, predictions)
print("Accuracy_score:", acc)

cm = confusion_matrix(test_set.classes, predictions)
print("Confusion matrix:\n", cm)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=list(index_to_class.values()),
            yticklabels=list(index_to_class.values()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("Classification report:")
print(classification_report(
    test_set.classes,
    predictions,
    target_names=list(index_to_class.values())
))

# =========================================================
# FUNKCIA NA PREDIKCIU
# =========================================================

def predict_image(model, img_name, image_path, img_size, index_to_class,
                  save_dir="zad2/dataset/predict"):

    file_name = os.path.basename(image_path)
    print(file_name)

    # načítanie a príprava obrázka pre model
    img = load_img(image_path, target_size=img_size)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0

    # predikcia
    result = model.predict(arr)
    print(result)
    print(training_set.class_indices)

    pred_idx = int(np.argmax(result, axis=1)[0])
    pred_class_name = index_to_class[pred_idx]
    print(f"{pred_class_name}")

    # uloženie obrázka s TEXTOM
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{img_name}.jpg")

    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print(f"CHYBA: cv2.imread() nevedelo načítať '{image_path}'")
    else:
        text = f"{pred_class_name}"
        cv2.putText(
            img_cv,
            text,
            (10, 35),                      # pozícia textu
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),                   # zelený text
            2,
            cv2.LINE_AA
        )

        ok = cv2.imwrite(save_path, img_cv)
        if not ok:
            print(f"CHYBA: nepodarilo sa uložiť '{save_path}'")
        else:
            print(save_path)

    return pred_class_name


# =========================================================
# TEST 1 OBRÁZKA Z KAŽDEJ TRIEDY
# =========================================================

single_images = {
    "dog":      "zad2/dataset/single_prediction/dog.png",
    "elephant": "zad2/dataset/single_prediction/elephant.png",
    "sheep":    "zad2/dataset/single_prediction/sheep.jpg",
    "horse":    "zad2/dataset/single_prediction/horse.jpg"
}

for class_name, path in single_images.items():
    pred = predict_image(
        model=cnn,
        img_name=class_name,
        image_path=path,
        img_size=img_size,
        index_to_class=index_to_class
    )

    print(f"Skutočná trieda: {class_name}, predikovaná trieda: {pred}")



# =========================================================
# ULOŽENIE MODELU
# =========================================================

model_json = cnn.to_json()
with open('cnn.json', 'w') as json_file:
    json_file.write(model_json)

save_model(cnn, 'weights.hdf5')
print("Model uložený.")


# =========================================================
# NAČÍTANIE MODELU
# =========================================================

with open('cnn.json', 'r') as json_file:
    json_saved_model = json_file.read()

network_loaded = model_from_json(json_saved_model)
network_loaded.load_weights('weights.hdf5')

network_loaded.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

network_loaded.summary()
print("Model úspešne načítaný.")


# single_images = {
#     "dog":      "zad2/dataset/single_prediction/dog.png",
#     "elephant": "zad2/dataset/single_prediction/elephant.png",
#     "sheep":    "zad2/dataset/single_prediction/sheep.jpg",
#     "horse":    "zad2/dataset/single_prediction/horse.jpg"
# }

# for class_name, path in single_images.items():
#     pred = predict_image(
#         model=network_loaded,
#         img_name=class_name,
#         image_path=path,
#         img_size=img_size,
#         index_to_class=index_to_class
#     )

#     print(f"Skutočná trieda: {class_name}, predikovaná trieda: {pred}")