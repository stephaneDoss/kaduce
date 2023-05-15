from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Charger le modèle
model = load_model('Models\model_cv.h5')

# Page d'accueil

@app.route('/')
def home():
    return render_template('index.html')

# Page de prédiction


@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer l'image téléchargée à partir du formulaire HTML
    file = request.files['file']
    file.save('fichier.jpeg')  # Enregistre le fichier sur le disque

    # Prétraiter l'image
    img = load_img('fichier.jpeg', target_size=(100, 100))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img/255

    # Faire une prédiction
    prediction = model.predict(img)
    class_labels = ['NORMAL', 'PNEUMONIA']
    predicted_class = class_labels[np.argmax(prediction)]

    # Renvoyer la prédiction
    return jsonify({'prediction': predicted_class})


if __name__ == '__main__':
    app.run(debug=True)
