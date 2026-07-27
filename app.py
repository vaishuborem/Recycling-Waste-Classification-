import torch
import clip  # CLIP model for image-text similarity
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the CLIP model and its tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Extended list of common recyclable materials
waste_categories = [
    # Paper and Cardboard
    "Newspaper", "Cardboard boxes", "Paperboard", "Magazines", "Junk mail",
    "Cartons", "Paper packaging", "Brown paper bags",
    
    # Plastics
    "Plastic bottles", "Plastic jars", "Plastic containers", "Grocery bags",
    "Plastic packaging", "Styrofoam",

    # Glass
    "Glass bottles", "Glass jars", "Glass containers",

    # Metals
    "Aluminum cans", "Steel cans", "Tin cans", "Aluminum foil", 
    "Steel and tin foil packaging",

    # Organic Materials
    "Food waste", "Yard trimmings", "Wood waste", "Garden waste",

    # Electronics
    "Computers", "Mobile phones", "TVs", "Printers", "Keyboards", "Gaming consoles",

    # Batteries
    "Alkaline batteries", "Nickel-cadmium batteries", "Nickel-metal hydride batteries", "Lead-acid batteries",

    # Other Materials
    "Textiles", "Shoes", "Housewares", "Furniture", "Appliances", "Hazardous waste"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file and file.filename != '':
        img = Image.open(file)
        img_preprocessed = preprocess(img).unsqueeze(0).to(device)

        # Encode the text descriptions for each waste category
        text_inputs = clip.tokenize(waste_categories).to(device)

        # Forward pass through the CLIP model
        with torch.no_grad():
            image_features = model.encode_image(img_preprocessed)
            text_features = model.encode_text(text_inputs)

        # Compute similarity between the image and each text category
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        best_category_idx = similarity.argmax().item()

        # Get the label of the most similar category
        label = waste_categories[best_category_idx]

        return jsonify({'label': label})
    
    return jsonify({'error': 'Invalid image'}), 400

if __name__ == '__main__':
    app.run(debug=True)