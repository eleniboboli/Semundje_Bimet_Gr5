import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


def single_prediction(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess image
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_idx = predicted.item()

    class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]
    
    
    
    # Safety check
    if predicted_idx >= len(class_names):
        print(f"ERROR: Predicted index {predicted_idx} is out of range!")
        print(f"class_names only has {len(class_names)} items")
        return None
    
    prediction = class_names[predicted_idx]
    confidence_score = confidence.item() * 100

    print("Original : ", image_path[56:-14])
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence_score:.2f}%")
    
    # Display image
    plt.imshow(image)
    plt.title(f"{prediction} ({confidence_score:.1f}%)")
    plt.axis('off')
    plt.show()
    
    return prediction

