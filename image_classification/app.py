from flask import Flask, request, jsonify
import torch
from torchvision import transforms, models
from PIL import Image

app = Flask(__name__)
device = torch.device('cpu')

ckpt = torch.load('model.pth', map_location=device)
classes = ckpt['classes']
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(ckpt['model_state'])
model.eval()

transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files.get('image')
    img = Image.open(img_file.stream).convert('RGB')
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(1).item()
    return jsonify({'label': classes[pred]})

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)