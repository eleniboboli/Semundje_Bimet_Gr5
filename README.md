# 🌱 Semundje_Bimet_Gr5 - Sistem për Zbulimin e Sëmundjeve të Bimëve

## 📋 Përshkrimi i Projektit

Ky projekt implementon një sistem inteligjent për zbulimin automatik të sëmundjeve të bimëve duke përdorur Deep Learning dhe Convolutional Neural Networks (CNN). Sistemi është në gjendje të klasifikojë imazhe të gjetheve në **39 kategori të ndryshme** të sëmundjeve dhe bimëve të shëndetshme.

###  Qëllimi

Projekti synon të ndihmojë fermerët dhe agronomët në identifikimin e shpejtë dhe preciz të sëmundjeve të bimëve, duke mundësuar trajtim të hershëm dhe të përshtatshëm për të minimizuar humbjet në prodhim.

## 👥 Grupi i Punës

**Grupi 5** - Anëtarët:
- Eleni Boboli
- Elva Rexhepi
- Zhaneta Koti
- Sara Bogdani

## 🚀 Veçoritë Kryesore

- ✅ Klasifikim në 39 klasa të ndryshme sëmundjesh
- ✅ Arkitekturë CNN e ndërtuar me PyTorch
- ✅ Ndërfaqe Web interaktive me Flask
- ✅ Përpunim i imazheve në kohë reale
- ✅ Rekomandime për trajtim dhe fertilizues
- ✅ Dataset i bazuar në PlantVillage
- ✅ Saktësi mbi 95%


## 🛠️ Teknologjitë e Përdorura

- **Python 3.8+**
- **PyTorch** - Deep Learning framework
- **Flask** - Web framework
- **OpenCV** - Image processing
- **NumPy & Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
- **Pillow** - Image handling

## 📦 Instalimi i Shpejtë

### Parakushtet
- Python 3.8 ose më i ri
- pip (Python package manager)
- Git

### Hapat

```bash
# 1. Clone repository
git clone https://github.com/eleniboboli/Semundje_Bimet_Gr5/
cd Semundje_Bimet_Gr5

# 2. Krijo virtual environment
python -m venv venv

# 3. Aktiv izan virtual environment
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# 4. Instalo dependencies
pip install -r requirements.txt

# 5. Shkarko modelin (rreth 50MB)
# Shkarko nga: https://drive.google.com/drive/folders/1ewJWAiduGuld_9oGSrTuLumg9y62qS6A?usp=share_link
# Vendose në: models/plant_disease_model_1.pt

# 6. Run aplikacionin
cd src/app
python app.py

# 7. Hap në browser
# http://localhost:5000
```

## 🎓 Si të Përdoret

### Web Interface

1. Hap `http://localhost:5000` në browser
2. Kliko "AI Engine" ose "Upload Image"
3. Zgjidh një imazh të gjethes së bimës
4. Prit disa sekonda për përpunimin
5. Shiko diagnozën, konfidencën dhe rekomandimet

```

## 📊 Dataset

### PlantVillage Dataset
- **Imazhe totale**: 61,486
- **Klasa**: 39
- **Format**: JPG/JPEG
- **Burimi**: https://data.mendeley.com/datasets/tywbtsjrjv/1

### Bimët e Mbulura
1.Apple_scab
2.Apple_black_rot
3.Apple_cedar_apple_rust
4.Apple_healthy
5.Background_without_leaves
6.Blueberry_healthy
7.Cherry_powdery_mildew
8.Cherry_healthy
9.Corn_gray_leaf_spot
10.Corn_common_rust
11.Corn_northern_leaf_blight
12.Corn_healthy
13.Grape_black_rot
14.Grape_black_measles
15.Grape_leaf_blight
16.Grape_healthy
17.Orange_haunglongbing
18.Peach_bacterial_spot
19.Peach_healthy
20.Pepper_bacterial_spot
21.Pepper_healthy
22.Potato_early_blight
23.Potato_healthy
24.Potato_late_blight
25.Raspberry_healthy
26.Soybean_healthy
27.Squash_powdery_mildew
28.Strawberry_healthy
29.Strawberry_leaf_scorch
30.Tomato_bacterial_spot
31.Tomato_early_blight
32.Tomato_healthy
33.Tomato_late_blight
34.Tomato_leaf_mold
35.Tomato_septoria_leaf_spot
36.Tomato_spider_mites_two-spotted_spider_mite
37.Tomato_target_spot
38.Tomato_mosaic_virus
39.Tomato_yellow_leaf_curl_virus

## 🧠 Modeli

### Arkitektura
- 5 Convolutional layers
- Batch Normalization
- Max Pooling
- Dropout (0.5)
- 3 Fully Connected layers
- image flipping
- Gamma correction
- noise injection
- PCA color augmentation
- rotation
- Scaling

### Performanca
- **Training Accuracy**: ~89%
- **Validation Accuracy**: ~90%
- **Test Accuracy**: ~85%
- **Inference Time**: ~50ms per image


