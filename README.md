# 🛒 Smart Grocery List Generator

An AI-powered full-stack web application that detects grocery items from kitchen images using a custom **YOLOv8** model and generates a dynamic list of required groceries by comparing detections with predefined inventory targets.

---

## 🚀 Key Features

- 🔐 User authentication (Flask-Login)  
- 🖼️ Upload one or multiple kitchen images  
- 🤖 Grocery detection using **YOLOv8 (Ultralytics)**  
- ⚖️ Quantity estimation in pieces, grams, or kilograms  
- 📊 Automatic calculation of missing groceries based on target inventory  
- 📄 Export results as **CSV or PDF**  
- 🎨 Responsive UI with Bootstrap 5 & Animate.css  

---

## 🧠 How It Works

1. **Object Detection**  
   YOLOv8 detects grocery items from uploaded images.

2. **Quantity Estimation**  
   - Uses bounding box area to estimate quantity  
   - Rules defined in `ITEM_QUANTITY_RULES`  
   - Supports:
     - Piece-based items  
     - Weight-based estimation (grams/kg)

3. **Inventory Comparison**  
   - `inventory.json` stores target quantities  
   - Detected quantities are subtracted from targets  

4. **Output Generation**  
   - Displays missing items  
   - Allows export as CSV/PDF  

---

## ⚙️ Quantity Logic

- Detection is based only on visible items  
- Quantity is inferred using bounding box area  
- `area_per_piece` is an approximate value derived from dataset annotations  

⚠️ **Note:**  
Accurate counting requires training data with one bounding box per item.

---

## 🧩 Tech Stack

- **Backend:** Flask  
- **AI/ML:** YOLOv8 (Ultralytics)  
- **Frontend:** HTML, CSS, Bootstrap 5, Animate.css  
- **Data Handling:** JSON (inventory), CSV, PDF (FPDF)  

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python app.py