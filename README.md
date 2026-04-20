🛒 Smart Grocery List Generator

A Flask web application that detects grocery items from kitchen images using a custom **YOLOv8** model, then calculates the groceries still needed by comparing detections with the target stock in `inventory.json`.

✅ User sign-up & login (Flask-Login)  
✅ Upload one or more kitchen images  
✅ Detect grocery items using YOLOv8 / Ultralytics  
✅ Measure detected quantities as pieces, grams, or kg  
✅ Generate a needed-groceries list from target inventory levels  
✅ Download the needed list as CSV or PDF  
✅ Responsive UI with Bootstrap 5 & Animate.css  

## Quantity logic

The model only detects visible grocery items. Needed groceries are calculated in the app layer:

1. YOLO detects item names and bounding boxes.
2. `ITEM_QUANTITY_RULES` in `app.py` decides whether each item is counted from box area as pieces, estimated by area in grams, or treated as kg.
3. `inventory.json` stores the target quantity for each grocery.
4. The app subtracts detected quantity from target quantity and shows the missing amount.

This means you can tune target groceries without retraining the model.

For piece-based groceries, `area_per_piece` is an approximate normalized box area learned from the current dataset labels. It helps when YOLO returns one large box around multiple pieces, but accurate counting still needs training labels with one box per visible item.

## Run

```bash
pip install -r requirements.txt
python app.py
```


