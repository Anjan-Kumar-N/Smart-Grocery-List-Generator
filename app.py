from flask import Flask, render_template, request, redirect, url_for, send_file, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import json
import csv
import math
from fpdf import FPDF
from ultralytics import YOLO
import cv2
from uuid import uuid4

# ============================================================
# App Setup
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
os.makedirs(INSTANCE_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")

default_sqlite_uri = f"sqlite:///{os.path.join(INSTANCE_DIR, 'users.db')}"
database_uri = os.environ.get("DATABASE_URL", default_sqlite_uri)
if database_uri.startswith("postgres://"):
    database_uri = database_uri.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_uri
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(BASE_DIR, 'runs', 'detect', 'train_milk_ghee', 'weights', 'best.pt')
)
INVENTORY_PATH = os.path.join(BASE_DIR, 'inventory.json')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================================
# Categories
# ============================================================

categories = {
    "Fruits": ["apple","orange","grapes","pomegranate","lemon"],
    "Vegetables": ["tomato","capsicum","carrot","cucumber","turnip","ladyfinger"],
    "Pulses": ["chana","chanadal","moongdal","greengram","groundnut"],
    "Dairy": ["paneer","milk","ghee"]
}

ITEM_QUANTITY_RULES = {
    "apple": {"unit": "pcs", "measure": "area_count", "target": 4, "area_per_piece": 0.1439},
    "orange": {"unit": "pcs", "measure": "area_count", "target": 5, "area_per_piece": 0.2154},
    "grapes": {"unit": "grams", "measure": "area", "target": 500, "pixels_per_gram": 30000},
    "pomegranate": {"unit": "pcs", "measure": "area_count", "target": 4, "area_per_piece": 0.1220},
    "lemon": {"unit": "pcs", "measure": "area_count", "target": 6, "area_per_piece": 0.0660},
    "tomato": {"unit": "grams", "measure": "area", "target": 1000, "pixels_per_gram": 15000},
    "capsicum": {"unit": "pcs", "measure": "area_count", "target": 2, "area_per_piece": 0.1299},
    "carrot": {"unit": "grams", "measure": "area", "target": 250, "pixels_per_gram": 28000},
    "cucumber": {
        "unit": "pcs",
        "measure": "area_count",
        "target": 8,
        "area_per_piece": 0.20,
        "width_per_piece": 0.20,
        "single_aspect_threshold": 1.8,
        "rounding": "round",
        "width_rounding": "round"
    },
    "turnip": {"unit": "pcs", "measure": "area_count", "target": 9, "area_per_piece": 0.14},
    "ladyfinger": {"unit": "grams", "measure": "area", "target": 500, "pixels_per_gram": 45000},
    "chana": {"unit": "grams", "measure": "area", "target": 1000, "pixels_per_gram": 700},
    "chanadal": {"unit": "grams", "measure": "area", "target": 1000, "pixels_per_gram": 800},
    "moongdal": {"unit": "kg", "measure": "count", "target": 1},
    "greengram": {"unit": "kg", "measure": "count", "target": 1},
    "groundnut": {"unit": "kg", "measure": "count", "target": 1},
    "paneer": {"unit": "grams", "measure": "area", "target": 500, "pixels_per_gram": 900},
    "milk": {"unit": "pcs", "measure": "area_count", "target": 2, "area_per_piece": 0.62},
    "ghee": {"unit": "pcs", "measure": "area_count", "target": 1, "area_per_piece": 0.62}
}

PIXELS_PER_GRAM = 30000.0
DETECTION_MIN_CONF = 0.45
CLASS_MIN_CONF = {
    "lemon": 0.55
}


def get_category(item_name):
    for category, items in categories.items():
        if item_name in items:
            return category
    return None


def load_inventory_targets():
    targets = {
        item: rule["target"]
        for item, rule in ITEM_QUANTITY_RULES.items()
    }

    try:
        with open(INVENTORY_PATH, "r", encoding="utf-8") as file:
            saved_targets = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return targets

    for item, target in saved_targets.items():
        if item in targets and isinstance(target, (int, float)):
            targets[item] = float(target)

    return targets


def measure_detection(item_name, box, image_shape=None):
    rule = ITEM_QUANTITY_RULES.get(item_name, {"measure": "count"})
    x1, y1, x2, y2 = box
    area = max((x2 - x1) * (y2 - y1), 0)

    if rule["measure"] == "area_count":
        if image_shape is None:
            return 1.0

        image_height, image_width = image_shape[:2]
        image_area = max(image_width * image_height, 1)
        box_width = max(x2 - x1, 0)
        box_height = max(y2 - y1, 0)
        normalized_width = float(box_width / max(image_width, 1))
        normalized_height = float(box_height / max(image_height, 1))
        aspect_ratio = normalized_height / max(normalized_width, 0.0001)
        normalized_area = float(area / image_area)
        area_per_piece = rule.get("area_per_piece", normalized_area)
        raw_count = normalized_area / max(area_per_piece, 0.0001)
        rounding = rule.get("rounding", "round")

        if rounding == "ceil":
            estimated_count = math.ceil(raw_count)
        elif rounding == "floor":
            estimated_count = math.floor(raw_count)
        else:
            estimated_count = round(raw_count)

        # For grouped side-by-side objects like cucumber, width is a strong count cue.
        if "width_per_piece" in rule:
            width_per_piece = max(rule["width_per_piece"], 0.0001)
            raw_width_count = normalized_width / width_per_piece
            width_rounding = rule.get("width_rounding", "ceil")

            if width_rounding == "floor":
                width_estimate = math.floor(raw_width_count)
            elif width_rounding == "round":
                width_estimate = round(raw_width_count)
            else:
                width_estimate = math.ceil(raw_width_count)

            estimated_count = max(estimated_count, width_estimate)

        # Tall narrow box is typically a single cucumber packet/item.
        single_aspect_threshold = rule.get("single_aspect_threshold")
        if single_aspect_threshold and aspect_ratio >= single_aspect_threshold:
            estimated_count = 1

        return float(max(estimated_count, 1))

    if rule["measure"] == "area":
        pixels_per_gram = rule.get("pixels_per_gram", PIXELS_PER_GRAM)
        return float(area / pixels_per_gram)

    return 1.0


def format_quantity(value, unit):
    if unit == "pcs":
        return f"{int(round(value))} pcs"

    if unit == "kg":
        rounded = float(round(value, 2))
        if rounded.is_integer():
            rounded = int(rounded)
        return f"{rounded} kg"

    grams = int(round(value))
    if grams >= 1000:
        kg = float(round(grams / 1000, 2))
        if kg.is_integer():
            kg = int(kg)
        return f"{kg} kg"

    return f"{grams} grams"


def build_file_name(original_name):
    safe_name = secure_filename(original_name)
    stem, ext = os.path.splitext(safe_name)

    if not stem:
        stem = "upload"

    return f"{stem}_{uuid4().hex[:8]}{ext}"

# ============================================================
# User Model
# ============================================================

class User(UserMixin, db.Model):

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True)
    password_hash = db.Column(db.String(150))


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


with app.app_context():
    db.create_all()

# ============================================================
# Load Model
# ============================================================

model = YOLO(MODEL_PATH)

# ============================================================
# Signup
# ============================================================

@app.route('/signup', methods=['GET','POST'])
def signup():

    if request.method == 'POST':

        username = request.form['username']
        password = request.form['password']

        existing_user = User.query.filter_by(username=username).first()

        if existing_user:
            flash('Username already exists!')
            return redirect(url_for('signup'))

        hashed_pw = generate_password_hash(password)

        new_user = User(username=username,password_hash=hashed_pw)

        db.session.add(new_user)
        db.session.commit()

        flash('Account created! Please login.')

        return redirect(url_for('login'))

    return render_template('signup.html')

# ============================================================
# Login
# ============================================================

@app.route('/login', methods=['GET','POST'])
def login():

    if request.method == 'POST':

        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password_hash,password):

            login_user(user)

            return redirect(url_for('index'))

        flash('Invalid credentials.')

    return render_template('login.html')

# ============================================================
# Logout
# ============================================================

@app.route('/logout')
@login_required
def logout():

    logout_user()

    return redirect(url_for('login'))

# ============================================================
# Main Detection
# ============================================================

@app.route('/', methods=['GET','POST'])
@login_required
def index():

    if request.method == 'POST':

        images = request.files.getlist('images')

        if not images or images[0].filename == '':
            flash("Upload at least one image.")
            return redirect(url_for('index'))

        detected_items = {}
        quantity_required = {}
        annotated_images = []

        for image in images:

            filename = build_file_name(image.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            image.save(path)

            results = model(path)

            for result in results:

                boxes = result.boxes
                names = result.names

                if boxes is None or len(boxes) == 0:
                    continue

                cls_ids = boxes.cls.cpu().numpy()
                coords = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy() if boxes.conf is not None else [1.0] * len(cls_ids)

                for cls, box, conf in zip(cls_ids, coords, confs):

                    item = names[int(cls)]

                    if item not in ITEM_QUANTITY_RULES:
                        continue

                    min_conf = CLASS_MIN_CONF.get(item, DETECTION_MIN_CONF)
                    if conf < min_conf:
                        continue

                    detected_items[item] = detected_items.get(item,0) + measure_detection(item, box, result.orig_shape)

            annotated = results[0].plot()
            annotated_filename = f"annotated_{filename}"

            annotated_path = os.path.join(
                UPLOAD_FOLDER,
                annotated_filename
            )

            cv2.imwrite(
                annotated_path,
                cv2.cvtColor(annotated,cv2.COLOR_RGB2BGR)
            )

            annotated_images.append(annotated_filename)

        inventory_targets = load_inventory_targets()

        # Required Quantity

        for item, required in inventory_targets.items():
            detected = detected_items.get(item,0)
            quantity_required[item] = float(max(required-detected,0))

        grocery_list = []

        for item,remaining in quantity_required.items():

            if remaining > 0:
                rule = ITEM_QUANTITY_RULES[item]
                category = get_category(item)

                grocery_list.append({
                    "name": item,
                    "category": category,
                    "quantity": remaining,
                    "display_quantity": format_quantity(remaining, rule["unit"])
                })

        # ============================
        # Categorize Detected
        # ============================

        detected_categories = {cat: [] for cat in categories}

        for k,v in detected_items.items():

            rule = ITEM_QUANTITY_RULES[k]

            item_data = {
                "name":k.capitalize(),
                "quantity":format_quantity(v, rule["unit"])
            }

            category = get_category(k)

            if category:
                detected_categories[category].append(item_data)

        # ============================
        # Categorize Required
        # ============================

        required_categories = {cat: [] for cat in categories}

        for entry in grocery_list:

            item_data = {
                "name":entry["name"].capitalize(),
                "quantity":entry["display_quantity"]
            }

            category = entry["category"]

            if category:
                required_categories[category].append(item_data)

        session["grocery_list_data"] = grocery_list

        return render_template(
            'result.html',
            detected_categories=detected_categories,
            required_categories=required_categories,
            annotated_images=annotated_images
        )

    return render_template('index.html')

# ============================================================
# CSV Download
# ============================================================

@app.route('/download_csv')
@login_required
def download_csv():

    grocery_list_data = session.get("grocery_list_data", [])

    csv_path = os.path.join(BASE_DIR, 'static', 'grocery_list.csv')

    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:

        writer = csv.writer(file)

        # Header
        writer.writerow(["Category", "Item", "Required Quantity"])

        # Loop through categories
        for category in categories:

            for entry in grocery_list_data:

                if entry.get("category") == category:

                    writer.writerow([
                        category,
                        entry["name"].capitalize(),
                        entry["display_quantity"]
                    ])

    return send_file(csv_path, as_attachment=True)


# ============================================================
# PDF Download 
# ============================================================
@app.route('/download_pdf')
@login_required
def download_pdf():

    grocery_list_data = session.get("grocery_list_data", [])

    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(34,139,34)
    pdf.set_text_color(255,255,255)
    pdf.set_font("Helvetica","B",22)

    pdf.cell(0,15,"Smart Grocery List Generator",align="C",fill=True,new_x="LMARGIN",new_y="NEXT")

    pdf.ln(5)

    pdf.set_text_color(60,60,60)
    pdf.set_font("Helvetica","I",12)

    pdf.cell(
        0,
        8,
        "Fresh kitchen inventory generated by Smart Grocery AI",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT"
    )

    pdf.ln(8)

    row_num = 1

    # Loop categories
    for category in categories:

        category_rows = []

        # Collect rows belonging to this category
        for entry in grocery_list_data:

            if entry.get("category") == category:
                category_rows.append((
                    entry["name"],
                    entry["display_quantity"]
                ))

        # Skip category if empty
        if len(category_rows) == 0:
            continue

        # Category title
        pdf.set_font("Helvetica","B",15)
        pdf.set_text_color(34,139,34)

        pdf.cell(0,10,category,new_x="LMARGIN",new_y="NEXT")

        pdf.set_text_color(0,0,0)

        # Table header
        pdf.set_font("Helvetica","B",12)
        pdf.set_fill_color(200,230,200)

        pdf.cell(15,10,"#",border=1,align="C",fill=True)
        pdf.cell(90,10,"Item",border=1,align="C",fill=True)
        pdf.cell(85,10,"Required Quantity",border=1,align="C",fill=True,new_x="LMARGIN",new_y="NEXT")

        pdf.set_font("Helvetica","",11)

        # Table rows
        for name, qty in category_rows:

            fill_color = (255,255,255) if row_num % 2 == 0 else (240,255,240)
            pdf.set_fill_color(*fill_color)

            pdf.cell(15,10,str(row_num),border=1,align="C",fill=True)
            pdf.cell(90,10,name.capitalize(),border=1,align="L",fill=True)
            pdf.cell(85,10,qty,border=1,align="C",fill=True,new_x="LMARGIN",new_y="NEXT")

            row_num += 1

        pdf.ln(4)

    # Footer
    pdf.ln(6)
    pdf.set_font("Helvetica","I",11)
    pdf.set_text_color(120,120,120)

    pdf.multi_cell(
        0,
        8,
        "Generated using Smart Grocery List Generator!",
        align="C"
    )

    pdf_path = os.path.join(BASE_DIR, "static", "grocery_list.pdf")
    pdf.output(pdf_path)

    return send_file(pdf_path, as_attachment=True)


# ============================================================
# Run App
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
