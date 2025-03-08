from flask import Flask
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from dotenv import load_dotenv
import os

from config import CurrentConfig
from models import db, Teacher

app = Flask(__name__, static_folder="static")
app.config.from_object(CurrentConfig)

# Load environment variables
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Extensions
db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.login_view = "index"  # Redirect to login if not authenticated
login_manager.init_app(app)

# Import routes AFTER initializing extensions
from routes import *

@login_manager.user_loader
def load_user(user_id):
    return Teacher.query.get(int(user_id))

def check_and_insert_teacher():
    """Check if a teacher exists and insert if not."""
    name = 'Ashish'
    email = 'ashish@gmail.com'
    password = 'abcd1234'

    with app.app_context():
        existing_teacher = Teacher.query.filter_by(email=email).first()
        if not existing_teacher:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            new_teacher = Teacher(name=name, email=email, password=hashed_password)
            db.session.add(new_teacher)
            db.session.commit()
            print(f"New teacher {name} inserted successfully!")

# Run database check at startup
with app.app_context():
    check_and_insert_teacher()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
