import os
from flask import Flask
from models import db, Teacher
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv
from config import CurrentConfig

app = Flask(__name__, static_folder="static")
app.config.from_object(CurrentConfig)

# Load environment variables
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Extensions
db.init_app(app)
bcrypt = Bcrypt(app)

# Import routes after initializing extensions
from routes import *  

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
