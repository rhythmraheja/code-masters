import os
from flask import Flask
from models import *
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv
from config import CurrentConfig


app = Flask(__name__)
app.config.from_object(CurrentConfig)

# Load the .env file
# load_dotenv()


# app.config.from_object(Config)
# Load environment variables



# Initialize SQLAlchemy



app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
bcrypt = Bcrypt(app)

from routes import *

def check_and_insert_teacher():
    """Check if a teacher exists and insert if not."""
    # Teacher details
    name = 'Ashish'
    email = 'ashish@gmail.com'
    password = 'abcd1234'

    # Check if the teacher already exists in the database by email
    existing_teacher = Teacher.query.filter_by(email=email).first()

    if not existing_teacher:
        # Hash the password with bcrypt
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        # Create a new teacher object
        new_teacher = Teacher(name=name, email=email, password=hashed_password)

        # Add to the database and commit the transaction
        db.session.add(new_teacher)
        db.session.commit()
        print(f"New teacher {name} inserted successfully!")


# Ensure this code runs when the app starts
@app.before_request
def startup():
    with app.app_context():
        check_and_insert_teacher()






    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))

