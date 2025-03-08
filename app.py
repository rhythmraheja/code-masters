from flask import Flask
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from dotenv import load_dotenv
import os

from config import CurrentConfig
from models import db, Teacher


app = Flask(__name__, static_folder="static")
with app.app_context():
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



# Run database check at startup


    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port)
