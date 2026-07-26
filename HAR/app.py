from flask import Flask
from routes.activity_routes import activity_bp

app = Flask(__name__)

app.register_blueprint(activity_bp)


@app.route("/")
def home():
    return "HAR Backend Running Successfully"


if __name__ == "__main__":
    app.run(debug=True)