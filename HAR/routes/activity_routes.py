from flask import Blueprint, request, jsonify
from services.activity_service import process_activity

activity_bp = Blueprint("activity", __name__)

@activity_bp.route("/activity", methods=["POST"])
def receive_activity():

    data = request.get_json()

    result = process_activity(data)

    return jsonify(result), 200