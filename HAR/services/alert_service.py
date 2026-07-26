from database.firebase_config import db
from utils.timestamp import get_current_timestamp


def create_alert(elder_id):

    alert = {
        "elderId": elder_id,
        "type": "Fall Detected",
        "timestamp": get_current_timestamp(),
        "status": "Unread"
    }

    db.collection("alerts").add(alert)

    return True