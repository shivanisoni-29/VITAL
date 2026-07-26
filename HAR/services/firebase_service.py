from database.firebase_config import db

def save_activity(activity):

    db.collection("activities").add(activity)

    return True
def get_current_activity(elder_id):
    doc = db.collection("current_activity").document(elder_id).get()

    if doc.exists:
        return doc.to_dict()

    return None


def update_current_activity(elder_id, activity):
    db.collection("current_activity").document(elder_id).set(activity)
    return True