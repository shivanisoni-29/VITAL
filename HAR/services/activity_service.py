from models.activity_model import Activity
from utils.timestamp import get_current_timestamp
from utils.validator import validate_activity
from services.firebase_service import (
    save_activity, get_current_activity,
    update_current_activity
)
from services.alert_service import create_alert

def process_activity(data):
    print("Received Data:", data)

    # Validate incoming data
    is_valid, message = validate_activity(data)

    if not is_valid:
        return {
            "success": False,
            "message": message
        }

    # Create Activity object
    activity = Activity(
    elder_id=data["elderId"],
    activity=data["activity"],
    timestamp=get_current_timestamp()
)

    # Save to Firebase
    # Check last saved activity
    current_activity = get_current_activity(activity.elder_id)
    print("Current Activity:", current_activity)

# Save only if activity changed
    if current_activity is None or current_activity["activity"] != activity.activity:
      print("Saving activity...")

      save_activity(activity.to_dict())
      print("Activity Saved")
      update_current_activity(
        activity.elder_id,
        activity.to_dict()
      )
      print("Current activity updated")

    # Generate alert only when a new Falling activity is saved
      if activity.activity == "Fall Detected":
        create_alert(activity.elder_id)

    return {
        "success": True,
        "message": "Activity saved successfully",
        "data": activity.to_dict()
    }