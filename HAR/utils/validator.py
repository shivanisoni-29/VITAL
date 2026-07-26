VALID_ACTIVITIES = [
    "Walking",
    "Standing",
    "Sitting",
    "Sleeping",
    "Running",
    "Jump",
    "Checking Fall...",
    "Fall Detected"
]

def validate_activity(data):

    if "elderId" not in data:
        return False, "elderId is required"

    if "activity" not in data:
        return False, "activity is required"

    if data["activity"] not in VALID_ACTIVITIES:
        return False, "Invalid Activity"

    return True, "Valid"