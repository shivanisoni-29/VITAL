class Activity:

    def __init__(self, elder_id, activity, timestamp):
        self.elder_id = elder_id
        self.activity = activity
        self.timestamp = timestamp

    def to_dict(self):
        return {
            "elderId": self.elder_id,
            "activity": self.activity,
            "timestamp": self.timestamp
        }