from database.firebase_config import db

doc_ref = db.collection("test").document()

doc_ref.set({
    "message": "Backend Connected Successfully"
})

print("Firebase Connected Successfully!")