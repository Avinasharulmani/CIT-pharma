from pymongo import MongoClient

from .config import MONGODB_DB, MONGODB_URI


client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
db = client[MONGODB_DB]

key_messages_collection = db["key_messages"]
analysis_results_collection = db["analysis_results"]
review_annotations_collection = db["review_annotations"]


def init_mongo() -> None:
    client.admin.command("ping")
    try:
        key_messages_collection.drop_index("key_message_id_1")
    except Exception:
        pass
    try:
        key_messages_collection.drop_index("record_key_1")
    except Exception:
        pass
    for document in key_messages_collection.find({"$or": [{"record_key": {"$exists": False}}, {"record_key": None}]}, {"_id": 1, "key_message_id": 1}):
        record_key = f"{document.get('key_message_id', '')}::{document['_id']}"
        key_messages_collection.update_one({"_id": document["_id"]}, {"$set": {"record_key": record_key}})
    key_messages_collection.create_index("record_key", unique=True)
    key_messages_collection.create_index("key_message_id")
    analysis_results_collection.create_index("created_at")
    analysis_results_collection.create_index("file_name")
    review_annotations_collection.create_index([("asset_id", 1), ("content_version_id", 1), ("created_at", -1)])
    review_annotations_collection.create_index([("file_hash", 1), ("created_at", -1)])
    review_annotations_collection.create_index("status")
