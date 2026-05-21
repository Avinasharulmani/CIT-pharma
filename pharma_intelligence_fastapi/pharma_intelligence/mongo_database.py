from pymongo import MongoClient

from .config import MONGODB_DB, MONGODB_URI


client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
db = client[MONGODB_DB]

key_messages_collection = db["key_messages"]
analysis_results_collection = db["analysis_results"]
review_annotations_collection = db["review_annotations"]
model_accuracy_history_collection = db["model_accuracy_history"]
pharma_drug_names_collection = db["pharma_drug_names"]
pharma_terms_collection = db["pharma_terms"]
pharma_dosage_units_collection = db["pharma_dosage_units"]
ocr_error_patterns_collection = db["ocr_error_patterns"]


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
    model_accuracy_history_collection.create_index("model_name")
    model_accuracy_history_collection.create_index([("model_name", 1), ("is_measured", 1)])
    for collection in (pharma_drug_names_collection, pharma_terms_collection, pharma_dosage_units_collection):
        collection.create_index([("term", 1), ("active", 1)])
        collection.create_index("source")
    ocr_error_patterns_collection.create_index([("pattern", 1), ("replacement", 1), ("context", 1), ("active", 1)])
