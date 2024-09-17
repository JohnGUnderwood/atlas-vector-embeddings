from classes import MongoDBConnection,Embeddings

# Set up your MongoDB connection and specify collection and inputs/outputs
connection=MongoDBConnection()
db=connection.get_database()
collection = db.azure
embedder=Embeddings()

# Initialize the change stream
change_stream = collection.watch([], full_document='updateLookup')

# Function to populate all the initial embeddings by detecting any fields with missing embeddings
def initial_sync():
    # We only care about documents with missing keys
    query = {"embedding": {"$exists": False}}
    results = collection.find(query)

    # Every document gets a new embedding
    total_records = 0
    for result in results:
        collection.update_one({"_id":result["_id"]}, {"$set": {"embedding":embedder.get_embedding(result['content'])}})

    return total_records

# Function to handle changes in the collection
def handle_changes(change):
    # Extract the necessary information from the change document
    operation_type = change['operationType']

    # Bail out if the detected update is the embedding we just did!
    if operation_type == "update" and 'embedding' in change['updateDescription']['updatedFields']:
        return

    # Anytime we create, update or replace documents, the embedding needs to be updated
    if operation_type == "replace" or operation_type == "update" or operation_type == "insert":
        # Get the _id for update later and our input field to vectorize
        entry = change['fullDocument']
        collection.update_one({"_id":entry["_id"]}, {"$set": {"embedding":embedder.get_embedding(entry['content'])}})
        print(f"Updated embedding for {entry['_id']}")
            
# Perform initial sync
print(f"Initial sync for {db.name} db and {collection.name} collection. Watching for changes to 'content' and writing to 'embedding'...")
total_records = initial_sync()
print(f"Sync complete {total_records} missing embeddings")
print(f"Change stream active...")

# Start consuming changes from the change stream
for change in change_stream:
    handle_changes(change)