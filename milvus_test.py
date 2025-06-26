import os
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    MilvusClient,
)

# --- 1. Simulation of Your Application (Replace with your actual app) ---
class MockFaceApp:
    def __init__(self, embedding_dim=512):
        self.embedding_dim = embedding_dim
        print(f"MockFaceApp initialized to generate embeddings of dimension {self.embedding_dim}.")

    class Face:
        def __init__(self, embedding):
            self.embedding = embedding

    def get(self, image_path: str):
        """Simulates finding one face in an image and returning its embedding."""
        print(f"  - Analyzing '{image_path}'...")
        random_embedding = np.random.rand(self.embedding_dim).astype('float32')
        random_embedding /= np.linalg.norm(random_embedding)
        return [self.Face(embedding=random_embedding)]

# --- 2. Configuration ---
DB_FILE = "face_database.db"
COLLECTION_NAME = "face_recognition_collection"
EMBEDDING_DIM = 512  # Correctly set to your model's dimension
FACES_DIR = "Faces"  # The folder containing your source images

# Initialize your application
app = MockFaceApp(embedding_dim=EMBEDDING_DIM)

# --- 3. Main Script ---
try:
    # Start and connect to Milvus Lite
    server = MilvusClient("./milvus_demo.db")
    print(f"\nMilvus Lite server started. Using database file: '{DB_FILE}'")
    connections.connect(alias="default", uri=DB_FILE)
    print("✓ Connected to Milvus Lite.")

    # Drop old collection if it exists, for a clean start
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"Dropped existing collection: '{COLLECTION_NAME}'")

    # --- MODIFICATION 1: Add a 'person_name' field to the schema ---
    print("Defining collection schema with a field for the person's name...")
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        # NEW FIELD: Store the name derived from the filename
        FieldSchema(name="person_name", dtype=DataType.VARCHAR, max_length=200),
        # Store the full path to the image for easy retrieval
        FieldSchema(name="source_image_path", dtype=DataType.VARCHAR, max_length=500),
    ]
    schema = CollectionSchema(fields, "Schema for storing face embeddings and names")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    print(f"✓ Collection '{COLLECTION_NAME}' created successfully.")

    # --- MODIFICATION 2: Read images from the 'Faces' folder ---
    print(f"\nPreparing to read images from the '{FACES_DIR}/' folder...")
    
    if not os.path.exists(FACES_DIR):
        print(f"Error: The directory '{FACES_DIR}' was not found. Please create it and add images.")
    else:
        image_files = [f for f in os.listdir(FACES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Prepare lists to hold the data for batch insertion
        embeddings_to_insert = []
        names_to_insert = []
        paths_to_insert = []

        for image_file in image_files:
            full_image_path = os.path.join(FACES_DIR, image_file)
            
            # Use the filename without extension as the person's name
            person_name = os.path.splitext(image_file)[0]
            
            faces = app.get(full_image_path)
            if faces:
                # Store data from the first face found in the image
                embeddings_to_insert.append(faces[0].embedding)
                names_to_insert.append(person_name)
                paths_to_insert.append(full_image_path)

        # --- MODIFICATION 3: Insert the new 'person_name' data ---
        if embeddings_to_insert:
            data_to_insert = [
                embeddings_to_insert,
                names_to_insert,
                paths_to_insert
            ]
            print(f"Inserting {len(embeddings_to_insert)} embeddings into Milvus...")
            insert_result = collection.insert(data_to_insert)
            collection.flush()
            print(f"✓ Data inserted and flushed. Total entities: {collection.num_entities}")

            # --- Create an index for efficient searching ---
            print("\nCreating index for searching...")
            index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
            collection.create_index(field_name="embedding", index_params=index_params)
            collection.load()
            print("✓ Index created and collection loaded into memory.")

            # --- MODIFICATION 4: Retrieve 'person_name' during search ---
            print("\nPerforming a similarity search...")
            query_vector = [embeddings_to_insert[0]] # Use the first face as an example query
            
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            
            results = collection.search(
                data=query_vector,
                anns_field="embedding",
                param=search_params,
                limit=3,
                # Crucially, ask Milvus to return our new fields in the output
                output_fields=["person_name", "source_image_path"] 
            )

            print("\n--- Search Results ---")
            print(f"Searching for faces similar to: '{names_to_insert[0]}'")
            for i, hit in enumerate(results[0]):
                print(f"\nResult {i+1}:")
                print(f"  - Identity (Name): {hit.entity.get('person_name')}")
                print(f"  - Image Location: {hit.entity.get('source_image_path')}")
                print(f"  - Similarity (Distance): {hit.distance:.4f} (lower is more similar)")
            print("--------------------")

finally:
    # Stop the server
    server.stop()
    print("\nServer stopped.")