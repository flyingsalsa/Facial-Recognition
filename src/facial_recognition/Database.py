from pymilvus import connections, utility
from pymilvus.server import default_server

# 1. Start the Milvus Lite server
# This will create a file named 'milvus_demo.db' in your current directory
# to store all the data.
print("Starting Milvus Lite...")
default_server.start()
print("Milvus Lite started.")

# 2. Connect to the server
# The alias 'default' is the standard name for the connection.
try:
    connections.connect(alias="default", uri="default_server.db")
    print("Successfully connected to Milvus Lite.")

    # 3. Check if the server is healthy
    print(f"Server health: {utility.get_server_version()}")

    # You can now perform Milvus operations here
    # For example:
    # collection_name = "my_first_collection"
    # if utility.has_collection(collection_name):
    #     utility.drop_collection(collection_name)
    # ... and so on

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # 4. Stop the server when you are done
    # (Optional, you can leave it running)
    print("Stopping Milvus Lite...")
    default_server.stop()
    print("Milvus Lite stopped.")