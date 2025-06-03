import os

# Get the absolute path to the app directory
app_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(app_dir, "app", "static")

# Create the static directory if it doesn't exist
os.makedirs(static_dir, exist_ok=True)

print(f"Created static directory at: {static_dir}")