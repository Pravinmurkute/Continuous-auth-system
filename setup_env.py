import os

# Define environment variables
env_content = """SECRET_KEY="5d609b34c7eba8b0cb3fcd9765c3e917183c772ce56577cd91d40dd3a9818p0606"
MYSQL_HOST="localhost"
MYSQL_USER="root"
MYSQL_PASSWORD="Pravin0606"
MYSQL_DB="continuous_auth"
MAIL_USERNAME="pravinmurkute2025@gmail.com"
MAIL_PASSWORD="Pravi@0606##2447Project"
"""

# Create .env file
env_path = ".env"

if not os.path.exists(env_path):
    with open(env_path, "w") as env_file:
        env_file.write(env_content)
    print("✅ .env file created successfully!")
else:
    print("⚠️ .env file already exists. No changes made.")

# Add .env to .gitignore
gitignore_path = ".gitignore"

if not os.path.exists(gitignore_path):
    with open(gitignore_path, "w") as gitignore_file:
        gitignore_file.write(".env\n")
    print("✅ .gitignore file created and .env added!")
else:
    with open(gitignore_path, "r+") as gitignore_file:
        content = gitignore_file.read()
        if ".env" not in content:
            gitignore_file.write("\n.env\n")
            print("✅ Added .env to .gitignore!")
        else:
            print("⚠️ .env already present in .gitignore.")

# Print confirmation
print("\n🔹 Your environment variables:")
print(env_content)

print("\n🚀 Now you can load these variables in Flask using `dotenv`.")
