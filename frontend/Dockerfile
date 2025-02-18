# Use the official Node.js image
FROM node:14

# Set working directory
WORKDIR /app

# Install dependencies
COPY package.json /app/
RUN npm install

# Copy all the app files
COPY . /app/

# Expose the port
EXPOSE 3000

# Run the app
CMD ["npm", "start"]
