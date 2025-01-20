# **Genre Detect API**

This repository provides an API for detecting the genre of an audio file (MP3 format). The API uses a trained machine learning model to classify the genre of music from an uploaded audio file.

---

## **Usage**

### **Publicly Hosted API**
You can directly use the publicly hosted API at **https://jakobrossi.com/api/predict**.

#### Example with `curl`:
```bash
curl -X POST https://jakobrossi.com/api/predict -F "file=@/path/to/audio/file.mp3"
```

#### Example with Postman:
1. Open Postman and create a new **POST** request.
2. Set the URL to: `https://jakobrossi.com/api/predict`.
3. Go to the **Body** tab and select **form-data**.
4. Add a key named `file` and upload an audio file.
5. Click **Send** to receive the predicted genre in the response.

#### Response Example:
```json
{
    "genre": "hiphop"
}
```

---

### **Run Locally**
If you prefer to run the API locally, follow these steps:

#### Prerequisites:
- Python 3.x
- `pip` (Python package installer)

#### Steps to Run Locally:

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the Flask app:
   ```bash
   python app.py
   ```

4. Test the API using `curl`:
   ```bash
   curl -X POST http://127.0.0.1:5000/api/predict -F "file=@path_to_audio_file.mp3"
   ```

5. You will receive a response like:
   ```json
   {
       "genre": "hiphop"
   }
   ```

---

## **Docker Setup**

You can also run the application in a Docker container for easy deployment.

### **Build Docker Image:**

1. Ensure Docker is installed and running on your machine.
2. Navigate to the project directory where the Dockerfile is located.
3. Build the Docker image:
   ```bash
   docker build -t genre-detect-api .
   ```

### **Run Docker Container:**

After the image is built, run it in a container:
```bash
docker run -p 5000:5000 genre-detect-api
```

This will expose the API at `http://localhost:5000`.

---

## **File Structure**
```
/project-root
    /app.py              # Main Flask application
    /requirements.txt    # Python dependencies
    /Dockerfile          # Docker setup file
    /uploads             # Folder for uploaded audio files
```

---

## **Technologies Used**

- **Flask**: Web framework for building the API.
- **librosa**: Python package for music and audio analysis.
- **TensorFlow**: Used for genre prediction (machine learning model).
- **Docker**: Containerization for easy deployment.

---

## **Contributing**

If you would like to contribute to this project, feel free to fork the repository and submit a pull request with your changes. All contributions are welcome!

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
