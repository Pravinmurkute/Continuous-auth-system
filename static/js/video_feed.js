fetch("http://127.0.0.1:5000/video_feed")
  .then(response => response.blob())
  .then(blob => {
      document.getElementById("videoElement").src = URL.createObjectURL(blob);
  })
  .catch(error => console.error("❌ Fetch error:", error));
