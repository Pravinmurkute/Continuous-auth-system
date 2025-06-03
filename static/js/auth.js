fetch("/authenticate")
  .then(response => response.json())
  .then(data => {
      console.log("API Response:", data); // Debugging log
      document.getElementById("auth-status").innerText = 
         data.status === "success" ? "Authentication Status: Success" : "Authentication Status: Fail";
  })
  .catch(error => console.error("Error:", error));
