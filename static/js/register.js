document.getElementById('register-form').addEventListener('submit', function(event) {
    event.preventDefault();
    const formData = new FormData(this);

    fetch('/register', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('error-message').innerText = data.error;  // Show actual error
        } else {
            window.location.href = '/login';  // Redirect on success
        }
    })
    .catch(error => console.error('Error:', error));
});
