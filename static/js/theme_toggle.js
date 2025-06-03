document.addEventListener('DOMContentLoaded', () => {
    const themeToggleButton = document.getElementById('theme-toggle-sidebar');
    const body = document.body;
    const themeIcon = themeToggleButton?.querySelector('.theme-icon');
    const themeText = themeToggleButton?.querySelector('.theme-text');

    const updateButtonAppearance = (isDarkMode) => {
        if (!themeIcon || !themeText) return;
        themeIcon.textContent = isDarkMode ? '☀️' : '🌙';
        themeText.textContent = isDarkMode ? 'Light Mode' : 'Dark Mode';
    };

    updateButtonAppearance(body.classList.contains('dark-mode'));

    themeToggleButton?.addEventListener('click', (event) => {
        event.preventDefault();
        const csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');

        fetch('/toggle_theme', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.theme) {
                const isDarkMode = data.theme === 'dark';
                body.classList.toggle('dark-mode', isDarkMode);
                updateButtonAppearance(isDarkMode);
            } else {
                console.error("Theme data missing in response:", data);
            }
        })
        .catch(error => {
            console.error('Error toggling theme:', error);
            alert('Failed to toggle theme. Please try again.');
        });
    });
});
