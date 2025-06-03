document.addEventListener('DOMContentLoaded', () => {
  const themeToggleBtn = document.getElementById('theme-toggle');
  const body = document.body;

  function applyTheme(theme) {
    if (theme === 'dark') {
      body.classList.add('dark-mode');
      localStorage.setItem('theme', 'dark');
    } else {
      body.classList.remove('dark-mode');
      localStorage.setItem('theme', 'light');
    }
  }

  themeToggleBtn.addEventListener('click', () => {
    const isDarkMode = body.classList.contains('dark-mode');
    applyTheme(isDarkMode ? 'light' : 'dark');
  });

  const savedTheme = localStorage.getItem('theme');
  if (savedTheme) {
    applyTheme(savedTheme);
  } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
    applyTheme('dark');
  }

  // Update button icons
  themeToggleBtn.querySelector('.fa-moon').style.display = savedTheme === 'dark' ? 'none' : 'inline';
  themeToggleBtn.querySelector('.fa-sun').style.display = savedTheme === 'dark' ? 'inline' : 'none';

  // Toggle theme on button click
  themeToggleBtn.addEventListener('click', () => {
    const isDarkMode = body.classList.toggle('dark-mode');
    localStorage.setItem('theme', isDarkMode ? 'dark' : 'light');

    // Update button icons
    themeToggleBtn.querySelector('.fa-moon').style.display = isDarkMode ? 'none' : 'inline';
    themeToggleBtn.querySelector('.fa-sun').style.display = isDarkMode ? 'inline' : 'none';
  });
});

// Get the new sidebar toggle element
const themeToggleSidebarBtn = document.getElementById('theme-toggle-sidebar');
const themeIconSpan = themeToggleSidebarBtn?.querySelector('.theme-icon');
const themeTextSpan = themeToggleSidebarBtn?.querySelector('.theme-text');

const body = document.body;

// Function to apply the theme and update the toggle button's appearance
function applyTheme(theme) {
    if (theme === 'dark') {
        body.classList.add('dark-mode');
        if (themeIconSpan) themeIconSpan.innerHTML = '☀️'; // Sun icon for dark mode
        if (themeTextSpan) themeTextSpan.textContent = 'Light Mode';
        localStorage.setItem('theme', 'dark');
    } else {
        body.classList.remove('dark-mode');
        if (themeIconSpan) themeIconSpan.innerHTML = '🌙'; // Moon icon for light mode
        if (themeTextSpan) themeTextSpan.textContent = 'Dark Mode';
        localStorage.setItem('theme', 'light');
    }
}

// Event listener for the sidebar button
if (themeToggleSidebarBtn) {
    themeToggleSidebarBtn.addEventListener('click', (e) => {
        e.preventDefault();
        const isDarkMode = body.classList.contains('dark-mode');
        applyTheme(isDarkMode ? 'light' : 'dark');
    });
}

// Apply the saved theme or default based on system preference on initial load
document.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

    if (savedTheme) {
        applyTheme(savedTheme);
    } else if (prefersDark) {
        applyTheme('dark');
    } else {
        applyTheme('light');
    }
});
