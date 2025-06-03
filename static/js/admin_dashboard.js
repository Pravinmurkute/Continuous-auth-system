function fetchSystemStats() {
    const statsContainer = document.getElementById('systemStats');
    fetch('/api/admin/stats')
        .then(response => response.json())
        .then(data => {
            document.getElementById('stat-total-users').textContent = data.total_users || '--';
            document.getElementById('stat-total-logins').textContent = data.total_logins || '--';
            document.getElementById('stat-total-failures').textContent = data.total_failures || '--';
        })
        .catch(error => console.error('Error fetching stats:', error));
}

function fetchUserList() {
    const tableBody = document.getElementById('userListTable').querySelector('tbody');
    fetch('/api/admin/users')
        .then(response => response.json())
        .then(data => {
            tableBody.innerHTML = '';
            data.forEach(user => {
                const row = tableBody.insertRow();
                row.insertCell(0).textContent = user.user_id;
                row.insertCell(1).textContent = user.username;
                row.insertCell(2).textContent = user.email;
                row.insertCell(3).textContent = user.role;
                row.insertCell(4).textContent = user.last_login || 'Never';
            });
        })
        .catch(error => console.error('Error fetching user list:', error));
}

function fetchSystemLogs() {
    const tableBody = document.getElementById('systemLogsTable').querySelector('tbody');
    fetch('/api/admin/logs')
        .then(response => response.json())
        .then(data => {
            tableBody.innerHTML = '';
            data.forEach(log => {
                const row = tableBody.insertRow();
                row.insertCell(0).textContent = log.timestamp;
                row.insertCell(1).textContent = log.user_id || 'N/A';
                row.insertCell(2).textContent = log.username || 'Unknown';
                row.insertCell(3).textContent = log.event_type;
                row.insertCell(4).textContent = log.status;
                row.insertCell(5).textContent = log.details;
            });
        })
        .catch(error => console.error('Error fetching logs:', error));
}

document.addEventListener('DOMContentLoaded', () => {
    fetchSystemStats();
    fetchUserList();
    fetchSystemLogs();
});
