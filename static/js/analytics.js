document.addEventListener('DOMContentLoaded', () => {
    loadLoginTrendChart();
    loadSuccessRateChart();
    loadUserActivityTable();
});

async function loadLoginTrendChart() {
    try {
        const response = await fetch('/api/analytics/logins_over_time');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        const ctx = document.getElementById('loginTrendChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels || [],
                datasets: [
                    {
                        label: 'Success',
                        data: data.success_data || [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    },
                    {
                        label: 'Fail',
                        data: data.fail_data || [],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    }
                ]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    } catch (error) {
        console.error("Error loading login trend chart:", error);
    }
}

async function loadSuccessRateChart() {
    try {
        const response = await fetch('/api/analytics/success_rate');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        const total = (data.success || 0) + (data.fail || 0);
        const successRate = total > 0 ? ((data.success / total) * 100).toFixed(1) : 0;
        const failRate = total > 0 ? ((data.fail / total) * 100).toFixed(1) : 0;
        document.getElementById('successRateText').innerText =
            `Success: ${data.success} (${successRate}%), Fail: ${data.fail} (${failRate}%)`;
        const ctx = document.getElementById('successRateChart').getContext('2d');
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Success', 'Fail'],
                datasets: [{
                    data: [data.success || 0, data.fail || 0],
                    backgroundColor: ['rgb(75, 192, 192)', 'rgb(255, 99, 132)'],
                    hoverOffset: 4
                }]
            }
        });
    } catch (error) {
        console.error("Error loading success rate chart:", error);
    }
}

async function loadUserActivityTable() {
    try {
        const response = await fetch('/api/analytics/logins_per_user');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        const tableBody = document.querySelector('#userActivityTable tbody');
        tableBody.innerHTML = '';
        if (data && data.length > 0) {
            data.forEach(user => {
                const row = tableBody.insertRow();
                row.insertCell().textContent = user.username;
                row.insertCell().textContent = user.login_count;
            });
        } else {
            const row = tableBody.insertRow();
            row.insertCell().colSpan = 2;
            row.insertCell().textContent = "No user login activity found.";
        }
    } catch (error) {
        console.error("Error loading user activity table:", error);
    }
}
