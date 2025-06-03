async function fetchLogs(userId) {
    const response = await fetch(`/get_logs/${userId}`);
    const logs = await response.json();

    let logsHtml = "<h3>Authentication History</h3><ul>";
    logs.forEach(log => {
        logsHtml += `<li>${log[1]} - ${log[0]} (${log[2]})</li>`;
    });
    logsHtml += "</ul>";

    document.getElementById("user-logs").innerHTML = logsHtml;
}
