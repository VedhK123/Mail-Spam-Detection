document.addEventListener('DOMContentLoaded', async () => {
    const statusBadge = document.getElementById('serverStatus');

    try {
        // Ping the Flask backend just to see if it responds
        const response = await fetch('http://localhost:5000/');
        
        // Flask might return a 404 if "/" isn't defined, but if the fetch 
        // succeeds, it means the server is up and listening!
        statusBadge.textContent = 'Connected';
        statusBadge.className = 'badge connected';
    } catch (error) {
        // If fetch fails entirely, the server is shut down
        statusBadge.textContent = 'Offline';
        statusBadge.className = 'badge disconnected';
    }
});