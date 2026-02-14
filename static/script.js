// session_id tracking
var currentSessionId = generateSessionId();

// Function to generate a unique session ID
function generateSessionId() {
    return 'session-' + new Date().getTime();
}

// Example function to send ask requests with session_id
function sendAskRequest(data) {
    // Include the session_id when sending requests
    data.session_id = currentSessionId;
    // Send the data...
}