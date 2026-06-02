// Function to send email text to your Python Flask server
async function checkSpam(emailText) {
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: emailText })
        });
        return await response.json();
    } catch (error) {
        console.error("Error connecting to ML backend:", error);
        return null;
    }
}

// Function to inject a visual warning banner into Gmail
function injectBanner(emailBodyElement, result) {
    // Prevent duplicate banners if already added
    if (document.getElementById('gmail-spam-checker-banner')) return;

    const banner = document.createElement('div');
    banner.id = 'gmail-spam-checker-banner';
    
    const isSpam = result.prediction === 'spam';
    const confidence = (result.confidence * 100).toFixed(1);

    // Style the banner based on the prediction
    banner.style.padding = '12px';
    banner.style.margin = '10px 0';
    banner.style.borderRadius = '4px';
    banner.style.fontWeight = 'bold';
    banner.style.fontFamily = 'Arial, sans-serif';
    
    if (isSpam) {
        banner.style.backgroundColor = '#fce8e6';
        banner.style.color = '#c5221f';
        banner.style.border = '1px solid #fad2cf';
        banner.textContent = `⚠️ Warning: Our AI model classified this email as SPAM (${confidence}% confidence).`;
    } else {
        banner.style.backgroundColor = '#e6f4ea';
        banner.style.color = '#137333';
        banner.style.border = '1px solid #ceead6';
        banner.textContent = `✅ AI Analysis: This email appears safe (Ham - ${confidence}% confidence).`;
    }

    // Insert the banner right above the email body text
    emailBodyElement.parentNode.insertBefore(banner, emailBodyElement);
}

// Watch Gmail's UI to see when an email container is opened
const observer = new MutationObserver(() => {
    // Gmail uses the 'ii gt' classes for the container holding the email body text
    const emailBody = document.querySelector('.ii.gt');
    
    if (emailBody && !emailBody.hasAttribute('data-ml-checked')) {
        // Mark it so we don't scan the same email repeatedly in a loop
        emailBody.setAttribute('data-ml-checked', 'true');

        // Extract the plain text from the email element
        const emailText = emailBody.innerText;

        if (emailText.trim().length > 10) {
            checkSpam(emailText).then(result => {
                if (result) {
                    injectBanner(emailBody, result);
                }
            });
        }
    }
});

// Start monitoring the entire Gmail web page for changes
observer.observe(document.body, { childList: true, subtree: true });