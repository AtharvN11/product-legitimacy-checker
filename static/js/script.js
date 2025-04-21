document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("url-form");
    const resultContainer = document.getElementById("result");
  
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
  
        const url = document.getElementById("url").value;
        resultContainer.innerHTML = `
            <div class="loading">
                <p>Analyzing product... Please wait</p>
                <div class="spinner"></div>
            </div>
        `;
  
        try {
            const res = await fetch("/", {
                method: "POST",
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                body: `url=${encodeURIComponent(url)}`
            });
  
            const data = await res.json();
  
            if (data.error) {
                resultContainer.innerHTML = `
                    <div class="error-message">
                        <h3>⚠️ Error</h3>
                        <p>${data.error}</p>
                    </div>`;
                return;
            }
  
            // Calculate color based on legitimacy score
            const scoreColor = data.confidence >= 70 ? '#4CAF50' : 
                             data.confidence >= 50 ? '#FFA726' : '#F44336';
  
            resultContainer.innerHTML = `
                <div class="result-card">
                    <h2>Analysis Results</h2>
                    
                    <div class="product-info">
                        <h3>${data.title}</h3>
                        <p><strong>Price:</strong> ${data.price}</p>
                        <p><strong>Rating:</strong> ${data.rating}</p>
                    </div>

                    <div class="score-section">
                        <h3>Legitimacy Score</h3>
                        <div class="score-bar">
                            <div class="score-bar-inner" 
                                 style="width: ${data.confidence}%; background-color: ${scoreColor}">
                            </div>
                        </div>
                        <p class="score-label" style="color: ${scoreColor}">
                            ${data.confidence}% Legitimate
                        </p>
                    </div>

                    <div class="risk-scores">
                        <div class="risk-item">
                            <h4>Risk Assessment</h4>
                            <p>Malicious Score: ${data.malicious_score}%</p>
                            <p>Unworthy Score: ${data.unworthy_score}%</p>
                        </div>
                    </div>

                    <div class="expert-analysis">
                        <h4>Expert Analysis</h4>
                        <p>${data.reason}</p>
                    </div>
                </div>`;
        } catch (err) {
            console.error("Error:", err);
            resultContainer.innerHTML = `
                <div class="error-message">
                    <h3>⚠️ Error</h3>
                    <p>Something went wrong. Please try again.</p>
                </div>`;
        }
    });
});
  