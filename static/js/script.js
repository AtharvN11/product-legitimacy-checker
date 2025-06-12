async function scrape() {
  const url = document.getElementById("urlInput").value.trim();
  const loading = document.getElementById("loading");
  const result = document.getElementById("result");

  if (!url) {
    result.innerHTML = `<p style="color:red;">Please enter a valid Amazon product URL.</p>`;
    return;
  }

  loading.style.display = "block";
  result.innerHTML = "";

  try {
    const response = await fetch("/scrape", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url })
    });

    const data = await response.json();
    console.log("RESPONSE JSON:", data);
    loading.style.display = "none";

    if (!response.ok) {
      console.error("Backend error:", data);
      result.innerHTML = `<p style="color:red;">${data.error || "An unknown error occurred."}</p>`;
      return;
    }

    // Handle insufficient review case
    // if (data.status === "insufficient_reviews") {
    //   result.innerHTML = `
    //     <div class="warning p-3 bg-warning-subtle border rounded">
    //       <h4 class="text-warning">⚠️ Insufficient Reviews</h4>
    //       <p>${data.message}</p>
    //     </div>`;
    //   return;
    // }

    // Determine card color based on legitimacy
    let bgColor = "#f8d7da", verdict = "Likely Scam", badge = "danger";
    if (data.confidence >= 70) {
      bgColor = "#d4edda";
      verdict = "Legitimate";
      badge = "success";
    } else if (data.confidence >= 50) {
      bgColor = "#fff3cd";
      verdict = "Suspicious";
      badge = "warning";
    }

    // Render legitimacy metrics
    result.innerHTML = `
      <div class="result-card p-4 rounded shadow-sm mb-4" style="background-color: ${bgColor};">
        <h3>ASIN: ${data.asin}</h3>
        <p><strong>Legitimacy Score:</strong> ${data.confidence}%</p>
        <p><strong>Fake Review Estimate:</strong> ${data.fake_review_percent}%</p>
        <p><strong>Unworthy Score:</strong> ${data.unworthy_score}%</p>
        <p><strong>Verdict:</strong> <span class="badge bg-${badge}">${verdict}</span></p>
        <hr/>
        <h4>Top Reviews:</h4>
      </div>`;

    // Append reviews
    data.reviews.slice(0, 5).forEach((r) => {
      result.innerHTML += `
        <div class="review shadow-sm mb-3 p-3 rounded">
          <h5>${r.title || "No Title"}</h5>
          <p><strong>Author:</strong> ${r.author || "Unknown"}</p>
          <p><strong>Rating:</strong> ${r.rating || "N/A"}/5</p>
          <p>${(r.content || "No content available.").slice(0, 300)}...</p>
        </div>`;
    });

  } catch (err) {
    console.error("Request failed:", err);
    loading.style.display = "none";
    result.innerHTML = `<p style="color:red;">Something went wrong. Please try again later.</p>`;
  }
}
