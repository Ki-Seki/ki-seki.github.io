// Convert all time elements to user's local timezone
document.addEventListener("DOMContentLoaded", () => {
  const timeElements = document.querySelectorAll("time[datetime]");

  timeElements.forEach((timeEl) => {
    const isoString = timeEl.getAttribute("datetime");
    if (!isoString) return;

    try {
      const date = new Date(isoString);
      
      // Check if the element is in a list view (moments feed) or detailed view
      // List views typically show just date, detailed views show date + time
      const isListView = timeEl.closest(".moment-card") !== null;
      
      // Check if this is an "Updated" time (contains "Updated:" text)
      const originalText = timeEl.textContent;
      const isUpdatedTime = originalText.includes("Updated:");
      
      let localDateString;
      
      if (isListView) {
        // For list view, show date only
        const options = {
          year: "numeric",
          month: "short",
          day: "numeric"
        };
        localDateString = date.toLocaleDateString(undefined, options);
      } else {
        // For detailed view (post pages), show date and time with timezone
        const options = {
          year: "numeric",
          month: "short",
          day: "numeric",
          hour: "2-digit",
          minute: "2-digit",
          timeZoneName: "short"
        };
        localDateString = date.toLocaleString(undefined, options);
      }
      
      // Preserve the "Updated:" prefix if present
      if (isUpdatedTime) {
        timeEl.textContent = "Updated: " + localDateString;
      } else {
        timeEl.textContent = localDateString;
      }
      
      // Update the title attribute to show full local datetime
      const fullOptions = {
        year: "numeric",
        month: "long",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        timeZoneName: "long"
      };
      const fullLocalString = date.toLocaleString(undefined, fullOptions);
      timeEl.setAttribute("title", fullLocalString);
    } catch (e) {
      console.error("Error parsing date:", isoString, e);
    }
  });
});
