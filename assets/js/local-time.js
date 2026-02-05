document.addEventListener("DOMContentLoaded", function() {
    const timeElements = document.querySelectorAll(".post-meta time, .moment-meta time");

    timeElements.forEach(function(timeElement) {
        const datetime = timeElement.getAttribute("datetime");
        if (datetime) {
            const utcDate = new Date(datetime);
            const localDateString = utcDate.toLocaleString(undefined, {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
                hour12: false
            });
            timeElement.textContent = localDateString;
        }
    });
});
