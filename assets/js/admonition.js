/**
 * Admonition details interaction
 * Copied from LoveIt theme
 */
document.addEventListener('DOMContentLoaded', function() {
    // Initialize admonition details
    function initDetails() {
        var detailsElements = document.getElementsByClassName('details');
        for (var i = 0; i < detailsElements.length; i++) {
            var details = detailsElements[i];
            var summary = details.getElementsByClassName('details-summary')[0];
            if (summary) {
                summary.addEventListener('click', function() {
                    this.parentElement.classList.toggle('open');
                }, false);
            }
        }
    }

    // Initialize on DOM ready
    initDetails();
});
