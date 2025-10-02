document.addEventListener("DOMContentLoaded", () => {
  const toggles = document.querySelectorAll('[data-moment-toggle]');

  toggles.forEach((button) => {
    const targetId = button.getAttribute("aria-controls");
    if (!targetId) return;

    const content = document.getElementById(targetId);
    if (!content) {
      button.hidden = true;
      return;
    }

    const expandLabel = button.dataset.expandLabel || "Read full note";
    const collapseLabel = button.dataset.collapseLabel || "Hide note";

    const setState = (expanded) => {
      button.setAttribute("aria-expanded", String(expanded));
      button.textContent = expanded ? collapseLabel : expandLabel;
      content.classList.toggle("is-expanded", expanded);
      content.classList.toggle("is-collapsed", !expanded);
      content.setAttribute("data-collapsed", expanded ? "false" : "true");
    };

    button.addEventListener("click", () => {
      const isExpanded = button.getAttribute("aria-expanded") === "true";
      setState(!isExpanded);
    });
  });
});
