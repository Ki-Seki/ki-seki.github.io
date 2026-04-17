document.addEventListener("DOMContentLoaded", () => {
  /* ── Read-more / collapse toggles ── */
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

  /* ── Infinite scroll / lazy-load ── */
  const feed = document.querySelector(".moments-feed");
  const sentinel = document.querySelector(".moments-sentinel");
  if (!feed || !sentinel) return;

  const batchSize = parseInt(feed.dataset.batchSize, 10) || 5;
  const hiddenItems = Array.from(feed.querySelectorAll(".moment-lazy"));

  if (hiddenItems.length === 0) {
    sentinel.classList.add("is-done");
    return;
  }

  const revealBatch = () => {
    const batch = hiddenItems.splice(0, batchSize);
    if (batch.length === 0) return;

    batch.forEach((el) => {
      el.classList.remove("moment-lazy");
    });

    if (hiddenItems.length === 0) {
      sentinel.classList.add("is-done");
      observer.disconnect();
    }
  };

  const observer = new IntersectionObserver(
    (entries) => {
      if (entries[0].isIntersecting) {
        revealBatch();
      }
    },
    { rootMargin: "200px" }
  );

  observer.observe(sentinel);
});
