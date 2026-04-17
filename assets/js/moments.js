document.addEventListener("DOMContentLoaded", () => {
  // --- Expand/collapse toggles ---
  function bindToggles(root) {
    root.querySelectorAll("[data-moment-toggle]").forEach((button) => {
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
        setState(button.getAttribute("aria-expanded") !== "true");
      });
    });
  }
  bindToggles(document);

  // --- Infinite scroll ---
  const feed = document.querySelector(".moments-feed");
  const loader = document.querySelector(".moments-loader");
  if (!feed || !loader) return;

  let nextURL = feed.dataset.nextUrl;
  if (!nextURL) return;

  let done = false;

  function finish() {
    done = true;
    observer.disconnect();
    loader.remove();
  }

  async function loadMore() {
    if (loading || done || !nextURL) return;
    loading = true;
    loader.hidden = false;

    try {
      const resp = await fetch(nextURL);
      if (!resp.ok) throw new Error(resp.status);
      const html = await resp.text();
      const doc = new DOMParser().parseFromString(html, "text/html");
      const remoteFeed = doc.querySelector(".moments-feed");
      if (!remoteFeed) { finish(); return; }

      // Determine next page URL
      nextURL = remoteFeed.dataset.nextUrl || null;

      const lastYear = getLastYear();
      const children = Array.from(remoteFeed.children);

      for (const node of children) {
        // Deduplicate year separators
        if (node.classList.contains("moment-year-separator")) {
          const year = node.querySelector(".moment-year-label").textContent.trim();
          if (year === lastYear) continue;
        }

        // Re-key content IDs to avoid duplicates
        if (node.classList.contains("moment-card")) {
          contentIdCounter++;
          const newId = "moment-content-" + contentIdCounter;
          const content = node.querySelector(".moment-content");
          if (content) content.id = newId;
          const toggle = node.querySelector("[data-moment-toggle]");
          if (toggle) toggle.setAttribute("aria-controls", newId);
        }

        feed.appendChild(node);
      }

      // Bind toggles on the newly added nodes
      bindToggles(feed);
    } catch (e) {
      console.error("Failed to load more moments:", e);
      nextURL = null;
    } finally {
      loading = false;
      if (!nextURL) finish();
    }
  }

  // Use IntersectionObserver to trigger loading
  loader.hidden = false;
  const observer = new IntersectionObserver(
    (entries) => {
      if (!done && entries[0].isIntersecting) loadMore();
    },
    { rootMargin: "400px" }
  );
  observer.observe(loader);
});
