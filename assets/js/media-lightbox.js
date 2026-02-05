(function () {
  const initLightbox = () => {
    if (typeof GLightbox === "undefined") {
      return;
    }

    if (window.__siteMediaLightbox) {
      window.__siteMediaLightbox.destroy();
    }

    window.__siteMediaLightbox = GLightbox({
      selector: ".glightbox",
    });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initLightbox);
  } else {
    initLightbox();
  }

})();
