(function () {
  function revealHost(host) {
    if (!host) return;
    host.classList.remove("is-loading");
    host.classList.add("is-ready");
    const item = host.querySelector(".skel-item");
    if (item) {
      window.setTimeout(() => {
        item.remove();
      }, 220);
    }
  }

  function revealHostByName(name) {
    revealHost(document.querySelector(`[data-skel-host="${name}"]`));
  }

  function revealAllStatic() {
    document
      .querySelectorAll("[data-skel-host]:not([data-skel-chart])")
      .forEach((host) => revealHost(host));
  }

  function prepareReload() {
    document.querySelectorAll("[data-skel-host][data-skel-chart]").forEach((host) => {
      host.classList.add("is-loading");
      host.classList.remove("is-ready");
      if (!host.querySelector(".skel-item")) {
        const type = host.dataset.skelType || "block";
        const item = document.createElement("div");
        item.className = `skel-item skel-${type}`;
        item.setAttribute("aria-hidden", "true");
        host.insertBefore(item, host.firstChild);
      }
    });
  }

  function init() {
    revealAllStatic();
  }

  window.DashboardLoader = {
    revealHost: revealHostByName,
    revealAllStatic,
    prepareReload,
    init,
  };

  window.dashboardRevealHost = revealHostByName;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
