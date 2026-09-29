(function () {
  function syncJsonScripts(doc) {
    const incoming = new Map();
    doc.querySelectorAll('script[type="application/json"][id]').forEach((script) => {
      incoming.set(script.id, script.textContent);
    });

    document.querySelectorAll('script[type="application/json"][id]').forEach((current) => {
      if (incoming.has(current.id)) {
        current.textContent = incoming.get(current.id);
      } else {
        current.remove();
      }
    });

    incoming.forEach((text, id) => {
      if (document.getElementById(id)) return;
      const script = document.createElement("script");
      script.id = id;
      script.type = "application/json";
      script.textContent = text;
      document.body.appendChild(script);
    });
  }

  function applyPageHtml(html) {
    const doc = new DOMParser().parseFromString(html, "text/html");

    const currentBody = document.querySelector(".dashboard-body");
    const nextBody = doc.querySelector(".dashboard-body");
    if (currentBody && nextBody) {
      currentBody.innerHTML = nextBody.innerHTML;
    }

    const currentMessages = document.getElementById("dashboard-upload-messages");
    const nextMessages = doc.getElementById("dashboard-upload-messages");
    if (currentMessages && nextMessages) {
      currentMessages.innerHTML = nextMessages.innerHTML;
    }

    syncJsonScripts(doc);
  }

  function updatePolicyToast(meta) {
    let container = document.getElementById("policy-toast-container");
    if (!container) {
      container = document.createElement("div");
      container.id = "policy-toast-container";
      const content = document.querySelector(".dashboard-content");
      content?.insertBefore(container, content.firstChild);
    }

    if (!meta.show_policy_toast) {
      container.innerHTML = "";
      return;
    }

    container.innerHTML = `
      <div class="policy-toast" role="status" aria-live="polite">
        <span class="policy-toast__message">
          Solución seleccionada no cumple políticas de grosor único entre 3, 4.5 mm y 5
        </span>
        <form action="/policy-toast/dismiss" method="post">
          <button type="submit" class="policy-toast__close" aria-label="Cerrar advertencia">×</button>
        </form>
      </div>
    `;
  }

  window.dashboardCharts = window.dashboardCharts || [];
  window.dashboardDestroyCharts =
    window.dashboardDestroyCharts ||
    function dashboardDestroyCharts() {
      window.dashboardCharts.forEach((chart) => chart?.destroy?.());
      window.dashboardCharts = [];
      if (window.dashboardMap?.remove) {
        window.dashboardMap.remove();
        window.dashboardMap = null;
      }
      const mapNode = document.getElementById("americas-map");
      if (mapNode) mapNode.innerHTML = "";
    };

  async function refreshPageContent() {
    const response = await fetch(window.location.pathname, {
      credentials: "same-origin",
      headers: { Accept: "text/html", "Cache-Control": "no-cache" },
    });
    if (!response.ok) {
      throw new Error("No se pudo actualizar la página");
    }

    applyPageHtml(await response.text());
    window.DashboardLoader?.revealAllStatic();
    window.dashboardDestroyCharts();

    if (typeof window.dashboardPageInit === "function") {
      await window.dashboardPageInit();
    }

    window.dispatchEvent(new CustomEvent("dashboard:content-updated"));
    if (window.feather) window.feather.replace();
  }

  async function selectSolution(name) {
    const select = document.getElementById("solution-select");
    if (!name) return;

    window.DashboardLoader?.prepareReload();

    const response = await fetch("/api/solutions/select", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ solution_name: name }),
    });

    const meta = await response.json();
    if (!response.ok || !meta.ok) {
      if (select) select.value = select.dataset.selected || "";
      throw new Error(meta.error || "No se pudo cambiar la solución");
    }

    if (select) {
      select.value = name;
      select.dataset.selected = name;
    }

    updatePolicyToast(meta);
    await refreshPageContent();
  }

  function bindSolutionSelect() {
    const select = document.getElementById("solution-select");
    if (!select || select.dataset.ajaxBound === "1") return;
    select.dataset.ajaxBound = "1";

    select.addEventListener("change", async () => {
      if (select.value === "__upload__") {
        select.value = select.dataset.selected || "";
        window.dispatchEvent(new CustomEvent("dashboard:open-upload"));
        return;
      }

      if (!select.value) return;

      const previous = select.dataset.selected || "";
      if (select.value === previous) return;

      select.disabled = true;
      try {
        await selectSolution(select.value);
      } catch (error) {
        select.value = previous;
        console.error(error);
      } finally {
        select.disabled = false;
      }
    });
  }

  window.dashboardSelectSolution = selectSolution;
  window.dashboardRefreshContent = refreshPageContent;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindSolutionSelect);
  } else {
    bindSolutionSelect();
  }
})();
