(function () {
  const productsNode = document.getElementById("packaging-products-data");
  const palletNode = document.getElementById("packaging-pallet-data");
  const backdrop = document.getElementById("product-drawer-backdrop");
  const drawer = document.getElementById("product-drawer");
  const closeButton = document.getElementById("close-product-drawer");
  const tableBody = document.getElementById("products-table-body");
  const vizCanvas = document.getElementById("pallet-viz-canvas");
  const emptyMessage = document.getElementById("drawer-empty-message");
  const packagingContent = document.getElementById("drawer-packaging-content");

  if (!backdrop || !drawer || !vizCanvas) {
    return;
  }

  const products = productsNode ? JSON.parse(productsNode.textContent || "[]") : [];
  const pallet = palletNode ? JSON.parse(palletNode.textContent || "{}") : {};
  const productsById = Object.fromEntries(products.map((item) => [item.codigo_producto, item]));
  let activeRow = null;
  let activeProductId = null;

  const fields = {
    code: document.getElementById("drawer-product-code"),
    box: document.getElementById("drawer-product-box"),
    ocupacionCaja: document.getElementById("drawer-ocupacion-caja"),
    utilPallet: document.getElementById("drawer-util-pallet"),
    cajasPallet: document.getElementById("drawer-cajas-pallet"),
    apilado: document.getElementById("drawer-apilado"),
    productDims: document.getElementById("drawer-product-dims"),
    boxDims: document.getElementById("drawer-box-dims"),
    palletCaption: document.getElementById("drawer-pallet-caption"),
    palletLegend: document.getElementById("drawer-pallet-legend"),
  };

  const formatDims = (largo, ancho, alto) =>
    `${Math.round(largo)} × ${Math.round(ancho)} × ${Math.round(alto)} mm`;

  function projectIso(x, y, z, scale, originX, originY) {
    return {
      x: originX + (x - z) * 0.8660254 * scale,
      y: originY - y * scale - (x + z) * 0.5 * scale,
    };
  }

  function drawFace(ctx, points, fill, stroke = "rgba(38, 38, 38, 0.16)") {
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i += 1) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  function drawIsoBox(ctx, x, y, z, w, h, d, scale, originX, originY, colors) {
    const p = (px, py, pz) => projectIso(px, py, pz, scale, originX, originY);

    const A = p(x, y, z);
    const B = p(x + w, y, z);
    const D = p(x, y, z + d);
    const E = p(x, y + h, z);
    const F = p(x + w, y + h, z);
    const G = p(x + w, y + h, z + d);
    const H = p(x, y + h, z + d);

    drawFace(ctx, [A, D, H, E], colors.left);
    drawFace(ctx, [A, B, F, E], colors.right);
    drawFace(ctx, [E, F, G, H], colors.top);
  }

  function clearCanvas() {
    const rect = vizCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    vizCanvas.width = Math.max(1, Math.floor(rect.width * dpr));
    vizCanvas.height = Math.max(1, Math.floor(rect.height * dpr));
    const ctx = vizCanvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
  }

  function renderPalletVisualization(product) {
    const canvas = vizCanvas;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const boxW = product.caja_ancho;
    const boxD = product.caja_largo;
    const boxH = product.caja_alto;
    const nx = product.cajas_piso_ancho;
    const ny = product.cajas_piso_largo;
    const nz = product.capas_alto;

    const loadH = nz * boxH;
    const spanGround = pallet.largo_mm + pallet.ancho_mm;
    const maxY = Math.max(loadH, pallet.alto_mm);

    const scale = Math.min(
      (rect.width * 0.86) / (spanGround * 0.8660254),
      (rect.height * 0.82) / (maxY + spanGround * 0.5)
    );

    const originX =
      rect.width * 0.5 - ((pallet.largo_mm - pallet.ancho_mm) / 2) * 0.8660254 * scale;
    const originY = rect.height * 0.9;

    const p = (px, py, pz) => projectIso(px, py, pz, scale, originX, originY);

    const palletBaseH = 18;
    const palletA = p(0, 0, 0);
    const palletB = p(pallet.largo_mm, 0, 0);
    const palletC = p(pallet.largo_mm, 0, pallet.ancho_mm);
    const palletD = p(0, 0, pallet.ancho_mm);
    const palletE = p(0, -palletBaseH, 0);
    const palletF = p(pallet.largo_mm, -palletBaseH, 0);
    const palletG = p(pallet.largo_mm, -palletBaseH, pallet.ancho_mm);
    const palletHpt = p(0, -palletBaseH, pallet.ancho_mm);

    drawFace(ctx, [palletE, palletF, palletG, palletHpt], "#525252");
    drawFace(ctx, [palletA, palletB, palletC, palletD], "#3f3f46", "rgba(23, 23, 23, 0.35)");

    const boundsA = p(0, 0, 0);
    const boundsB = p(pallet.largo_mm, 0, 0);
    const boundsC = p(pallet.largo_mm, 0, pallet.ancho_mm);
    const boundsD = p(0, 0, pallet.ancho_mm);
    const boundsE = p(0, pallet.alto_mm, 0);
    const boundsF = p(pallet.largo_mm, pallet.alto_mm, 0);
    const boundsG = p(pallet.largo_mm, pallet.alto_mm, pallet.ancho_mm);
    const boundsH = p(0, pallet.alto_mm, pallet.ancho_mm);

    ctx.setLineDash([4, 4]);
    drawFace(ctx, [boundsA, boundsB, boundsC, boundsD], "rgba(255,255,255,0)", "rgba(163, 163, 163, 0.55)");
    drawFace(ctx, [boundsE, boundsF, boundsG, boundsH], "rgba(255,255,255,0)", "rgba(163, 163, 163, 0.35)");
    ctx.setLineDash([]);

    const cells = [];
    for (let iz = 0; iz < nz; iz += 1) {
      for (let iy = 0; iy < ny; iy += 1) {
        for (let ix = 0; ix < nx; ix += 1) {
          const isVisible = ix === 0 || iy === 0 || iz === nz - 1;
          if (isVisible) {
            cells.push({ ix, iy, iz });
          }
        }
      }
    }

    cells.sort((a, b) => {
      const depthA = a.iz * boxH + 0.5 * (a.ix * boxW + a.iy * boxD);
      const depthB = b.iz * boxH + 0.5 * (b.ix * boxW + b.iy * boxD);
      return depthB - depthA;
    });

    const colors = { top: "#B8D892", right: "#78A844", left: "#5E8434" };

    cells.forEach(({ ix, iy, iz }) => {
      drawIsoBox(
        ctx,
        ix * boxW,
        iz * boxH,
        iy * boxD,
        boxW,
        boxH,
        boxD,
        scale,
        originX,
        originY,
        colors
      );
    });

    const usedHeight = product.capas_alto * product.caja_alto;
    fields.palletCaption.textContent =
      `Base ${pallet.largo_mm} × ${pallet.ancho_mm} mm · Altura max ${pallet.alto_mm} mm`;
    fields.palletLegend.textContent =
      `${product.cajas_por_pallet} cajas (${nx}×${ny}×${nz}) · ` +
      `altura usada ${Math.round(usedHeight)} mm · util. teorica ${product.utilizacion_pallet_teorica_pct.toFixed(1)}%`;
  }

  function setEmptyState(productId) {
    fields.code.textContent = productId;
    fields.box.textContent = "Sin packaging en la solución activa";
    fields.ocupacionCaja.textContent = "-";
    fields.utilPallet.textContent = "-";
    fields.cajasPallet.textContent = "-";
    fields.apilado.textContent = "-";
    fields.productDims.textContent = "-";
    fields.boxDims.textContent = "-";
    fields.palletCaption.textContent = "1200 × 800 × 1800 mm";
    fields.palletLegend.textContent = "";
    emptyMessage.classList.remove("hidden");
    packagingContent.classList.add("hidden");
    clearCanvas();
  }

  function setPackagingState(product) {
    emptyMessage.classList.add("hidden");
    packagingContent.classList.remove("hidden");

    fields.code.textContent = product.codigo_producto;
    fields.box.textContent = product.caja_label;
    fields.ocupacionCaja.textContent = `${product.ocupacion_caja_pct.toFixed(1)}%`;
    fields.utilPallet.textContent = `${product.utilizacion_pallet_teorica_pct.toFixed(1)}%`;
    fields.cajasPallet.textContent = String(product.cajas_por_pallet);
    fields.apilado.textContent =
      `${product.cajas_piso_ancho} × ${product.cajas_piso_largo} × ${product.capas_alto}`;
    fields.productDims.textContent = formatDims(
      product.producto_largo,
      product.producto_ancho,
      product.producto_alto
    );
    fields.boxDims.textContent = formatDims(product.caja_largo, product.caja_ancho, product.caja_alto);

    renderPalletVisualization(product);
  }

  function openDrawer(productId) {
    activeProductId = productId;

    if (activeRow) activeRow.classList.remove("is-active");
    activeRow = tableBody
      ? tableBody.querySelector(`[data-product-id="${productId}"]`)
      : null;
    if (activeRow) activeRow.classList.add("is-active");

    const product = productsById[productId];
    if (!product) {
      setEmptyState(productId);
    } else {
      setPackagingState(product);
    }

    backdrop.classList.remove("hidden");
    requestAnimationFrame(() => {
      backdrop.classList.add("is-open");
      drawer.classList.add("is-open");
    });
    drawer.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
  }

  function closeDrawer() {
    backdrop.classList.remove("is-open");
    drawer.classList.remove("is-open");
    drawer.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
    activeProductId = null;

    if (activeRow) {
      activeRow.classList.remove("is-active");
      activeRow = null;
    }

    window.setTimeout(() => {
      if (!drawer.classList.contains("is-open")) {
        backdrop.classList.add("hidden");
      }
    }, 220);
  }

  if (tableBody) {
    tableBody.addEventListener("click", (event) => {
      const row = event.target.closest(".product-row");
      if (!row) return;
      openDrawer(row.dataset.productId);
    });
  }

  closeButton.addEventListener("click", closeDrawer);
  backdrop.addEventListener("click", closeDrawer);

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && drawer.classList.contains("is-open")) {
      closeDrawer();
    }
  });

  window.addEventListener("resize", () => {
    if (!drawer.classList.contains("is-open") || !activeProductId) return;
    const product = productsById[activeProductId];
    if (product) {
      renderPalletVisualization(product);
    }
  });

  window.openProductDrawer = openDrawer;
})();
