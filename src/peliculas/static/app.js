const state = {
  section: "inicio",
  reference: null,
  clusterApproach: "characteristics",
  clusterId: null,
  clusterPage: 1,
  catalogPage: 1,
  catalogSort: "title",
  catalogDirection: "asc",
  searchTimer: null,
  suggestionTimer: null,
  toastTimer: null,
};

const $ = (selector) => document.querySelector(selector);
const numberFormat = new Intl.NumberFormat("es-ES");

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "No se pudo completar la solicitud.");
  return payload;
}

function metadata(movie) {
  return [
    movie.release_year || "Año desconocido",
    movie.duration,
    movie.rating,
  ].filter(Boolean).join(" · ");
}

function details(movie) {
  return [movie.country, movie.listed_in].filter(Boolean).join(" · ");
}

function goTo(section) {
  state.section = section;
  document.querySelectorAll(".page-section").forEach((element) => {
    element.classList.toggle("active", element.id === `section-${section}`);
  });
  document.querySelectorAll(".nav-link").forEach((element) => {
    element.classList.toggle("active", element.dataset.section === section);
  });
  const names = {
    inicio: "Resumen",
    clusters: "Exploración de clusters",
    recommender: "Recomendador",
    catalog: "Catálogo de películas",
  };
  $("#breadcrumb").textContent = names[section];
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (section === "clusters" && $("#cluster-bars").childElementCount === 0) loadClusters();
  if (section === "catalog" && $("#catalog-rows").childElementCount === 0) loadCatalog();
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => toast.classList.remove("visible"), 2600);
}

function renderClusterBars(data) {
  const maximum = Math.max(...data.clusters.map((cluster) => cluster.count), 1);
  const bars = $("#cluster-bars");
  bars.dataset.approach = state.clusterApproach;
  bars.innerHTML = data.clusters.map((cluster) => `
    <button class="cluster-bar ${cluster.id === state.clusterId ? "selected" : ""}" data-cluster="${cluster.id}" type="button">
      <span class="bar-label"><span>Cluster ${cluster.id}</span><small>${cluster.percentage}%</small></span>
      <span class="bar-track"><span class="bar-fill" style="display:block;width:${cluster.count / maximum * 100}%"></span></span>
      <span class="bar-count">${numberFormat.format(cluster.count)} películas</span>
    </button>
  `).join("");
  $("#cluster-count").textContent = numberFormat.format(data.total_clusters);
  $("#cluster-movies").textContent = numberFormat.format(data.clustered_movies);
  $("#cluster-approach-title").textContent = state.clusterApproach === "characteristics"
    ? "Características estructuradas" : "Descripción textual";
  $("#cluster-approach-description").textContent = state.clusterApproach === "characteristics"
    ? "Año, duración, clasificación, país y géneros codificados."
    : "Contenido textual de las descripciones vectorizado con TF-IDF.";
}

async function loadClusters() {
  try {
    const data = await api(`/api/clusters?approach=${state.clusterApproach}`);
    renderClusterBars(data);
    if (state.clusterId === null || !data.clusters.some((item) => item.id === state.clusterId)) {
      state.clusterId = data.clusters[0]?.id ?? null;
    }
    document.querySelectorAll(".cluster-bar").forEach((element) => {
      element.classList.toggle("selected", Number(element.dataset.cluster) === state.clusterId);
    });
    if (state.clusterId !== null) await loadClusterMovies();
  } catch (error) {
    $("#cluster-bars").innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

function renderPagination(container, page, pages, onChange) {
  if (pages <= 1) {
    container.innerHTML = "";
    return;
  }
  container.innerHTML = `
    <button class="page-button" data-page="${page - 1}" ${page <= 1 ? "disabled" : ""}>← Anterior</button>
    <span>Página ${numberFormat.format(page)} de ${numberFormat.format(pages)}</span>
    <button class="page-button" data-page="${page + 1}" ${page >= pages ? "disabled" : ""}>Siguiente →</button>
  `;
  container.querySelectorAll("[data-page]").forEach((button) => {
    button.addEventListener("click", () => onChange(Number(button.dataset.page)));
  });
}

async function loadClusterMovies() {
  const query = new URLSearchParams({
    approach: state.clusterApproach,
    cluster: String(state.clusterId),
    page: String(state.clusterPage),
    page_size: "18",
    q: $("#cluster-search").value.trim(),
  });
  try {
    const data = await api(`/api/cluster-movies?${query}`);
    $("#selected-cluster-title").textContent = `Cluster ${state.clusterId}`;
    $("#selected-cluster-caption").textContent = `${numberFormat.format(data.results_count)} películas · Enfoque ${state.clusterApproach === "characteristics" ? "Characteristics" : "Description"}`;
    $("#cluster-members").innerHTML = data.items.length
      ? data.items.map((movie) => `
        <article class="movie-card">
          <div class="movie-card-title" title="${escapeHtml(movie.title)}">${escapeHtml(movie.title)}</div>
          <div class="movie-card-meta">${escapeHtml(metadata(movie))}</div>
          <div class="movie-card-tags">${escapeHtml(details(movie) || "Sin país o género disponible")}</div>
        </article>
      `).join("")
      : '<div class="empty-state">No hay películas que coincidan con la búsqueda.</div>';
    renderPagination($("#cluster-pagination"), data.page, data.pages, (page) => {
      state.clusterPage = page;
      loadClusterMovies();
    });
  } catch (error) {
    $("#cluster-members").innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

function renderMovieSuggestions(movies) {
  $("#reference-suggestions").innerHTML = movies.map((movie) => `
    <button class="suggestion-option" type="button" role="option" data-show-id="${escapeHtml(movie.show_id)}">
      <strong>${escapeHtml(movie.title)}</strong>
      <small>${escapeHtml([movie.release_year, movie.rating, movie.country].filter(Boolean).join(" · ") || "Datos adicionales no disponibles")}</small>
    </button>
  `).join("");
  $("#reference-suggestions").querySelectorAll("[data-show-id]").forEach((button) => {
    button.addEventListener("click", () => {
      const movie = movies.find((item) => item.show_id === button.dataset.showId);
      selectReference(movie);
    });
  });
}

function selectReference(movie) {
  state.reference = movie || null;
  $("#reference-suggestions").innerHTML = "";
  $("#reference-search").value = movie?.title || "";
  const display = $("#selected-reference");
  if (!movie) {
    display.classList.add("hidden");
    return;
  }
  display.classList.remove("hidden");
  display.innerHTML = `<span><strong>${escapeHtml(movie.title)}</strong><small>${escapeHtml(metadata(movie))}</small></span><button class="clear-reference" type="button" aria-label="Quitar película">×</button>`;
  display.querySelector("button").addEventListener("click", () => {
    state.reference = null;
    $("#reference-search").value = "";
    display.classList.add("hidden");
    $("#reference-search").focus();
  });
}

async function loadRecommendations(event) {
  event.preventDefault();
  $("#recommend-error").classList.add("hidden");
  if (!state.reference) {
    $("#recommend-error").textContent = "Busca y selecciona una película de la lista de sugerencias.";
    $("#recommend-error").classList.remove("hidden");
    return;
  }
  const count = Number($("#recommend-count").value);
  if (!Number.isInteger(count) || count < 1) {
    $("#recommend-error").textContent = "El número de recomendaciones debe ser un entero mayor que 0.";
    $("#recommend-error").classList.remove("hidden");
    return;
  }
  const submit = $("#recommend-form button[type=submit]");
  submit.disabled = true;
  submit.innerHTML = "Buscando…";
  try {
    const result = await api("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        show_id: state.reference.show_id,
        approach: $("#recommend-approach").value,
        count,
      }),
    });
    $("#recommend-results-title").textContent = `Películas similares a ${result.reference.title}`;
    $("#recommend-result-count").textContent = `${result.recommendations.length} resultados`;
    $("#recommend-result-count").classList.remove("hidden");
    $("#recommendations").innerHTML = result.recommendations.length
      ? result.recommendations.map((movie, index) => `
        <article class="recommendation-item">
          <span class="recommendation-rank">${String(index + 1).padStart(2, "0")}</span>
          <span><span class="recommendation-title">${escapeHtml(movie.title)}</span><span class="recommendation-meta">${escapeHtml(metadata(movie))}${details(movie) ? ` · ${escapeHtml(details(movie))}` : ""}</span></span>
          <span class="distance-pill">Distancia ${Number(movie.distance).toFixed(3)}</span>
        </article>
      `).join("")
      : '<div class="empty-panel"><strong>No hay recomendaciones disponibles</strong><small>No hay otras películas elegibles para este enfoque.</small></div>';
    if (result.message) showToast(result.message);
  } catch (error) {
    $("#recommend-error").textContent = error.message;
    $("#recommend-error").classList.remove("hidden");
  } finally {
    submit.disabled = false;
    submit.innerHTML = 'Obtener recomendaciones <span>→</span>';
  }
}

async function loadCatalog() {
  const query = new URLSearchParams({
    q: $("#catalog-search").value.trim(),
    page: String(state.catalogPage),
    page_size: "20",
    sort: state.catalogSort,
    direction: state.catalogDirection,
  });
  try {
    const data = await api(`/api/catalog?${query}`);
    $("#catalog-count").textContent = `${numberFormat.format(data.results_count)} resultados de ${numberFormat.format(data.catalog_total)} títulos`;
    $("#catalog-page-label").textContent = `Página ${data.page} de ${data.pages}`;
    $("#catalog-empty").classList.toggle("hidden", data.items.length > 0);
    $("#catalog-rows").innerHTML = data.items.map((movie) => `
      <tr>
        <td>${escapeHtml(movie.title)}<span class="type-pill">${escapeHtml(movie.type)}</span></td>
        <td>${escapeHtml(movie.release_year ?? "—")}</td>
        <td>${escapeHtml(movie.duration || "—")}</td>
        <td>${escapeHtml(movie.rating || "—")}</td>
        <td title="${escapeHtml(movie.country)}">${escapeHtml(movie.country || "—")}</td>
        <td title="${escapeHtml(movie.listed_in)}">${escapeHtml(movie.listed_in || "—")}</td>
        <td>${movie.type === "Movie" ? `<button class="table-action" type="button" data-reference="${escapeHtml(movie.show_id)}">Usar</button>` : "—"}</td>
      </tr>
    `).join("");
    $("#catalog-rows").querySelectorAll("[data-reference]").forEach((button) => {
      button.addEventListener("click", () => {
        const movie = data.items.find((item) => item.show_id === button.dataset.reference);
        selectReference(movie);
        goTo("recommender");
        showToast("Película seleccionada como referencia.");
      });
    });
    renderPagination($("#catalog-pagination"), data.page, data.pages, (page) => {
      state.catalogPage = page;
      loadCatalog();
    });
  } catch (error) {
    $("#catalog-rows").innerHTML = `<tr><td colspan="7">${escapeHtml(error.message)}</td></tr>`;
  }
}

async function loadSummary() {
  try {
    const data = await api("/api/summary");
    $("#metric-catalog").textContent = numberFormat.format(data.catalog_total);
    $("#metric-movies").textContent = numberFormat.format(data.movie_total);
    $("#metric-char-clusters").textContent = numberFormat.format(data.cluster_totals.characteristics);
    $("#metric-desc-clusters").textContent = numberFormat.format(data.cluster_totals.description);
    $("#metric-char-total").textContent = `${numberFormat.format(data.clustered.characteristics)} películas agrupadas`;
    $("#metric-desc-total").textContent = `${numberFormat.format(data.clustered.description)} películas agrupadas`;
  } catch (error) {
    showToast(error.message);
  }
}

document.querySelectorAll(".nav-link").forEach((button) => {
  button.addEventListener("click", () => goTo(button.dataset.section));
});
document.querySelectorAll("[data-go]").forEach((button) => {
  button.addEventListener("click", () => goTo(button.dataset.go));
});
$("#cluster-approach").addEventListener("change", (event) => {
  state.clusterApproach = event.target.value;
  state.clusterId = null;
  state.clusterPage = 1;
  loadClusters();
});
$("#cluster-bars").addEventListener("click", (event) => {
  const button = event.target.closest("[data-cluster]");
  if (!button) return;
  state.clusterId = Number(button.dataset.cluster);
  state.clusterPage = 1;
  document.querySelectorAll(".cluster-bar").forEach((item) => {
    item.classList.toggle("selected", item === button);
  });
  loadClusterMovies();
});
$("#cluster-search").addEventListener("input", () => {
  state.clusterPage = 1;
  clearTimeout(state.searchTimer);
  state.searchTimer = setTimeout(loadClusterMovies, 250);
});
$("#recommend-form").addEventListener("submit", loadRecommendations);
$("#reference-search").addEventListener("input", (event) => {
  state.reference = null;
  $("#selected-reference").classList.add("hidden");
  const query = event.target.value.trim();
  clearTimeout(state.suggestionTimer);
  if (query.length < 2) {
    $("#reference-suggestions").innerHTML = "";
    return;
  }
  state.suggestionTimer = setTimeout(async () => {
    try {
      const movies = await api(`/api/search?q=${encodeURIComponent(query)}&limit=12`);
      renderMovieSuggestions(movies);
      if (!movies.length) {
        $("#reference-suggestions").innerHTML = '<span class="suggestion-option"><small>No se encontraron películas con ese título.</small></span>';
      }
    } catch (error) {
      $("#reference-suggestions").innerHTML = `<span class="suggestion-option"><small>${escapeHtml(error.message)}</small></span>`;
    }
  }, 220);
});
document.addEventListener("click", (event) => {
  if (!event.target.closest(".reference-field")) $("#reference-suggestions").innerHTML = "";
});
$("#catalog-search").addEventListener("input", () => {
  state.catalogPage = 1;
  clearTimeout(state.searchTimer);
  state.searchTimer = setTimeout(loadCatalog, 250);
});
$("#catalog-sort").addEventListener("change", (event) => {
  state.catalogSort = event.target.value;
  state.catalogPage = 1;
  loadCatalog();
});
$("#sort-direction").addEventListener("click", () => {
  state.catalogDirection = state.catalogDirection === "asc" ? "desc" : "asc";
  $("#sort-direction").textContent = state.catalogDirection === "asc" ? "↑ A–Z" : "↓ Z–A";
  state.catalogPage = 1;
  loadCatalog();
});

loadSummary();
