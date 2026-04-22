const state = {
  sessionId: null,
  discovered: [],
  preview: null,
  targetScan: null,
  selectedClass: null,
  classDetail: null,
  classDetailSplit: null,
  detailSelectedImages: [],
  detailActionClass: "",
  detailActionSplit: "train",
  busy: false,
  currentPage: "merge",
  browserFilter: "all",
  browserSort: "size_desc",
  mergeFilter: "all",
  starredClasses: [],
  exportModelPath: "/workspace/anand/models/recognition_models/checkpoints/dinov3b_dessert_v1/best_dinov3b_dessert_v1_classifier.pth",
  exportOutputFilename: "best_dinov3b_dessert_v1_classifier_gallery_20.npz",
  exportOutputFilenameAutoValue: "best_dinov3b_dessert_v1_classifier_gallery_20.npz",
  exportSelectedClass: null,
  exportClassDetail: null,
  exportSearch: "",
  exportFilter: "all",
  exportSort: "size_desc",
  exportDetailMode: "all",
  exportPerClassLimit: 20,
  exportSelections: {},
  exportResult: null,
  exportStatus: {
    tone: "idle",
    title: "No export activity yet",
    message: "Save a selection snapshot or run an export to see progress and results here.",
    detail: "",
  },
};

const SESSION_SAVE_DEBOUNCE_MS = 300;
let sessionSaveTimer = null;
let sessionBootstrapped = false;

function el(id) {
  return document.getElementById(id);
}

function setStatus(text) {
  el("status-pill").textContent = text;
}

function targetPathValue() {
  return el("target-path").value.trim();
}

function sourcePathValue() {
  return el("source-path").value.trim();
}

function exportModelPathValue() {
  return el("export-model-path").value.trim();
}

function exportOutputFilenameValue() {
  return el("export-output-filename")?.value.trim() || "";
}

function defaultExportFilename(modelPath = exportModelPathValue(), perClassLimit = state.exportPerClassLimit) {
  const modelName = (modelPath.split("/").pop() || "gallery_model")
    .replace(/\.[^.]+$/, "");
  return `${modelName}_gallery_${perClassLimit}.npz`;
}

function syncExportOutputFilenameAutoValue(options = {}) {
  const input = el("export-output-filename");
  const nextAutoValue = defaultExportFilename();
  const shouldForce = Boolean(options.force);
  const currentValue = input?.value.trim() || state.exportOutputFilename || "";
  const shouldFollowAuto =
    shouldForce ||
    !currentValue ||
    currentValue === state.exportOutputFilenameAutoValue;

  state.exportOutputFilenameAutoValue = nextAutoValue;
  if (shouldFollowAuto) {
    state.exportOutputFilename = nextAutoValue;
    if (input) {
      input.value = nextAutoValue;
    }
  } else if (input) {
    state.exportOutputFilename = input.value.trim();
  }
}

function inferredExportOutputPath() {
  const targetPath = targetPathValue();
  if (!targetPath) {
    return "";
  }
  const targetBase = targetPath.replace(/\/+$/, "");
  const outputFilename = exportOutputFilenameValue() || state.exportOutputFilenameAutoValue || defaultExportFilename();
  return `${targetBase}/db/${outputFilename}`;
}

function setBusy(isBusy, text = "Working") {
  state.busy = isBusy;
  setStatus(text);
  const mergeReady = hasMergeMappings();
  el("btn-refresh-discovery").disabled = isBusy;
  el("btn-scan-target").disabled = isBusy;
  el("btn-preview-merge").disabled = isBusy;
  el("btn-commit-merge").disabled = isBusy || !mergeReady;
  el("btn-default-mappings").disabled = isBusy || !mergeReady;
  el("btn-open-browser").disabled = isBusy || !state.targetScan;
  if (el("btn-open-export")) {
    el("btn-open-export").disabled = isBusy || !state.targetScan;
  }
  el("nav-browse").disabled = isBusy || !state.targetScan;
  if (el("nav-export")) {
    el("nav-export").disabled = isBusy || !state.targetScan;
  }
  if (el("btn-export-npz")) {
    el("btn-export-npz").disabled = isBusy || !state.targetScan;
  }
  if (el("btn-export-save-selection")) {
    el("btn-export-save-selection").disabled = isBusy || !state.targetScan;
  }
  if (el("merge-filter-all")) {
    renderMergeFilters(state.preview);
  }
  if (el("filter-all-classes")) {
    renderBrowseFilters(state.targetScan);
  }
  if (el("export-filter-all-classes")) {
    renderExportFilters(state.targetScan);
  }
  if (el("export-output-copy")) {
    renderExportExperience();
  }
}

function stripMergeHistoryEntries(entries) {
  return (entries || []).map((entry) => ({
    merge_id: entry.merge_id,
    created_at: entry.created_at,
    source_path: entry.source_path,
    target_path: entry.target_path,
    source_tag: entry.source_tag,
    copied_count: entry.copied_count,
    skipped_existing: entry.skipped_existing,
    skipped_disabled: entry.skipped_disabled,
    per_target_counts: entry.per_target_counts,
  }));
}

function compactTargetScanForStorage(scan) {
  if (!scan) {
    return null;
  }

  return {
    ...scan,
    merge_history: stripMergeHistoryEntries(scan.merge_history),
  };
}

async function loadCurrentSession() {
  const payload = await api("/api/session/current");
  const session = payload.session || {};
  state.sessionId = session.session_id || null;

  if (el("target-path")) {
    el("target-path").value = session.target_path || "";
  }
  if (el("source-path")) {
    el("source-path").value = session.source_path || "";
  }

  state.preview = session.preview || null;
  state.targetScan = session.target_scan || null;
  state.selectedClass = session.selected_class || null;
  state.classDetailSplit = session.class_detail_split || null;
  state.currentPage = session.current_page || "merge";
  state.browserFilter = session.browser_filter || "all";
  state.browserSort = session.browser_sort || "size_desc";
  state.mergeFilter = session.merge_filter || "all";
  state.starredClasses = session.starred_classes || [];
  state.exportModelPath = session.export_model_path || state.exportModelPath;
  state.exportOutputFilename = session.export_output_filename || state.exportOutputFilename;
  state.exportSelectedClass = session.export_selected_class || null;
  state.exportSearch = session.export_search || "";
  state.exportFilter = session.export_filter || "all";
  state.exportSort = session.export_sort || "size_desc";
  state.exportDetailMode = session.export_detail_mode || "all";
  state.exportPerClassLimit = session.export_per_class_limit || 20;
  state.exportSelections = session.export_selections || {};
  state.exportStatus = session.export_status || state.exportStatus;
  state.exportResult = session.export_result || null;
  if (el("export-model-path")) {
    el("export-model-path").value = state.exportModelPath;
  }
  if (el("export-output-filename")) {
    el("export-output-filename").value = state.exportOutputFilename;
  }
  syncExportOutputFilenameAutoValue({ force: !session.export_output_filename });
  el("class-search").value = session.class_search || "";
  if (el("export-class-search")) {
    el("export-class-search").value = state.exportSearch || "";
  }
}

async function persistSessionState() {
  const payload = {
    target_path: targetPathValue(),
    source_path: sourcePathValue(),
    preview: state.preview,
    target_scan: compactTargetScanForStorage(state.targetScan),
    selected_class: state.selectedClass,
    class_detail_split: state.classDetailSplit,
    current_page: state.currentPage,
    browser_filter: state.browserFilter,
    browser_sort: state.browserSort,
    merge_filter: state.mergeFilter,
    class_search: el("class-search").value.trim(),
    starred_classes: state.starredClasses,
    export_model_path: exportModelPathValue(),
    export_output_filename: exportOutputFilenameValue(),
    export_selected_class: state.exportSelectedClass,
    export_search: el("export-class-search") ? el("export-class-search").value.trim() : "",
    export_filter: state.exportFilter,
    export_sort: state.exportSort,
    export_detail_mode: state.exportDetailMode,
    export_per_class_limit: state.exportPerClassLimit,
    export_selections: state.exportSelections,
    export_status: state.exportStatus,
    export_result: state.exportResult,
  };

  try {
    const response = await api("/api/session/current/state", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.sessionId = response.session_id || state.sessionId;
  } catch (error) {
    console.warn("Unable to persist dataset_studio session state.", error);
  }
}

function saveDraftState(options = {}) {
  if (!sessionBootstrapped) {
    return;
  }

  const immediate = Boolean(options.immediate);
  if (sessionSaveTimer) {
    clearTimeout(sessionSaveTimer);
    sessionSaveTimer = null;
  }

  if (immediate) {
    void persistSessionState();
    return;
  }

  sessionSaveTimer = window.setTimeout(() => {
    sessionSaveTimer = null;
    void persistSessionState();
  }, SESSION_SAVE_DEBOUNCE_MS);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch (error) {
      // Ignore parse errors and fall back to status text.
    }
    throw new Error(detail);
  }

  return response.json();
}

function formatNumber(value) {
  return new Intl.NumberFormat().format(value || 0);
}

function splitSummary(counts) {
  const train = counts.train || 0;
  const val = counts.val || 0;
  const test = counts.test || 0;
  return `train ${formatNumber(train)} / val ${formatNumber(val)} / test ${formatNumber(test)}`;
}

function totalImages(scan) {
  return scan?.total_images || 0;
}

function previewStats(preview) {
  const mappings = preview?.mappings || [];
  const enabledMappings = mappings.filter((item) => item.enabled !== false);
  const incomingImages = enabledMappings.reduce((sum, item) => sum + (item.source_total || 0), 0);
  const merges = enabledMappings.filter((item) => item.action === "merge").length;
  const creates = enabledMappings.filter((item) => item.action === "create").length;
  const placeholders = enabledMappings.filter((item) => item.action === "create_placeholder").length;

  return {
    mappings: mappings.length,
    enabledMappings: enabledMappings.length,
    incomingImages,
    merges,
    creates,
    placeholders,
  };
}

function summaryCard({ label, value, meta, tone = "" }) {
  return `
    <div class="summary-card ${tone}">
      <p class="summary-kicker">${label}</p>
      <strong>${value}</strong>
      <p class="summary-foot">${meta}</p>
    </div>
  `;
}

function renderSummaryCards(source, target, preview = state.preview) {
  const wrap = el("merge-summary");
  const cards = [];
  const stats = previewStats(preview);
  const isTargetOnly = preview?.mode === "target_only";

  if (target) {
    cards.push(
      summaryCard({
        label: "Target Dataset",
        value: `${formatNumber(target.class_count || 0)} classes`,
        meta: `${formatNumber(totalImages(target))} images · ${splitSummary(target.splits || {})}`,
      })
    );
  }

  if (source) {
    cards.push(
      summaryCard({
        label: "Source Export",
        value: `${formatNumber(source.class_count || 0)} classes`,
        meta: `${formatNumber(totalImages(source))} images · ${splitSummary(source.splits || {})}`,
      })
    );
  }

  if (isTargetOnly) {
    cards.push(
      summaryCard({
        label: "Mode",
        value: "Target Only",
        meta: "No merge source selected. Browse, curate, and export this dataset directly.",
        tone: "highlight",
      })
    );
  }

  if (source) {
    cards.push(
      summaryCard({
        label: "Net Growth",
        value: `+${formatNumber(stats.incomingImages)}`,
        meta: `${formatNumber(stats.enabledMappings)} mapped classes ready to copy`,
        tone: "highlight",
      })
    );
  }

  if (preview && source) {
    cards.push(
      summaryCard({
        label: "Overlap",
        value: `${formatNumber(stats.merges)} merges`,
        meta: `${formatNumber(stats.creates)} new classes · ${formatNumber(stats.placeholders)} placeholders`,
        tone: stats.placeholders ? "warning" : "",
      })
    );
  }

  if (!cards.length) {
    wrap.innerHTML = `<div class="empty-state">Scan a dataset or build a preview to see merge stats.</div>`;
    return;
  }

  wrap.innerHTML = cards.join("");
}

function renderPageNav() {
  const browseReady = Boolean(state.targetScan);
  const exportReady = Boolean(state.targetScan);
  el("nav-merge").classList.toggle("is-active", state.currentPage === "merge");
  el("nav-browse").classList.toggle("is-active", state.currentPage === "browse");
  el("nav-export").classList.toggle("is-active", state.currentPage === "export");
  el("nav-browse").disabled = state.busy || !browseReady;
  el("nav-export").disabled = state.busy || !exportReady;
  el("btn-open-browser").disabled = state.busy || !browseReady;
  if (el("btn-open-export")) {
    el("btn-open-export").disabled = state.busy || !exportReady;
  }
  el("page-merge").classList.toggle("hidden", state.currentPage !== "merge");
  el("page-browse").classList.toggle("hidden", state.currentPage !== "browse");
  el("page-export").classList.toggle("hidden", state.currentPage !== "export");
}

function switchPage(page) {
  if ((page === "browse" || page === "export") && !state.targetScan) {
    return;
  }
  state.currentPage = page;
  renderPageNav();
  saveDraftState();
}

function classByName(className) {
  return state.targetScan?.classes?.find((item) => item.name === className) || null;
}

function hasMergeMappings(preview = state.preview) {
  return Boolean(preview && Array.isArray(preview.mappings) && preview.mappings.length > 0);
}

function exportableClasses(dataset = state.targetScan) {
  return (dataset?.classes || []).filter((item) => (item.counts?.train || 0) > 0);
}

function normalizedStarredClasses() {
  return Array.from(new Set((state.starredClasses || []).filter(Boolean))).sort((left, right) => left.localeCompare(right));
}

function starredClassNames(dataset = state.targetScan) {
  const validNames = new Set((dataset?.classes || []).map((item) => item.name));
  return new Set(normalizedStarredClasses().filter((name) => validNames.has(name)));
}

function isClassStarred(className) {
  return starredClassNames().has(className);
}

function toggleStarredClass(className) {
  if (!className) {
    return;
  }
  const starred = new Set(normalizedStarredClasses());
  if (starred.has(className)) {
    starred.delete(className);
  } else {
    starred.add(className);
  }
  state.starredClasses = Array.from(starred).sort((left, right) => left.localeCompare(right));
  renderBrowser();
  renderClassDetail();
  renderExportExperience();
  saveDraftState({ immediate: true });
}

function pruneStarredState() {
  const validNames = new Set((state.targetScan?.classes || []).map((item) => item.name));
  state.starredClasses = normalizedStarredClasses().filter((name) => validNames.has(name));
}

function exportClassExists(className) {
  return Boolean(exportableClasses().some((item) => item.name === className));
}

function effectiveExportSelectionCount(item) {
  const custom = state.exportSelections?.[item.name];
  if (Array.isArray(custom)) {
    return Math.min(custom.length, state.exportPerClassLimit);
  }
  return Math.min(item.counts?.train || 0, state.exportPerClassLimit);
}

function customizedExportClassCount(dataset = state.targetScan) {
  const validNames = new Set(exportableClasses(dataset).map((item) => item.name));
  return Object.keys(state.exportSelections || {}).filter((name) => validNames.has(name)).length;
}

function missingExportClassCount(dataset = state.targetScan) {
  return exportableClasses(dataset).filter((item) => effectiveExportSelectionCount(item) === 0).length;
}

function totalSelectedExportImages(dataset = state.targetScan) {
  return exportableClasses(dataset).reduce((sum, item) => sum + effectiveExportSelectionCount(item), 0);
}

function autoSelectionPathsFromImages(images) {
  return images.slice(0, state.exportPerClassLimit).map((image) => image.path);
}

function randomSelectionPathsFromImages(images) {
  const pool = [...images];
  for (let index = pool.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(Math.random() * (index + 1));
    [pool[index], pool[swapIndex]] = [pool[swapIndex], pool[index]];
  }
  return pool.slice(0, state.exportPerClassLimit).map((image) => image.path);
}

function orderedSelectedPaths(images, selectedSet) {
  return images
    .filter((image) => selectedSet.has(image.path))
    .map((image) => image.path)
    .slice(0, state.exportPerClassLimit);
}

function arraysEqual(left, right) {
  if (left.length !== right.length) {
    return false;
  }
  return left.every((value, index) => value === right[index]);
}

function trainSplitFromDetail(detail) {
  return detail?.splits?.find((splitBlock) => splitBlock.split === "train") || {
    split: "train",
    count: 0,
    truncated: false,
    images: [],
  };
}

function effectiveExportSelectionPaths(className, images) {
  const custom = state.exportSelections?.[className];
  if (Array.isArray(custom)) {
    const allowed = new Set(images.map((image) => image.path));
    return custom.filter((path) => allowed.has(path)).slice(0, state.exportPerClassLimit);
  }
  return autoSelectionPathsFromImages(images);
}

function pruneExportState() {
  const validNames = new Set(exportableClasses().map((item) => item.name));
  const nextSelections = {};
  for (const [className, paths] of Object.entries(state.exportSelections || {})) {
    if (!validNames.has(className) || !Array.isArray(paths)) {
      continue;
    }
    nextSelections[className] = Array.from(new Set(paths)).slice(0, state.exportPerClassLimit);
  }
  state.exportSelections = nextSelections;
  if (!exportClassExists(state.exportSelectedClass)) {
    state.exportSelectedClass = null;
    state.exportClassDetail = null;
  }
}

function trimExportSelectionsToLimit() {
  for (const [className, paths] of Object.entries(state.exportSelections || {})) {
    state.exportSelections[className] = Array.from(new Set(paths)).slice(0, state.exportPerClassLimit);
  }
}

function setExportLimit(limit) {
  if (state.exportPerClassLimit === limit) {
    return;
  }
  state.exportPerClassLimit = limit;
  syncExportOutputFilenameAutoValue();
  state.exportResult = null;
  trimExportSelectionsToLimit();
  renderExportExperience();
  saveDraftState();
}

function renderExportSortControls() {
  el("export-sort-size-desc").classList.toggle("is-active", state.exportSort === "size_desc");
  el("export-sort-size-asc").classList.toggle("is-active", state.exportSort === "size_asc");
  el("export-limit-20").classList.toggle("is-active", state.exportPerClassLimit === 20);
  el("export-limit-40").classList.toggle("is-active", state.exportPerClassLimit === 40);
  el("export-sort-size-desc").disabled = state.busy;
  el("export-sort-size-asc").disabled = state.busy;
  el("export-limit-20").disabled = state.busy;
  el("export-limit-40").disabled = state.busy;
}

function renderExportDetailModeControls() {
  const hasDetail = Boolean(state.exportClassDetail);
  el("export-detail-mode-all").classList.toggle("is-active", state.exportDetailMode === "all");
  el("export-detail-mode-selected").classList.toggle("is-active", state.exportDetailMode === "selected");
  el("export-detail-mode-all").disabled = state.busy || !hasDetail;
  el("export-detail-mode-selected").disabled = state.busy || !hasDetail;
}

function setExportStatus(tone, title, message, detail = "") {
  state.exportStatus = { tone, title, message, detail };
}

function exportFilterSets(dataset = state.targetScan) {
  const exportNames = new Set(exportableClasses(dataset).map((item) => item.name));
  const mergedNames = new Set([...mergedClassNames(dataset)].filter((name) => exportNames.has(name)));
  const newNames = new Set([...newProductClassNames(dataset)].filter((name) => exportNames.has(name)));
  const unknownNames = new Set([...unknownClassNames(dataset)].filter((name) => exportNames.has(name)));
  return { mergedNames, newNames, unknownNames };
}

function renderExportFilters(dataset) {
  const { mergedNames, newNames, unknownNames } = exportFilterSets(dataset);
  const starredNames = new Set([...starredClassNames(dataset)].filter((name) => exportableClasses(dataset).some((item) => item.name === name)));
  el("export-merged-class-count").textContent = `${formatNumber(mergedNames.size)} merged`;
  el("export-new-product-count").textContent = `${formatNumber(newNames.size)} new`;
  el("export-unknown-class-count").textContent = `${formatNumber(unknownNames.size)} unknown`;
  el("export-starred-class-count").textContent = `${formatNumber(starredNames.size)} starred`;
  el("export-filter-all-classes").classList.toggle("is-active", state.exportFilter === "all");
  el("export-filter-merged-classes").classList.toggle("is-active", state.exportFilter === "merged");
  el("export-filter-new-products").classList.toggle("is-active", state.exportFilter === "new_products");
  el("export-filter-unknown-classes").classList.toggle("is-active", state.exportFilter === "unknowns");
  el("export-filter-starred-classes").classList.toggle("is-active", state.exportFilter === "starred");
  el("export-filter-all-classes").disabled = state.busy;
  el("export-filter-merged-classes").disabled = state.busy || mergedNames.size === 0;
  el("export-filter-new-products").disabled = state.busy || newNames.size === 0;
  el("export-filter-unknown-classes").disabled = state.busy || unknownNames.size === 0;
  el("export-filter-starred-classes").disabled = state.busy || starredNames.size === 0;
}

function renderExportStatusPanel() {
  const wrap = el("export-run-status");
  if (!wrap) {
    return;
  }

  const status = state.exportStatus || {
    tone: "idle",
    title: "No export activity yet",
    message: "Save a selection snapshot or run an export to see progress and results here.",
    detail: "",
  };

  const toneClass = ["running", "success", "error"].includes(status.tone) ? status.tone : "idle";
  wrap.className = toneClass === "idle" ? "export-status-panel empty-state" : `export-status-panel ${toneClass}`;
  wrap.innerHTML = `
    <div class="export-status-head">
      <span class="export-status-dot"></span>
      <strong>${status.title}</strong>
    </div>
    <p>${status.message}</p>
    ${status.detail ? `<div class="export-status-detail">${status.detail}</div>` : ""}
  `;
}

function renderExportSummaryPanel() {
  const dataset = state.targetScan;
  const cardsWrap = el("export-summary-cards");
  const copy = el("export-page-copy");
  const output = el("export-output-copy");
  const pill = el("export-page-pill");
  const button = el("btn-export-npz");
  const saveButton = el("btn-export-save-selection");
  const randomButton = el("btn-export-random-all");
  const clearAllButton = el("btn-export-clear-all");

  if (!cardsWrap || !copy || !output || !pill || !button || !saveButton || !randomButton || !clearAllButton) {
    return;
  }

  renderExportSortControls();
  renderExportStatusPanel();
  button.disabled = state.busy || !dataset;

  if (!dataset) {
    pill.textContent = "No target loaded";
    copy.textContent = "Scan a target dataset or complete a merge, then handpick train images per product for NPZ export.";
    output.textContent = "Output: scan a target dataset to resolve the export path.";
    saveButton.disabled = true;
    randomButton.disabled = true;
    clearAllButton.disabled = true;
    cardsWrap.innerHTML = `<div class="empty-state">No export target loaded yet.</div>`;
    return;
  }

  const classes = exportableClasses(dataset);
  const customizedCount = customizedExportClassCount(dataset);
  const missingCount = missingExportClassCount(dataset);
  const selectedCount = totalSelectedExportImages(dataset);

  pill.textContent = `${formatNumber(classes.length)} exportable classes`;
  copy.textContent = "Handpick the train images you want per product. Untouched products keep the current auto-pick baseline, and any class with zero picked images will block export.";
  output.textContent = `Output: ${inferredExportOutputPath()}`;
  saveButton.disabled = state.busy || !dataset || classes.length === 0;
  button.disabled = state.busy || !dataset || classes.length === 0 || missingCount > 0;
  randomButton.disabled = state.busy || !dataset || classes.length === 0;
  clearAllButton.disabled = state.busy || !dataset || classes.length === 0;

  const cards = [
    summaryCard({
      label: "Export Limit",
      value: `${formatNumber(state.exportPerClassLimit)} / class`,
      meta: `${formatNumber(classes.length)} classes with train images`,
    }),
    summaryCard({
      label: "Selected Images",
      value: `${formatNumber(selectedCount)}`,
      meta: `${formatNumber(customizedCount)} customized classes`,
      tone: "highlight",
    }),
    summaryCard({
      label: "Needs Picks",
      value: `${formatNumber(missingCount)}`,
      meta: missingCount ? "Clear selections or empty classes need attention" : "Ready to export",
      tone: missingCount ? "warning" : "",
    }),
  ];

  if (state.exportResult) {
    cards.push(
      summaryCard({
        label: "Last Export",
        value: `${formatNumber(state.exportResult.image_count)} images`,
        meta: state.exportResult.output_path,
      })
    );
  }

  cardsWrap.innerHTML = cards.join("");
}

function renderExportBrowser() {
  const browser = el("export-class-browser");
  const dataset = state.targetScan;
  const query = (el("export-class-search")?.value || "").trim().toLowerCase();
  const { mergedNames, newNames, unknownNames } = exportFilterSets(dataset);
  const starredNames = new Set([...starredClassNames(dataset)].filter((name) => exportableClasses(dataset).some((item) => item.name === name)));

  if (!browser) {
    return;
  }

  renderExportSortControls();
  renderExportFilters(dataset);
  el("export-customized-count").textContent = `${formatNumber(customizedExportClassCount(dataset))} customized`;
  el("export-missing-count").textContent = `${formatNumber(missingExportClassCount(dataset))} missing`;

  if (!dataset) {
    browser.className = "class-browser empty-state";
    browser.innerHTML = "Scan a target dataset to handpick export images here.";
    el("export-browser-count").textContent = "0 classes";
    return;
  }

  const classes = exportableClasses(dataset)
    .filter((item) => {
      if (!item.name.toLowerCase().includes(query)) {
        return false;
      }
      if (state.exportFilter === "merged" && !mergedNames.has(item.name)) {
        return false;
      }
      if (state.exportFilter === "new_products" && !newNames.has(item.name)) {
        return false;
      }
      if (state.exportFilter === "unknowns" && !unknownNames.has(item.name)) {
        return false;
      }
      if (state.exportFilter === "starred" && !starredNames.has(item.name)) {
        return false;
      }
      return true;
    })
    .sort((left, right) => {
      if (state.exportSort === "size_asc") {
        if ((left.counts?.train || 0) !== (right.counts?.train || 0)) {
          return (left.counts?.train || 0) - (right.counts?.train || 0);
        }
      } else if ((left.counts?.train || 0) !== (right.counts?.train || 0)) {
        return (right.counts?.train || 0) - (left.counts?.train || 0);
      }
      return left.name.localeCompare(right.name);
    });

  const filteredLabel =
    state.exportFilter === "merged"
      ? `${formatNumber(classes.length)} merged classes`
      : state.exportFilter === "new_products"
        ? `${formatNumber(classes.length)} new products`
        : state.exportFilter === "unknowns"
          ? `${formatNumber(classes.length)} unknown classes`
          : state.exportFilter === "starred"
            ? `${formatNumber(classes.length)} starred classes`
          : `${formatNumber(classes.length)} classes`;
  el("export-browser-count").textContent = filteredLabel;
  if (!classes.length) {
    browser.className = "class-browser empty-state";
    browser.innerHTML =
      state.exportFilter === "merged"
        ? "No merged exportable classes match the current search yet."
        : state.exportFilter === "new_products"
          ? "No new exportable products match the current search yet."
          : state.exportFilter === "unknowns"
            ? "No unknown exportable classes match the current search yet."
            : state.exportFilter === "starred"
              ? "No starred exportable classes match the current search yet."
            : "No exportable classes match the current search.";
    return;
  }

  browser.className = "class-browser";
  browser.innerHTML = classes
    .map((item) => {
      const selectedCount = effectiveExportSelectionCount(item);
      const isCustomized = Array.isArray(state.exportSelections?.[item.name]);
      const isMissing = selectedCount === 0;
      return `
        <div class="class-row ${state.exportSelectedClass === item.name ? "active" : ""}" data-export-class="${item.name}">
          <div class="class-row-layout">
            <div class="class-row-preview-wrap">
              ${
                item.sample_path
                  ? `<img class="class-row-preview" src="${imageUrl(item.sample_path)}" alt="${item.name}" loading="lazy" />`
                  : `<div class="class-row-preview class-row-preview-empty">No Preview</div>`
              }
            </div>
            <div class="class-row-content">
              <div class="class-row-head">
                <strong>${item.name}</strong>
                <div class="class-row-meta">
                  ${mergedNames.has(item.name) ? '<span class="merge-badge">Merged</span>' : ""}
                  ${newNames.has(item.name) ? '<span class="merge-badge">New</span>' : ""}
                  ${unknownNames.has(item.name) ? '<span class="merge-badge unknown-badge">Unknown</span>' : ""}
                  ${starredNames.has(item.name) ? '<span class="merge-badge star-badge" title="Starred">★</span>' : ""}
                  <span class="selection-badge ${isMissing ? "warning" : "muted"}">${formatNumber(selectedCount)} / ${formatNumber(state.exportPerClassLimit)}</span>
                  <span class="selection-badge ${isCustomized ? "" : "muted"}">${isCustomized ? "Custom" : "Auto"}</span>
                </div>
              </div>
              <p>train ${formatNumber(item.counts?.train || 0)} · total ${formatNumber(item.total || 0)}</p>
            </div>
          </div>
        </div>
      `;
    })
    .join("");

  browser.querySelectorAll("[data-export-class]").forEach((row) => {
    row.addEventListener("click", () => loadExportClassDetail(row.dataset.exportClass));
  });
}

function renderExportClassDetail() {
  const wrap = el("export-class-detail");
  const detail = state.exportClassDetail;
  renderExportDetailModeControls();

  if (!wrap) {
    return;
  }

  if (!detail) {
    wrap.className = "class-detail empty-state";
    wrap.innerHTML = "Pick a class from the export browser to handpick its train images.";
    el("export-detail-pill").textContent = "No class selected";
    return;
  }

  const trainSplit = trainSplitFromDetail(detail);
  const selectedPaths = effectiveExportSelectionPaths(detail.class_name, trainSplit.images);
  const selectedSet = new Set(selectedPaths);
  const isCustomized = Array.isArray(state.exportSelections?.[detail.class_name]);
  const isStarred = isClassStarred(detail.class_name);
  const visibleImages =
    state.exportDetailMode === "selected"
      ? trainSplit.images.filter((image) => selectedSet.has(image.path))
      : trainSplit.images;

  el("export-detail-pill").textContent = `${detail.class_name} · ${formatNumber(selectedPaths.length)} selected`;
  wrap.className = "class-detail";
  wrap.innerHTML = `
    <div class="class-detail-header">
      <div>
        <h3>${detail.class_name}</h3>
        <p>${formatNumber(trainSplit.count)} train images available. Click images to toggle them into the export set.</p>
      </div>
      <div class="button-row">
        <button id="btn-export-toggle-star" class="ghost-button star-toggle-button ${isStarred ? "is-active" : ""}" title="${isStarred ? "Unstar" : "Star"}">${isStarred ? "★" : "☆"}</button>
        <button id="btn-export-reset-auto" class="secondary-button">Randomize Selection</button>
        <button id="btn-export-clear-selection" class="ghost-button">Clear Selection</button>
      </div>
    </div>

    <div class="selection-summary-grid">
      ${summaryCard({
        label: "Selected",
        value: `${formatNumber(selectedPaths.length)} / ${formatNumber(state.exportPerClassLimit)}`,
        meta: isCustomized ? "Custom override saved" : "Auto-picked baseline",
        tone: selectedPaths.length ? "highlight" : "warning",
      })}
      ${summaryCard({
        label: "Train Images",
        value: `${formatNumber(trainSplit.count)}`,
        meta: trainSplit.truncated ? "Preview truncated" : "Full train split loaded",
      })}
      ${summaryCard({
        label: "Mode",
        value: isCustomized ? "Custom" : "Auto",
        meta: "Auto uses the first sorted images until you override it",
      })}
    </div>

    <div class="split-block">
      <h4>${state.exportDetailMode === "selected" ? "Selected" : "Train"} · ${formatNumber(visibleImages.length)} images</h4>
      ${
        visibleImages.length
          ? `
            <div class="image-grid">
              ${visibleImages
                .map(
                  (image) => `
                    <div class="image-card selectable-image-card ${selectedSet.has(image.path) ? "is-selected" : ""}" data-export-toggle="${image.path}">
                      <img src="${imageUrl(image.path)}" alt="${image.name}" loading="lazy" />
                      <div class="selection-state-row">
                        <span class="selection-badge ${selectedSet.has(image.path) ? "" : "muted"}">${selectedSet.has(image.path) ? "Selected" : "Not Selected"}</span>
                        <p>${image.source_hint ? `source ${image.source_hint}` : "manual or legacy file"}</p>
                      </div>
                      <p><strong>${image.name}</strong></p>
                    </div>
                  `
                )
                .join("")}
            </div>
          `
          : `<div class="empty-state">${
              state.exportDetailMode === "selected"
                ? "No selected images for this class yet."
                : "No train images are available for this class yet."
            }</div>`
      }
    </div>
  `;

  el("btn-export-toggle-star").addEventListener("click", () => toggleStarredClass(detail.class_name));
  el("btn-export-reset-auto").addEventListener("click", resetExportSelectionToAuto);
  el("btn-export-clear-selection").addEventListener("click", clearExportSelection);
  wrap.querySelectorAll("[data-export-toggle]").forEach((card) => {
    card.addEventListener("click", () => toggleExportImageSelection(card.dataset.exportToggle));
  });
}

function renderExportExperience() {
  pruneExportState();
  renderExportSummaryPanel();
  renderExportBrowser();
  renderExportClassDetail();
}

function latestMergeEntry(dataset) {
  const history = dataset?.merge_history || [];
  return history.length ? history[0] : null;
}

function mergedClassNames(dataset) {
  const names = new Set();
  const entry = latestMergeEntry(dataset);
  if (!entry) {
    return names;
  }
  for (const className of Object.keys(entry.per_target_counts || {})) {
    names.add(className);
  }
  for (const mapping of entry.class_mappings || []) {
    if (mapping?.enabled && mapping?.target_class) {
      names.add(mapping.target_class);
    }
  }
  return names;
}

function mergedFilenameProvenance(path) {
  if (!path) {
    return null;
  }
  const filename = String(path).split("/").pop() || "";
  const parts = filename.split("__");
  if (parts.length < 4) {
    return null;
  }

  const [sourceTag, split, sourceClass] = parts;
  if (!["train", "val", "test"].includes(split) || !sourceTag || !sourceClass) {
    return null;
  }

  return { sourceTag, split, sourceClass };
}

function newProductClassNames(dataset) {
  const names = new Set();
  const entry = latestMergeEntry(dataset);
  if (!entry) {
    return names;
  }

  const latestNewSourceClasses = new Set();
  for (const mapping of entry.class_mappings || []) {
    if (!mapping?.enabled || !mapping?.target_class || !mapping?.source_class) {
      continue;
    }

    if (mapping.action === "create" || mapping.action === "create_placeholder") {
      names.add(mapping.target_class);
      latestNewSourceClasses.add(mapping.source_class);
      continue;
    }

    if (
      typeof mapping.target_exists === "boolean" &&
      mapping.target_exists === false
    ) {
      names.add(mapping.target_class);
      latestNewSourceClasses.add(mapping.source_class);
      continue;
    }

    if (
      entry.source_tag &&
      mapping.target_class.startsWith(`${entry.source_tag}__unknown_`)
    ) {
      names.add(mapping.target_class);
      latestNewSourceClasses.add(mapping.source_class);
    }
  }

  for (const item of dataset?.classes || []) {
    const provenance = mergedFilenameProvenance(item.sample_path);
    if (!provenance) {
      continue;
    }
    if (
      provenance.sourceTag === entry.source_tag &&
      latestNewSourceClasses.has(provenance.sourceClass)
    ) {
      names.add(item.name);
    }
  }
  return names;
}

function isUnknownClassName(name) {
  return typeof name === "string" && name.toLowerCase().includes("unknown");
}

function unknownClassNames(dataset) {
  const names = new Set();
  for (const item of dataset?.classes || []) {
    if (isUnknownClassName(item.name)) {
      names.add(item.name);
    }
  }
  return names;
}

function renderBrowseSortControls() {
  el("sort-size-desc").classList.toggle("is-active", state.browserSort === "size_desc");
  el("sort-size-asc").classList.toggle("is-active", state.browserSort === "size_asc");
  el("sort-size-desc").disabled = state.busy;
  el("sort-size-asc").disabled = state.busy;
}

function renderBrowseFilters(dataset) {
  const mergedNames = mergedClassNames(dataset);
  const newNames = newProductClassNames(dataset);
  const unknownNames = unknownClassNames(dataset);
  const starredNames = starredClassNames(dataset);
  el("merged-class-count").textContent = `${formatNumber(mergedNames.size)} merged`;
  el("new-product-count").textContent = `${formatNumber(newNames.size)} new`;
  el("unknown-class-count").textContent = `${formatNumber(unknownNames.size)} unknown`;
  el("starred-class-count").textContent = `${formatNumber(starredNames.size)} starred`;
  el("filter-all-classes").classList.toggle("is-active", state.browserFilter === "all");
  el("filter-merged-classes").classList.toggle("is-active", state.browserFilter === "merged");
  el("filter-new-products").classList.toggle("is-active", state.browserFilter === "new_products");
  el("filter-unknown-classes").classList.toggle("is-active", state.browserFilter === "unknowns");
  el("filter-starred-classes").classList.toggle("is-active", state.browserFilter === "starred");
  el("filter-merged-classes").disabled = state.busy || mergedNames.size === 0;
  el("filter-new-products").disabled = state.busy || newNames.size === 0;
  el("filter-unknown-classes").disabled = state.busy || unknownNames.size === 0;
  el("filter-starred-classes").disabled = state.busy || starredNames.size === 0;
  renderBrowseSortControls();
}

function fillPath(kind, path) {
  if (kind === "target") {
    el("target-path").value = path;
  } else {
    el("source-path").value = path;
  }
  saveDraftState();
}

function datasetRow(item, sourceRole) {
  return `
    <div class="dataset-row">
      <div class="dataset-row-head">
        <strong>${item.relative_path || item.path}</strong>
        <span class="muted-pill">${formatNumber(item.class_count)} classes</span>
      </div>
      <p>${splitSummary(item.splits || {})}</p>
      <p>${item.kind === "recognition_export" ? "Recognition export" : "Working dataset"}</p>
      <div class="dataset-buttons">
        <button class="secondary-button" data-fill="target" data-path="${item.path}">Use as target</button>
        <button class="ghost-button" data-fill="source" data-path="${item.path}">${sourceRole}</button>
      </div>
    </div>
  `;
}

function renderDiscovery() {
  const working = state.discovered.filter((item) => item.kind === "working_dataset");
  const exportsList = state.discovered.filter((item) => item.kind === "recognition_export");
  el("discovery-count").textContent = `${state.discovered.length}`;
  el("working-datasets").innerHTML = working.length
    ? working.map((item) => datasetRow(item, "Use as source")).join("")
    : "No working datasets found under /workspace/vdata.";
  el("recognition-exports").innerHTML = exportsList.length
    ? exportsList.map((item) => datasetRow(item, "Use as source")).join("")
    : "No recognition exports found under /workspace/vdata.";

  document.querySelectorAll("[data-fill]").forEach((button) => {
    button.addEventListener("click", () => fillPath(button.dataset.fill, button.dataset.path));
  });
}

function actionLabel(action) {
  if (action === "merge") return "Merge into existing";
  if (action === "create_placeholder") return "Create placeholder";
  return "Create new";
}

function filteredPreviewMappings(preview) {
  const mappings = preview?.mappings || [];
  return mappings
    .map((mapping, index) => ({ mapping, index }))
    .filter(({ mapping }) => {
      if (state.mergeFilter === "source_only") {
        return !mapping.target_exists;
      }
      if (state.mergeFilter === "existing") {
        return mapping.target_exists;
      }
      if (state.mergeFilter === "placeholders") {
        return mapping.action === "create_placeholder";
      }
      return true;
    });
}

function renderMergeFilters(preview) {
  const hasPreview = hasMergeMappings(preview);
  const filterIds = [
    ["merge-filter-all", "all"],
    ["merge-filter-source-only", "source_only"],
    ["merge-filter-existing", "existing"],
    ["merge-filter-placeholders", "placeholders"],
  ];

  for (const [id, value] of filterIds) {
    const button = el(id);
    button.classList.toggle("is-active", state.mergeFilter === value);
    button.disabled = state.busy || !hasPreview;
  }
}

function renderMergeTable() {
  const wrap = el("merge-table-wrap");
  const preview = state.preview;
  const mergeReady = hasMergeMappings(preview);
  el("btn-commit-merge").disabled = state.busy || !mergeReady;
  el("btn-default-mappings").disabled = state.busy || !mergeReady;
  renderMergeFilters(preview);
  if (!preview) {
    wrap.className = "table-wrap empty-state";
    wrap.innerHTML = "Build a merge preview to review source classes and choose where they land.";
    el("mapping-count").textContent = "0 classes";
    return;
  }

  if (!hasMergeMappings(preview)) {
    wrap.className = "table-wrap empty-state";
    wrap.innerHTML =
      preview.mode === "target_only"
        ? "No merge source selected. You can browse, edit, and export this target dataset directly."
        : "No merge mappings are available for this preview.";
    el("mapping-count").textContent = preview.mode === "target_only" ? "No merge" : "0 classes";
    return;
  }

  const visibleMappings = filteredPreviewMappings(preview);
  wrap.className = "table-wrap";
  el("mapping-count").textContent =
    visibleMappings.length === preview.mappings.length
      ? `${preview.mappings.length} classes`
      : `${visibleMappings.length} of ${preview.mappings.length} classes`;

  wrap.innerHTML = `
    <table>
      <thead>
        <tr>
          <th class="checkbox-cell">
            <input id="merge-select-all-visible" class="table-checkbox" type="checkbox" />
          </th>
          <th>Source class</th>
          <th>Source counts</th>
          <th>Target class</th>
          <th>Suggested action</th>
        </tr>
      </thead>
      <tbody>
        ${visibleMappings
          .map(
            ({ mapping, index }) => `
              <tr>
                <td class="checkbox-cell">
                  <input type="checkbox" class="mapping-enabled" data-index="${index}" ${mapping.enabled ? "checked" : ""} />
                </td>
                <td>
                  <strong>${mapping.source_class}</strong>
                </td>
                <td>${splitSummary(mapping.source_counts)}</td>
                <td>
                  <input class="mapping-input" data-index="${index}" value="${mapping.target_class}" />
                </td>
                <td>
                  <span class="action-pill">${actionLabel(mapping.action)}</span>
                </td>
              </tr>
            `
          )
          .join("")}
      </tbody>
    </table>
  `;

  const selectAllVisible = wrap.querySelector("#merge-select-all-visible");
  if (selectAllVisible) {
    const enabledCount = visibleMappings.filter(({ mapping }) => mapping.enabled).length;
    selectAllVisible.checked = visibleMappings.length > 0 && enabledCount === visibleMappings.length;
    selectAllVisible.indeterminate = enabledCount > 0 && enabledCount < visibleMappings.length;
    selectAllVisible.addEventListener("change", () => {
      for (const { index } of visibleMappings) {
        state.preview.mappings[index].enabled = selectAllVisible.checked;
      }
      renderMergeTable();
      saveDraftState({ immediate: true });
    });
  }

  wrap.querySelectorAll(".mapping-input").forEach((input) => {
    input.addEventListener("input", () => {
      const index = Number(input.dataset.index);
      state.preview.mappings[index].target_class = input.value.trim();
      saveDraftState();
    });
  });

  wrap.querySelectorAll(".mapping-enabled").forEach((input) => {
    input.addEventListener("change", () => {
      const index = Number(input.dataset.index);
      state.preview.mappings[index].enabled = input.checked;
      saveDraftState();
    });
  });
}

function renderBrowser() {
  const browser = el("class-browser");
  const dataset = state.targetScan;
  const query = el("class-search").value.trim().toLowerCase();
  const targetPath = el("target-path").value.trim();
  const mergedNames = mergedClassNames(dataset);
  const newNames = newProductClassNames(dataset);
  const unknownNames = unknownClassNames(dataset);
  const starredNames = starredClassNames(dataset);

  el("browse-target-pill").textContent = dataset ? `${formatNumber(dataset.class_count || 0)} classes` : "No target loaded";
  el("browse-target-copy").textContent = dataset
    ? `${targetPath} · ${formatNumber(totalImages(dataset))} images across the current target dataset.`
    : "Scan a target dataset or complete a merge, then browse and curate the resulting classes here.";
  renderBrowseFilters(dataset);

  if (!dataset || !dataset.classes || !dataset.classes.length) {
    browser.innerHTML = "Scan a target dataset to browse classes here.";
    browser.className = "class-browser empty-state";
    el("browser-count").textContent = "0 classes";
    return;
  }

  const classes = dataset.classes.filter((item) => {
    if (!item.name.toLowerCase().includes(query)) {
      return false;
    }
    if (state.browserFilter === "merged" && !mergedNames.has(item.name)) {
      return false;
    }
    if (state.browserFilter === "new_products" && !newNames.has(item.name)) {
      return false;
    }
    if (state.browserFilter === "unknowns" && !unknownNames.has(item.name)) {
      return false;
    }
    if (state.browserFilter === "starred" && !starredNames.has(item.name)) {
      return false;
    }
    return true;
  });

  classes.sort((left, right) => {
    if (state.browserSort === "size_asc") {
      if ((left.total || 0) !== (right.total || 0)) {
        return (left.total || 0) - (right.total || 0);
      }
    } else {
      if ((left.total || 0) !== (right.total || 0)) {
        return (right.total || 0) - (left.total || 0);
      }
    }
    return left.name.localeCompare(right.name);
  });

  const filteredLabel =
    state.browserFilter === "merged"
      ? `${classes.length} merged classes`
      : state.browserFilter === "new_products"
        ? `${classes.length} new products`
        : state.browserFilter === "unknowns"
          ? `${classes.length} unknown classes`
          : state.browserFilter === "starred"
            ? `${classes.length} starred classes`
        : `${classes.length} classes`;
  el("browser-count").textContent = filteredLabel;
  browser.className = "class-browser";
  if (!classes.length) {
    browser.className = "class-browser empty-state";
    browser.innerHTML =
      state.browserFilter === "merged"
        ? "No merged classes match the current search yet."
        : state.browserFilter === "new_products"
        ? "No new products match the current search yet."
        : state.browserFilter === "unknowns"
          ? "No unknown classes match the current search yet."
          : state.browserFilter === "starred"
            ? "No starred classes match the current search yet."
          : "No classes match the current search.";
    return;
  }

  browser.innerHTML = classes
    .map(
      (item) => `
        <div class="class-row ${state.selectedClass === item.name ? "active" : ""}" data-class="${item.name}">
          <div class="class-row-layout">
            <div class="class-row-preview-wrap">
              ${
                item.sample_path
                  ? `<img class="class-row-preview" src="${imageUrl(item.sample_path)}" alt="${item.name}" loading="lazy" />`
                  : `<div class="class-row-preview class-row-preview-empty">No Preview</div>`
              }
            </div>
            <div class="class-row-content">
              <div class="class-row-head">
                <strong>${item.name}</strong>
                <div class="class-row-meta">
                  ${mergedNames.has(item.name) ? '<span class="merge-badge">Merged</span>' : ""}
                  ${newNames.has(item.name) ? '<span class="merge-badge">New</span>' : ""}
                  ${unknownNames.has(item.name) ? '<span class="merge-badge unknown-badge">Unknown</span>' : ""}
                  ${starredNames.has(item.name) ? '<span class="merge-badge star-badge" title="Starred">★</span>' : ""}
                  <span class="muted-pill">${formatNumber(item.total)}</span>
                </div>
              </div>
              <p>${splitSummary(item.counts)}</p>
            </div>
          </div>
        </div>
      `
    )
    .join("");

  browser.querySelectorAll("[data-class]").forEach((row) => {
    row.addEventListener("click", () => loadClassDetail(row.dataset.class));
  });
}

function imageUrl(path) {
  return `/api/images/file?path=${encodeURIComponent(path)}`;
}

function classExists(className) {
  return Boolean(state.targetScan?.classes?.some((item) => item.name === className));
}

function visibleDetailImages(detail, activeSplit) {
  if (!detail || !activeSplit) {
    return [];
  }
  return activeSplit.images || [];
}

function pruneDetailSelection(visibleImages) {
  const allowed = new Set((visibleImages || []).map((image) => image.path));
  state.detailSelectedImages = (state.detailSelectedImages || []).filter((path) => allowed.has(path));
}

function resetDetailSelection() {
  state.detailSelectedImages = [];
}

function toggleDetailImageSelection(imagePath, additive) {
  const current = new Set(state.detailSelectedImages || []);
  if (additive) {
    if (current.has(imagePath)) {
      current.delete(imagePath);
    } else {
      current.add(imagePath);
    }
    state.detailSelectedImages = Array.from(current);
  } else {
    state.detailSelectedImages = [imagePath];
  }
  renderClassDetail();
}

function renderMergeHistory(entries) {
  if (!entries || !entries.length) {
    return "";
  }

  return `
    <div class="history-block">
      <h4>Recent Merge History</h4>
      ${entries
        .slice(0, 5)
        .map(
          (entry) => `
            <div class="history-card">
              <div class="dataset-row-head">
                <strong>${entry.merge_id}</strong>
                <span class="muted-pill">${formatNumber(entry.copied_count || 0)} copied</span>
              </div>
              <p>${entry.source_path}</p>
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function renderClassDetail() {
  const wrap = el("class-detail");
  const detail = state.classDetail;
  const dataset = state.targetScan;

  if (!detail) {
    wrap.className = "class-detail empty-state";
    wrap.innerHTML = "Pick a class from the browser to inspect images and curate them.";
    el("detail-pill").textContent = "No class selected";
    return;
  }

  wrap.className = "class-detail";
  const splitMap = new Map(detail.splits.map((splitBlock) => [splitBlock.split, splitBlock]));
  const allSplit = {
    split: "all",
    count: detail.total,
    truncated: detail.splits.some((splitBlock) => splitBlock.truncated),
    images: detail.splits.flatMap((splitBlock) => splitBlock.images),
  };
  const splitTabs = [
    allSplit,
    ...["train", "val", "test"].map((splitName) => {
      const existing = splitMap.get(splitName);
      if (existing) {
        return existing;
      }
      return {
        split: splitName,
        count: 0,
        truncated: false,
        images: [],
      };
    }),
  ];
  const availableSplits = splitTabs.map((splitBlock) => splitBlock.split);
  if (!availableSplits.includes(state.classDetailSplit)) {
    state.classDetailSplit = "all";
  }
  const activeSplit = splitTabs.find((splitBlock) => splitBlock.split === state.classDetailSplit) || splitTabs[0];
  const visibleImages = visibleDetailImages(detail, activeSplit);
  pruneDetailSelection(visibleImages);
  const isStarred = isClassStarred(detail.class_name);
  if (!state.detailActionClass) {
    state.detailActionClass = detail.class_name;
  }
  if (!state.detailActionSplit || !["train", "val", "test"].includes(state.detailActionSplit)) {
    state.detailActionSplit = activeSplit.split === "all" ? "train" : activeSplit.split;
  }
  const selectedCount = state.detailSelectedImages.length;
  const selectedSet = new Set(state.detailSelectedImages);

  el("detail-pill").textContent = `${detail.class_name} · ${formatNumber(detail.total)} images`;
  wrap.innerHTML = `
    <div class="class-detail-header">
      <div>
        <h3>${detail.class_name}</h3>
        <p>${formatNumber(detail.total)} images across ${detail.splits.length} splits.</p>
        <div class="rename-row">
          <input id="rename-class-input" type="text" value="${detail.class_name}" />
          <button id="btn-rename-class" class="secondary-button">Rename Class</button>
        </div>
      </div>
      <div class="detail-action-toolbar">
        <button id="btn-toggle-star-class" class="ghost-button star-toggle-button ${isStarred ? "is-active" : ""}" title="${isStarred ? "Unstar" : "Star"}">${isStarred ? "★" : "☆"}</button>
        <span class="selection-badge ${selectedCount ? "" : "muted"}">${formatNumber(selectedCount)} selected</span>
        <select id="detail-action-split">
          ${["train", "val", "test"]
            .map((split) => `<option value="${split}" ${split === state.detailActionSplit ? "selected" : ""}>${split}</option>`)
            .join("")}
        </select>
        <input id="detail-action-class" type="text" value="${state.detailActionClass}" />
        <button id="btn-move-selected-images" class="secondary-button" ${selectedCount ? "" : "disabled"}>Move</button>
        <button id="btn-trash-selected-images" class="danger-button" ${selectedCount ? "" : "disabled"}>Trash</button>
      </div>
    </div>

    <div class="split-tabs" role="tablist" aria-label="Class Detail Splits">
      ${splitTabs
        .map(
          (splitBlock) => `
            <button
              class="split-tab ${splitBlock.split === activeSplit.split ? "is-active" : ""}"
              type="button"
              data-split-tab="${splitBlock.split}"
            >
              ${splitBlock.split}
              <span class="split-tab-count">${formatNumber(splitBlock.count)}</span>
            </button>
          `
        )
        .join("")}
    </div>

    <div class="split-block">
      <h4>${activeSplit.split} · ${formatNumber(activeSplit.count)} images${activeSplit.truncated ? " · preview truncated" : ""}</h4>
      ${
        visibleImages.length
          ? `
            <div class="image-grid">
              ${visibleImages
                .map(
                  (image) => `
                    <div class="image-card selectable-image-card ${selectedSet.has(image.path) ? "is-selected" : ""}" data-detail-image="${image.path}">
                      <img src="${imageUrl(image.path)}" alt="${image.name}" loading="lazy" />
                      <div class="selection-state-row">
                        <span class="selection-badge ${selectedSet.has(image.path) ? "" : "muted"}">${selectedSet.has(image.path) ? "Selected" : "Not Selected"}</span>
                        <p>${image.split}</p>
                      </div>
                      <p><strong>${image.name}</strong></p>
                      <p>${image.source_hint ? `source ${image.source_hint}` : "manual or legacy file"}</p>
                    </div>
                  `
                )
                .join("")}
            </div>
          `
          : `<div class="empty-state">No images in ${activeSplit.split} for this class yet.</div>`
      }
    </div>

    ${renderMergeHistory(dataset?.merge_history || [])}
  `;

  el("btn-rename-class").addEventListener("click", renameSelectedClass);
  el("btn-toggle-star-class").addEventListener("click", () => toggleStarredClass(detail.class_name));
  wrap.querySelectorAll("[data-split-tab]").forEach((button) => {
    button.addEventListener("click", () => {
      state.classDetailSplit = button.dataset.splitTab;
      state.detailSelectedImages = [];
      if (button.dataset.splitTab !== "all") {
        state.detailActionSplit = button.dataset.splitTab;
      }
      renderClassDetail();
      saveDraftState();
    });
  });
  el("detail-action-split").addEventListener("change", (event) => {
    state.detailActionSplit = event.target.value;
  });
  el("detail-action-class").addEventListener("input", (event) => {
    state.detailActionClass = event.target.value.trim();
  });
  el("btn-move-selected-images").addEventListener("click", moveSelectedImages);
  el("btn-trash-selected-images").addEventListener("click", trashSelectedImages);
  wrap.querySelectorAll("[data-detail-image]").forEach((card) => {
    card.addEventListener("click", (event) => {
      toggleDetailImageSelection(card.dataset.detailImage, event.ctrlKey || event.metaKey);
    });
  });
}

async function refreshDiscovery() {
  setBusy(true, "Scanning /workspace/vdata");
  try {
    const data = await api("/api/datasets/discover");
    state.discovered = data.items || [];
    renderDiscovery();
    setStatus("Discovery updated");
  } catch (error) {
    alert(error.message);
    setStatus("Discovery failed");
  } finally {
    setBusy(false, state.preview ? "Preview loaded" : "Idle");
  }
}

async function scanTarget() {
  const targetPath = el("target-path").value.trim();
  if (!targetPath) {
    alert("Enter a target dataset path first.");
    return;
  }

  setBusy(true, "Scanning target");
  try {
    state.targetScan = await api("/api/datasets/scan", {
      method: "POST",
      body: JSON.stringify({ path: targetPath }),
    });
    state.exportResult = null;
    state.exportClassDetail = null;
    pruneStarredState();
    pruneExportState();
    renderSummaryCards(null, state.targetScan, null);
    renderBrowser();
    renderExportExperience();
    renderClassDetail();
    renderPageNav();
    saveDraftState({ immediate: true });
    setStatus("Target scanned");
  } catch (error) {
    alert(error.message);
    setStatus("Scan failed");
  } finally {
    setBusy(false, state.preview ? "Preview loaded" : "Idle");
  }
}

async function buildMergePreview() {
  const sourcePath = el("source-path").value.trim();
  const targetPath = el("target-path").value.trim();
  if (!targetPath) {
    alert("Enter a target dataset path first.");
    return;
  }

  setBusy(true, sourcePath ? "Building merge preview" : "Preparing target-only workspace");
  try {
    state.preview = await api("/api/merge/preview", {
      method: "POST",
      body: JSON.stringify({
        source_path: sourcePath,
        target_path: targetPath,
      }),
    });
    state.exportResult = null;
    state.targetScan = state.preview.target;
    state.exportClassDetail = null;
    pruneStarredState();
    pruneExportState();
    renderSummaryCards(state.preview.source, state.preview.target, state.preview);
    renderMergeTable();
    renderBrowser();
    renderExportExperience();
    renderClassDetail();
    renderPageNav();
    saveDraftState({ immediate: true });
    el("btn-commit-merge").disabled = !hasMergeMappings(state.preview);
    el("btn-default-mappings").disabled = !hasMergeMappings(state.preview);
    setStatus(sourcePath ? "Preview loaded" : "Target-only mode ready");
  } catch (error) {
    alert(error.message);
    setStatus("Preview failed");
  } finally {
    setBusy(false, state.preview ? "Preview loaded" : "Idle");
  }
}

function resetSuggestedMappings() {
  if (!state.preview) {
    return;
  }
  const refreshed = structuredClone(state.preview);
  state.preview = refreshed;
  renderMergeTable();
}

async function commitMerge() {
  if (!state.preview) {
    return;
  }

  setBusy(true, "Merging datasets");
  try {
    const payload = await api("/api/merge/commit", {
      method: "POST",
      body: JSON.stringify({
        source_path: el("source-path").value.trim(),
        target_path: el("target-path").value.trim(),
        mappings: state.preview.mappings.map((mapping) => ({
          source_class: mapping.source_class,
          target_class: mapping.target_class,
          enabled: Boolean(mapping.enabled),
          action: mapping.action,
          target_exists: mapping.target_exists,
        })),
      }),
    });

    state.targetScan = payload.target_scan;
    state.targetScan.merge_history = payload.merge_history || [];
    state.exportResult = null;
    state.exportClassDetail = null;
    pruneStarredState();
    pruneExportState();
    renderSummaryCards(state.preview.source, state.targetScan, state.preview);
    renderBrowser();
    renderPageNav();
    renderExportExperience();
    alert(
      `Merge complete.\nCopied: ${formatNumber(payload.copied_count)}\nSkipped existing: ${formatNumber(payload.skipped_existing)}\nSkipped disabled: ${formatNumber(payload.skipped_disabled)}`
    );
    setStatus(`Merged ${formatNumber(payload.copied_count)} images`);
    switchPage("browse");
    saveDraftState({ immediate: true });
  } catch (error) {
    alert(error.message);
    setStatus("Merge failed");
  } finally {
    setBusy(false, state.preview ? "Preview loaded" : "Idle");
  }
}

async function loadClassDetail(className) {
  const datasetPath = el("target-path").value.trim();
  if (!datasetPath) {
    return;
  }

  setBusy(true, `Loading ${className}`);
  try {
    state.selectedClass = className;
    state.detailActionClass = className;
    if (!state.classDetailSplit) {
      state.detailActionSplit = "train";
    } else if (state.classDetailSplit !== "all") {
      state.detailActionSplit = state.classDetailSplit;
    }
    resetDetailSelection();
    if (!state.classDetailSplit) {
      state.classDetailSplit = "all";
    }
    state.classDetail = await api("/api/classes/detail", {
      method: "POST",
      body: JSON.stringify({
        dataset_path: datasetPath,
        class_name: className,
        limit_per_split: 120,
      }),
    });
    renderBrowser();
    renderClassDetail();
    saveDraftState({ immediate: true });
    setStatus(`Loaded ${className}`);
  } catch (error) {
    alert(error.message);
    setStatus("Load failed");
  } finally {
    setBusy(false, state.preview ? "Preview loaded" : "Idle");
  }
}

async function moveSelectedImages() {
  const imagePaths = state.detailSelectedImages || [];
  const targetClass = (state.detailActionClass || "").trim();
  const targetSplit = state.detailActionSplit;
  if (!imagePaths.length) {
    alert("Select at least one image first.");
    return;
  }
  if (!targetClass) {
    alert("Enter a target class name first.");
    return;
  }

  setBusy(true, "Moving selected images");
  try {
    const payload = await api("/api/images/reassign-many", {
      method: "POST",
      body: JSON.stringify({
        dataset_path: targetPathValue(),
        image_paths: imagePaths,
        target_class: targetClass,
        target_split: targetSplit,
      }),
    });
    state.targetScan = payload.dataset;
    state.exportClassDetail = null;
    pruneExportState();
    resetDetailSelection();
    renderBrowser();
    renderExportExperience();
    const nextClass = classExists(state.selectedClass)
      ? state.selectedClass
      : classExists(targetClass)
        ? targetClass
        : null;
    if (nextClass) {
      await loadClassDetail(nextClass);
    } else {
      state.classDetail = null;
      state.selectedClass = null;
      renderClassDetail();
      saveDraftState({ immediate: true });
    }
    setStatus(`Moved ${formatNumber(payload.moved || 0)} images`);
  } catch (error) {
    alert(error.message);
    setStatus("Move failed");
  } finally {
    setBusy(false, state.preview ? "Preview loaded" : "Idle");
  }
}

async function trashSelectedImages() {
  const imagePaths = state.detailSelectedImages || [];
  if (!imagePaths.length) {
    alert("Select at least one image first.");
    return;
  }

  setBusy(true, "Trashing selected images");
  try {
    const payload = await api("/api/images/trash-many", {
      method: "POST",
      body: JSON.stringify({
        dataset_path: targetPathValue(),
        image_paths: imagePaths,
      }),
    });
    state.targetScan = payload.dataset;
    state.exportClassDetail = null;
    pruneExportState();
    resetDetailSelection();
    renderBrowser();
    renderExportExperience();
    if (classExists(state.selectedClass)) {
      await loadClassDetail(state.selectedClass);
    } else {
      state.classDetail = null;
      state.selectedClass = null;
      renderClassDetail();
      saveDraftState({ immediate: true });
    }
    setStatus(`Trashed ${formatNumber(payload.trashed || 0)} images`);
  } catch (error) {
    alert(error.message);
    setStatus("Trash failed");
  } finally {
    setBusy(false, state.preview ? "Preview loaded" : "Idle");
  }
}

async function loadExportClassDetail(className) {
  const datasetPath = targetPathValue();
  if (!datasetPath) {
    return;
  }

  setBusy(true, `Loading export ${className}`);
  try {
    state.exportSelectedClass = className;
    state.exportClassDetail = await api("/api/classes/detail", {
      method: "POST",
      body: JSON.stringify({
        dataset_path: datasetPath,
        class_name: className,
        limit_per_split: 1000,
      }),
    });
    renderExportExperience();
    saveDraftState({ immediate: true });
    setStatus(`Loaded export ${className}`);
  } catch (error) {
    alert(error.message);
    setStatus("Export load failed");
  } finally {
    setBusy(false, state.preview ? "Preview loaded" : "Idle");
  }
}

async function fetchExportClassDetailData(className) {
  return api("/api/classes/detail", {
    method: "POST",
    body: JSON.stringify({
      dataset_path: targetPathValue(),
      class_name: className,
      limit_per_split: 2000,
    }),
  });
}

function resetExportSelectionToAuto() {
  if (!state.exportSelectedClass || !state.exportClassDetail) {
    return;
  }
  const trainImages = trainSplitFromDetail(state.exportClassDetail).images;
  state.exportSelections[state.exportSelectedClass] = randomSelectionPathsFromImages(trainImages);
  renderExportExperience();
  saveDraftState();
}

function clearExportSelection() {
  if (!state.exportSelectedClass) {
    return;
  }
  state.exportSelections[state.exportSelectedClass] = [];
  renderExportExperience();
  saveDraftState();
}

function toggleExportImageSelection(imagePath) {
  if (!state.exportClassDetail) {
    return;
  }

  const className = state.exportClassDetail.class_name;
  const trainImages = trainSplitFromDetail(state.exportClassDetail).images;
  const selectedSet = new Set(effectiveExportSelectionPaths(className, trainImages));

  if (selectedSet.has(imagePath)) {
    selectedSet.delete(imagePath);
  } else {
    if (selectedSet.size >= state.exportPerClassLimit) {
      alert(`You can only pick up to ${state.exportPerClassLimit} images for this product.`);
      return;
    }
    selectedSet.add(imagePath);
  }

  const nextSelection = orderedSelectedPaths(trainImages, selectedSet);
  const autoSelection = autoSelectionPathsFromImages(trainImages);
  if (arraysEqual(nextSelection, autoSelection)) {
    delete state.exportSelections[className];
  } else {
    state.exportSelections[className] = nextSelection;
  }

  renderExportExperience();
  saveDraftState();
}

async function randomSelectAllExportClasses() {
  const classes = exportableClasses();
  if (!classes.length) {
    return;
  }

  setBusy(true, "Randomizing export selections");
  try {
    const nextSelections = {};
    for (let index = 0; index < classes.length; index += 1) {
      const item = classes[index];
      setStatus(`Randomizing ${index + 1}/${classes.length}`);
      const detail =
        state.exportSelectedClass === item.name && state.exportClassDetail
          ? state.exportClassDetail
          : await fetchExportClassDetailData(item.name);
      const trainImages = trainSplitFromDetail(detail).images;
      nextSelections[item.name] = randomSelectionPathsFromImages(trainImages);
      if (state.exportSelectedClass === item.name) {
        state.exportClassDetail = detail;
      }
    }
    state.exportSelections = nextSelections;
    renderExportExperience();
    saveDraftState({ immediate: true });
    setStatus("Random export picks ready");
  } catch (error) {
    alert(error.message);
    setStatus("Random export pick failed");
  } finally {
    setBusy(false, state.preview ? "Preview loaded" : "Idle");
  }
}

function clearAllExportSelections() {
  const classes = exportableClasses();
  if (!classes.length) {
    return;
  }

  const nextSelections = {};
  for (const item of classes) {
    nextSelections[item.name] = [];
  }
  state.exportSelections = nextSelections;
  renderExportExperience();
  saveDraftState({ immediate: true });
}

async function saveExportSelectionManifest() {
  const datasetPath = targetPathValue();
  const modelPath = exportModelPathValue();
  if (!datasetPath) {
    alert("Enter or scan a target dataset first.");
    return;
  }
  if (!modelPath) {
    alert("Enter an embedding model path first.");
    return;
  }

  setExportStatus(
    "running",
    "Saving selection snapshot",
    "Writing the current export picks into the dataset db folder.",
    `${datasetPath}/db`
  );
  setBusy(true, "Saving selection snapshot");
  try {
    const result = await api("/api/export/selection/save", {
      method: "POST",
      body: JSON.stringify({
        dataset_path: datasetPath,
        model_path: modelPath,
        output_filename: exportOutputFilenameValue(),
        per_class_limit: state.exportPerClassLimit,
        selected_paths_by_class: state.exportSelections,
      }),
    });
    setExportStatus(
      "success",
      "Selection snapshot saved",
      `${formatNumber(result.image_count)} picked images across ${formatNumber(result.class_count)} classes were saved.`,
      result.output_path
    );
    renderExportExperience();
    setStatus("Selection snapshot saved");
  } catch (error) {
    setExportStatus(
      "error",
      "Selection snapshot failed",
      "Dataset Studio could not save the current handpicked selection manifest.",
      error.message
    );
    renderExportExperience();
    setStatus("Selection snapshot failed");
  } finally {
    setBusy(false, state.preview ? "Preview loaded" : "Idle");
  }
}

async function exportTargetNpz() {
  const datasetPath = targetPathValue();
  const modelPath = exportModelPathValue();
  if (!datasetPath) {
    alert("Enter or scan a target dataset first.");
    return;
  }
  if (!modelPath) {
    alert("Enter an embedding model path first.");
    return;
  }
  if (missingExportClassCount() > 0) {
    alert("Some products still have zero selected train images. Fill those classes before exporting.");
    return;
  }

  setExportStatus(
    "running",
    "Exporting NPZ",
    "Embedding the selected train images and writing the gallery into the dataset db folder.",
    inferredExportOutputPath()
  );
  setBusy(true, "Exporting NPZ");
  try {
    state.exportResult = await api("/api/export/npz", {
      method: "POST",
      body: JSON.stringify({
        dataset_path: datasetPath,
        model_path: modelPath,
        output_filename: exportOutputFilenameValue(),
        per_class_limit: state.exportPerClassLimit,
        batch_size: 32,
        selected_paths_by_class: state.exportSelections,
      }),
    });
    setExportStatus(
      "success",
      "NPZ export complete",
      `${formatNumber(state.exportResult.image_count)} images across ${formatNumber(state.exportResult.class_count)} classes were embedded.`,
      `${state.exportResult.output_path}${state.exportResult.selection_manifest_path ? `\nSelection: ${state.exportResult.selection_manifest_path}` : ""}`
    );
    renderExportExperience();
    saveDraftState({ immediate: true });
    setStatus("NPZ exported");
  } catch (error) {
    setExportStatus(
      "error",
      "NPZ export failed",
      "Dataset Studio hit an error while generating the gallery file.",
      error.message
    );
    renderExportExperience();
    alert(error.message);
    setStatus("NPZ export failed");
  } finally {
    setBusy(false, state.preview ? "Preview loaded" : "Idle");
  }
}

async function renameSelectedClass() {
  if (!state.classDetail) {
    return;
  }

  const newName = el("rename-class-input").value.trim();
  if (!newName) {
    alert("Enter a new class name.");
    return;
  }

  setBusy(true, "Renaming class");
  try {
    const payload = await api("/api/classes/rename", {
      method: "POST",
      body: JSON.stringify({
        dataset_path: el("target-path").value.trim(),
        old_name: state.classDetail.class_name,
        new_name: newName,
      }),
    });
    if (Object.prototype.hasOwnProperty.call(state.exportSelections, state.classDetail.class_name)) {
      state.exportSelections[newName] = state.exportSelections[state.classDetail.class_name];
      delete state.exportSelections[state.classDetail.class_name];
    }
    if ((state.starredClasses || []).includes(state.classDetail.class_name)) {
      state.starredClasses = normalizedStarredClasses().filter((name) => name !== state.classDetail.class_name);
      state.starredClasses.push(newName);
      state.starredClasses = Array.from(new Set(state.starredClasses)).sort((left, right) => left.localeCompare(right));
    }
    if (state.exportSelectedClass === state.classDetail.class_name) {
      state.exportSelectedClass = newName;
      state.exportClassDetail = null;
    }
    state.targetScan = payload.dataset;
    state.selectedClass = newName;
    state.classDetailSplit = null;
    pruneExportState();
    await loadClassDetail(newName);
    renderExportExperience();
  } catch (error) {
    alert(error.message);
    setStatus("Rename failed");
  } finally {
    setBusy(false, state.preview ? "Preview loaded" : "Idle");
  }
}

async function moveImage(imagePath, targetClass, targetSplit) {
  if (!targetClass) {
    alert("Enter a target class name first.");
    return;
  }

  setBusy(true, "Moving image");
  try {
    const payload = await api("/api/images/reassign", {
      method: "POST",
      body: JSON.stringify({
        dataset_path: el("target-path").value.trim(),
        image_path: imagePath,
        target_class: targetClass,
        target_split: targetSplit,
      }),
    });
    state.targetScan = payload.dataset;
    state.exportClassDetail = null;
    pruneExportState();
    renderBrowser();
    renderExportExperience();
    const nextClass = classExists(state.selectedClass)
      ? state.selectedClass
      : classExists(targetClass)
        ? targetClass
        : null;
    if (nextClass) {
      await loadClassDetail(nextClass);
    } else {
      state.classDetail = null;
      state.selectedClass = null;
      renderClassDetail();
      saveDraftState({ immediate: true });
    }
  } catch (error) {
    alert(error.message);
    setStatus("Move failed");
  } finally {
    setBusy(false, state.preview ? "Preview loaded" : "Idle");
  }
}

async function trashImage(imagePath) {
  setBusy(true, "Trashing image");
  try {
    const payload = await api("/api/images/trash", {
      method: "POST",
      body: JSON.stringify({
        dataset_path: el("target-path").value.trim(),
        image_path: imagePath,
      }),
    });
    state.targetScan = payload.dataset;
    state.exportClassDetail = null;
    pruneExportState();
    renderBrowser();
    renderExportExperience();
    if (classExists(state.selectedClass)) {
      await loadClassDetail(state.selectedClass);
    } else {
      state.classDetail = null;
      state.selectedClass = null;
      renderClassDetail();
      saveDraftState({ immediate: true });
    }
  } catch (error) {
    alert(error.message);
    setStatus("Trash failed");
  } finally {
    setBusy(false, state.preview ? "Preview loaded" : "Idle");
  }
}

function wireEvents() {
  el("btn-refresh-discovery").addEventListener("click", refreshDiscovery);
  el("btn-scan-target").addEventListener("click", scanTarget);
  el("btn-preview-merge").addEventListener("click", buildMergePreview);
  el("btn-commit-merge").addEventListener("click", commitMerge);
  el("btn-default-mappings").addEventListener("click", buildMergePreview);
  el("btn-open-browser").addEventListener("click", () => switchPage("browse"));
  el("btn-open-export").addEventListener("click", () => switchPage("export"));
  el("btn-export-save-selection").addEventListener("click", saveExportSelectionManifest);
  el("btn-export-random-all").addEventListener("click", randomSelectAllExportClasses);
  el("btn-export-clear-all").addEventListener("click", clearAllExportSelections);
  el("btn-export-npz").addEventListener("click", exportTargetNpz);
  el("nav-merge").addEventListener("click", () => switchPage("merge"));
  el("nav-browse").addEventListener("click", () => switchPage("browse"));
  el("nav-export").addEventListener("click", () => switchPage("export"));
  el("export-detail-mode-all").addEventListener("click", () => {
    state.exportDetailMode = "all";
    renderExportClassDetail();
    saveDraftState();
  });
  el("export-detail-mode-selected").addEventListener("click", () => {
    state.exportDetailMode = "selected";
    renderExportClassDetail();
    saveDraftState();
  });
  el("merge-filter-all").addEventListener("click", () => {
    state.mergeFilter = "all";
    renderMergeTable();
    saveDraftState();
  });
  el("merge-filter-source-only").addEventListener("click", () => {
    state.mergeFilter = "source_only";
    renderMergeTable();
    saveDraftState();
  });
  el("merge-filter-existing").addEventListener("click", () => {
    state.mergeFilter = "existing";
    renderMergeTable();
    saveDraftState();
  });
  el("merge-filter-placeholders").addEventListener("click", () => {
    state.mergeFilter = "placeholders";
    renderMergeTable();
    saveDraftState();
  });
  el("class-search").addEventListener("input", () => {
    renderBrowser();
    saveDraftState();
  });
  el("filter-all-classes").addEventListener("click", () => {
    state.browserFilter = "all";
    renderBrowser();
    saveDraftState();
  });
  el("filter-merged-classes").addEventListener("click", () => {
    state.browserFilter = "merged";
    renderBrowser();
    saveDraftState();
  });
  el("filter-new-products").addEventListener("click", () => {
    state.browserFilter = "new_products";
    renderBrowser();
    saveDraftState();
  });
  el("filter-unknown-classes").addEventListener("click", () => {
    state.browserFilter = "unknowns";
    renderBrowser();
    saveDraftState();
  });
  el("filter-starred-classes").addEventListener("click", () => {
    state.browserFilter = "starred";
    renderBrowser();
    saveDraftState();
  });
  el("sort-size-desc").addEventListener("click", () => {
    state.browserSort = "size_desc";
    renderBrowser();
    saveDraftState();
  });
  el("sort-size-asc").addEventListener("click", () => {
    state.browserSort = "size_asc";
    renderBrowser();
    saveDraftState();
  });
  el("target-path").addEventListener("input", () => {
    state.exportResult = null;
    state.exportClassDetail = null;
    renderExportExperience();
    saveDraftState();
  });
  el("source-path").addEventListener("input", () => saveDraftState());
  el("export-model-path").addEventListener("input", () => {
    state.exportModelPath = exportModelPathValue();
    syncExportOutputFilenameAutoValue();
    state.exportResult = null;
    renderExportExperience();
    saveDraftState();
  });
  el("export-output-filename").addEventListener("input", () => {
    state.exportOutputFilename = exportOutputFilenameValue();
    state.exportResult = null;
    renderExportExperience();
    saveDraftState();
  });
  el("export-class-search").addEventListener("input", () => {
    state.exportSearch = el("export-class-search").value.trim();
    renderExportBrowser();
    saveDraftState();
  });
  el("export-filter-all-classes").addEventListener("click", () => {
    state.exportFilter = "all";
    renderExportBrowser();
    saveDraftState();
  });
  el("export-filter-merged-classes").addEventListener("click", () => {
    state.exportFilter = "merged";
    renderExportBrowser();
    saveDraftState();
  });
  el("export-filter-new-products").addEventListener("click", () => {
    state.exportFilter = "new_products";
    renderExportBrowser();
    saveDraftState();
  });
  el("export-filter-unknown-classes").addEventListener("click", () => {
    state.exportFilter = "unknowns";
    renderExportBrowser();
    saveDraftState();
  });
  el("export-filter-starred-classes").addEventListener("click", () => {
    state.exportFilter = "starred";
    renderExportBrowser();
    saveDraftState();
  });
  el("export-sort-size-desc").addEventListener("click", () => {
    state.exportSort = "size_desc";
    renderExportBrowser();
    saveDraftState();
  });
  el("export-sort-size-asc").addEventListener("click", () => {
    state.exportSort = "size_asc";
    renderExportBrowser();
    saveDraftState();
  });
  el("export-limit-20").addEventListener("click", () => setExportLimit(20));
  el("export-limit-40").addEventListener("click", () => setExportLimit(40));
}

async function init() {
  wireEvents();
  await loadCurrentSession();
  sessionBootstrapped = true;
  pruneStarredState();
  pruneExportState();
  renderPageNav();
  renderSummaryCards(state.preview?.source || null, state.targetScan || null, state.preview);
  renderMergeTable();
  renderBrowser();
  renderExportExperience();
  renderClassDetail();
  await refreshDiscovery();
  if (state.selectedClass && classExists(state.selectedClass)) {
    await loadClassDetail(state.selectedClass);
  }
  if (state.exportSelectedClass && exportClassExists(state.exportSelectedClass)) {
    await loadExportClassDetail(state.exportSelectedClass);
  }
}

init();
