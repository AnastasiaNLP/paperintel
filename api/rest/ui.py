PREVIEW_UI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PaperIntel</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d8dee8;
      --accent: #1769e0;
      --accent-strong: #0f4fac;
      --good: #12805c;
      --bad: #b42318;
      --warn: #946200;
      --shadow: 0 12px 32px rgba(20, 32, 50, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 15px;
    }

    button, input, select, textarea {
      font: inherit;
    }

    button {
      min-height: 38px;
      border: 1px solid transparent;
      border-radius: 7px;
      padding: 8px 12px;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      font-weight: 650;
    }

    button.secondary {
      background: #fff;
      color: var(--ink);
      border-color: var(--line);
    }

    button:disabled {
      opacity: 0.55;
      cursor: wait;
    }

    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 10px 11px;
      background: #fff;
      color: var(--ink);
      outline: none;
    }

    input.invalid, select.invalid, textarea.invalid {
      border-color: var(--bad);
      box-shadow: 0 0 0 3px rgba(180, 35, 24, 0.12);
    }

    textarea {
      min-height: 86px;
      resize: vertical;
    }

    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0;
      text-transform: uppercase;
    }

    .shell {
      display: grid;
      grid-template-columns: 380px minmax(0, 1fr);
      min-height: 100vh;
    }

    .sidebar {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 20px;
      overflow: auto;
    }

    .main {
      padding: 22px;
      overflow: auto;
    }

    .brand {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 18px;
    }

    .brand h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.15;
    }

    .status {
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 4px 9px;
      border-radius: 999px;
      background: #eef4ff;
      color: #174ea6;
      font-size: 12px;
      font-weight: 750;
      white-space: nowrap;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
      margin-bottom: 14px;
    }

    .panel h2 {
      margin: 0 0 13px;
      font-size: 15px;
      line-height: 1.2;
    }

    .grid {
      display: grid;
      gap: 12px;
    }

    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }

    .session-meta {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
      word-break: break-word;
    }

    .workspace-list {
      display: grid;
      gap: 8px;
    }

    .workspace-item {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 9px;
      background: #fbfcfe;
      cursor: pointer;
    }

    .workspace-item strong {
      display: block;
      font-size: 13px;
      margin-bottom: 4px;
    }

    .tabs {
      display: flex;
      gap: 8px;
      margin-bottom: 14px;
    }

    .tabs button {
      background: #fff;
      color: var(--ink);
      border-color: var(--line);
    }

    .tabs button.active {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }

    .view {
      display: none;
    }

    .view.active {
      display: grid;
      gap: 14px;
    }

    .output {
      min-height: 420px;
      white-space: pre-wrap;
      line-height: 1.52;
    }

    .output h3 {
      margin: 0 0 10px;
      font-size: 16px;
    }

    .payload {
      background: #0b1020;
      color: #dbe7ff;
      border-radius: 7px;
      padding: 12px;
      overflow: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
      max-height: 320px;
    }

    .message {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px;
      background: #fff;
    }

    .message small {
      display: block;
      color: var(--muted);
      margin-bottom: 8px;
      font-weight: 700;
    }

    .citations {
      display: grid;
      gap: 8px;
      margin-top: 12px;
    }

    .citation {
      border-left: 3px solid var(--accent);
      padding: 8px 10px;
      background: #f7faff;
      color: #243447;
      font-size: 13px;
      white-space: pre-wrap;
    }

    .notice {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }

    .error {
      color: var(--bad);
      font-weight: 700;
    }

    .ok {
      color: var(--good);
      font-weight: 700;
    }

    @media (max-width: 920px) {
      .shell {
        grid-template-columns: 1fr;
      }

      .sidebar {
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }

      .row {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand">
        <h1>PaperIntel</h1>
        <span id="healthBadge" class="status">checking</span>
      </div>

      <section class="panel grid">
        <h2>Access</h2>
        <label>Preview token
          <input id="tokenInput" type="password" autocomplete="off" placeholder="Bearer token">
        </label>
        <div class="actions">
          <button id="saveTokenBtn" class="secondary" type="button">Save token</button>
          <button id="clearTokenBtn" class="secondary" type="button">Clear</button>
        </div>
      </section>

      <section class="panel grid">
        <h2>Session</h2>
        <div class="row">
          <label>Persona
            <select id="personaInput">
              <option value="engineer">Engineer</option>
              <option value="researcher">Researcher</option>
              <option value="techlead">Tech Lead</option>
            </select>
          </label>
          <label>Session ID
            <input id="sessionIdInput" autocomplete="off" placeholder="session id">
          </label>
        </div>
        <label>Initial query
          <input id="initialQueryInput" autocomplete="off" placeholder="optional">
        </label>
        <div class="actions">
          <button id="createSessionBtn" type="button">New session</button>
          <button id="loadSessionBtn" class="secondary" type="button">Load</button>
        </div>
        <div id="sessionMeta" class="session-meta"></div>
      </section>

      <section class="panel grid">
        <h2>Workspace</h2>
        <div class="actions">
          <button id="refreshWorkspacesBtn" class="secondary" type="button">Refresh</button>
          <button id="loadComparisonBtn" class="secondary" type="button">Latest comparison</button>
        </div>
        <div id="workspaceList" class="workspace-list notice">No active session.</div>
      </section>
    </aside>

    <main class="main">
      <div class="tabs">
        <button class="active" data-view="analyzeView" type="button">Analyze</button>
        <button data-view="discoverView" type="button">Discover</button>
        <button data-view="askView" type="button">Ask</button>
        <button data-view="synthesisView" type="button">Compare</button>
      </div>

      <section id="analyzeView" class="view active">
        <div class="panel grid">
          <h2>Analyze Paper</h2>
          <label>arXiv URL
            <input id="paperUrlInput" autocomplete="off" value="https://arxiv.org/abs/1706.03762">
          </label>
          <div class="actions">
            <button id="analyzeUrlBtn" type="button">Analyze URL</button>
            <button id="queueUrlBtn" class="secondary" type="button">Queue URL</button>
          </div>
        </div>
        <div class="panel grid">
          <h2>Upload PDF</h2>
          <div class="row">
            <label>Paper ID
              <input id="pdfPaperIdInput" autocomplete="off" placeholder="optional">
            </label>
            <label>PDF file
              <input id="pdfFileInput" type="file" accept="application/pdf">
            </label>
          </div>
          <div class="actions">
            <button id="analyzePdfBtn" type="button">Analyze PDF</button>
            <button id="queuePdfBtn" class="secondary" type="button">Queue PDF</button>
          </div>
        </div>
      </section>

      <section id="discoverView" class="view">
        <div class="panel grid">
          <h2>Discover Papers</h2>
          <label>Topic
            <textarea id="topicInput">Find recent papers about retrieval augmented generation</textarea>
          </label>
          <div class="actions">
            <button id="discoverBtn" type="button">Discover</button>
          </div>
        </div>
        <div class="panel grid">
          <h2>Selection</h2>
          <label>Selection
            <input id="selectionInput" autocomplete="off" value="use 1">
          </label>
          <div class="actions">
            <button id="selectBtn" type="button">Select</button>
            <button id="analyzeSelectedBtn" type="button">Analyze selected</button>
            <button id="queueSelectedBtn" class="secondary" type="button">Queue selected</button>
          </div>
        </div>
      </section>

      <section id="askView" class="view">
        <div class="panel grid">
          <h2>Question</h2>
          <label>Prompt
            <textarea id="questionInput">What is the main contribution of this paper?</textarea>
          </label>
          <div class="actions">
            <button id="askBtn" type="button">Ask</button>
          </div>
        </div>
      </section>

      <section id="synthesisView" class="view">
        <div class="panel grid">
          <h2>Comparison And Synthesis</h2>
          <label>Prompt
            <textarea id="comparePromptInput">Compare implementation trade-offs across active papers.</textarea>
          </label>
          <div class="actions">
            <button id="compareBtn" type="button">Compare</button>
            <button id="synthesizeBtn" type="button">Synthesize</button>
          </div>
        </div>
      </section>

      <section class="panel output">
        <h3>Result</h3>
        <div id="output" class="notice">Ready.</div>
      </section>
    </main>
  </div>

  <script>
    const state = {
      session: null,
      busy: false,
    };

    const els = {
      healthBadge: document.getElementById("healthBadge"),
      tokenInput: document.getElementById("tokenInput"),
      personaInput: document.getElementById("personaInput"),
      sessionIdInput: document.getElementById("sessionIdInput"),
      initialQueryInput: document.getElementById("initialQueryInput"),
      sessionMeta: document.getElementById("sessionMeta"),
      workspaceList: document.getElementById("workspaceList"),
      output: document.getElementById("output"),
      paperUrlInput: document.getElementById("paperUrlInput"),
      pdfPaperIdInput: document.getElementById("pdfPaperIdInput"),
      pdfFileInput: document.getElementById("pdfFileInput"),
      topicInput: document.getElementById("topicInput"),
      selectionInput: document.getElementById("selectionInput"),
      questionInput: document.getElementById("questionInput"),
      comparePromptInput: document.getElementById("comparePromptInput"),
    };

    const token = localStorage.getItem("paperintel.previewToken") || "";
    els.tokenInput.value = token;

    function sessionId() {
      return els.sessionIdInput.value.trim();
    }

    function markInvalid(field, message) {
      if (field) {
        field.classList.add("invalid");
        field.focus();
      }
      throw new Error(message);
    }

    function clearInvalid(field) {
      if (field) field.classList.remove("invalid");
    }

    function requireSession() {
      const value = sessionId();
      if (!value) {
        markInvalid(els.sessionIdInput, "Create or load a session first.");
      }
      if (value.length > 120) {
        markInvalid(els.sessionIdInput, "Session ID is too long.");
      }
      clearInvalid(els.sessionIdInput);
      return value;
    }

    function requireText(field, label, maxLength) {
      const value = field.value.trim();
      if (!value) {
        markInvalid(field, `${label} is required.`);
      }
      if (maxLength && value.length > maxLength) {
        markInvalid(field, `${label} must be ${maxLength} characters or less.`);
      }
      clearInvalid(field);
      return value;
    }

    function optionalText(field, label, maxLength) {
      const value = field.value.trim();
      if (maxLength && value.length > maxLength) {
        markInvalid(field, `${label} must be ${maxLength} characters or less.`);
      }
      clearInvalid(field);
      return value;
    }

    function requireArxivUrl() {
      const value = requireText(els.paperUrlInput, "Paper URL", 500);
      let parsed;
      try {
        parsed = new URL(value);
      } catch {
        markInvalid(els.paperUrlInput, "Paper URL must be a valid URL.");
      }
      if (!["http:", "https:"].includes(parsed.protocol)) {
        markInvalid(els.paperUrlInput, "Paper URL must use http or https.");
      }
      if (!/arxiv\\.org$/i.test(parsed.hostname) && !/\\.arxiv\\.org$/i.test(parsed.hostname)) {
        markInvalid(els.paperUrlInput, "Use an arXiv URL for URL analysis.");
      }
      clearInvalid(els.paperUrlInput);
      return value;
    }

    function requirePdfFile() {
      const file = els.pdfFileInput.files[0];
      if (!file) {
        markInvalid(els.pdfFileInput, "Choose a PDF file.");
      }
      if (file.type && file.type !== "application/pdf") {
        markInvalid(els.pdfFileInput, "PDF upload must use application/pdf.");
      }
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        markInvalid(els.pdfFileInput, "Choose a file with a .pdf extension.");
      }
      if (file.size > 50 * 1024 * 1024) {
        markInvalid(els.pdfFileInput, "PDF must be 50 MB or smaller.");
      }
      clearInvalid(els.pdfFileInput);
      return file;
    }

    function authHeaders(extra = {}) {
      const headers = { ...extra };
      const tokenValue = els.tokenInput.value.trim();
      if (tokenValue) headers.authorization = `Bearer ${tokenValue}`;
      return headers;
    }

    function setBusy(isBusy) {
      state.busy = isBusy;
      document.querySelectorAll("button").forEach((button) => {
        button.disabled = isBusy;
      });
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function renderJson(value) {
      return `<pre class="payload">${escapeHtml(JSON.stringify(value, null, 2))}</pre>`;
    }

    function renderMessage(payload, title = "Response") {
      const text = escapeHtml(payload.response_text || payload.comparison_markdown || "");
      const citations = (payload.citations || []).map((citation) => {
        const label = citation.chunk_id || citation.paper_id || citation.source || "citation";
        const quote = citation.text || citation.quote || citation.snippet || "";
        return `<div class="citation"><strong>${escapeHtml(label)}</strong>\\n${escapeHtml(quote || JSON.stringify(citation))}</div>`;
      }).join("");
      const meta = [
        payload.intent ? `intent=${payload.intent}` : "",
        payload.phase ? `phase=${payload.phase}` : "",
        payload.referenced_paper_ids?.length ? `papers=${payload.referenced_paper_ids.join(", ")}` : "",
      ].filter(Boolean).join(" · ");
      els.output.innerHTML = `
        <div class="message">
          <small>${escapeHtml(title)}${meta ? " · " + escapeHtml(meta) : ""}</small>
          <div>${text}</div>
          ${citations ? `<div class="citations">${citations}</div>` : ""}
        </div>
        ${renderJson(payload)}
      `;
    }

    function renderSession(session) {
      state.session = session;
      els.sessionIdInput.value = session.id;
      els.sessionMeta.innerHTML = `
        <span><strong>ID:</strong> ${escapeHtml(session.id)}</span>
        <span><strong>Persona:</strong> ${escapeHtml(session.persona)}</span>
        <span><strong>Phase:</strong> ${escapeHtml(session.phase)}</span>
        <span><strong>Active papers:</strong> ${escapeHtml((session.active_paper_ids || []).join(", ") || "none")}</span>
      `;
    }

    function renderWorkspaces(payload) {
      const workspaces = payload.workspaces || [];
      if (!workspaces.length) {
        els.workspaceList.textContent = "No workspaces.";
        return;
      }
      els.workspaceList.innerHTML = workspaces.map((workspace) => `
        <div class="workspace-item" data-paper-id="${escapeHtml(workspace.paper_id)}">
          <strong>${escapeHtml(workspace.title || workspace.paper_id)}</strong>
          <span>${escapeHtml(workspace.paper_id)} · ${escapeHtml(workspace.pipeline_stage)} · benchmarks=${workspace.benchmark_count}</span>
        </div>
      `).join("");
    }

    async function request(path, options = {}) {
      const init = {
        ...options,
        headers: authHeaders(options.headers || {}),
      };
      const response = await fetch(path, init);
      const contentType = response.headers.get("content-type") || "";
      const body = contentType.includes("application/json")
        ? await response.json()
        : await response.text();
      if (!response.ok) {
        const detail = typeof body === "string" ? body : body.detail || body.error || JSON.stringify(body);
        throw new Error(`${response.status} ${detail}`);
      }
      return body;
    }

    async function runJobStatus(job) {
      const startedAt = Date.now();
      while (Date.now() - startedAt < 30 * 60 * 1000) {
        els.output.innerHTML = `<span class="notice">Job ${escapeHtml(job.id)} · ${escapeHtml(job.status)}...</span>`;
        if (job.status === "succeeded") {
          await refreshSession().catch(() => null);
          await refreshWorkspaces().catch(() => null);
          return job.result_json || job;
        }
        if (["failed", "canceled"].includes(job.status)) {
          const detail = job.error_json ? JSON.stringify(job.error_json) : job.status;
          throw new Error(`Job ${job.status}: ${detail}`);
        }
        await new Promise((resolve) => window.setTimeout(resolve, 2000));
        job = await request(`/jobs/${encodeURIComponent(job.id)}`);
      }
      throw new Error(`Job ${job.id} is still running. Reload its status from the Jobs endpoint.`);
    }

    async function runJob(path, body = null) {
      const options = { method: "POST" };
      if (body !== null) {
        options.headers = { "content-type": "application/json" };
        options.body = JSON.stringify(body);
      }
      return runJobStatus(await request(path, options));
    }

    async function run(label, task) {
      setBusy(true);
      els.output.innerHTML = `<span class="notice">${escapeHtml(label)}...</span>`;
      try {
        const payload = await task();
        if (payload && payload.response_text !== undefined) {
          renderMessage(payload, label);
        } else {
          els.output.innerHTML = renderJson(payload);
        }
        return payload;
      } catch (error) {
        els.output.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
      } finally {
        setBusy(false);
      }
    }

    async function refreshSession() {
      if (!sessionId()) return;
      const session = await request(`/sessions/${encodeURIComponent(sessionId())}`);
      renderSession(session);
      return session;
    }

    async function refreshWorkspaces() {
      if (!sessionId()) throw new Error("Create or load a session first.");
      const payload = await request(`/sessions/${encodeURIComponent(sessionId())}/workspaces`);
      renderWorkspaces(payload);
      return payload;
    }

    document.querySelectorAll("[data-view]").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll("[data-view]").forEach((tab) => tab.classList.remove("active"));
        document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(button.dataset.view).classList.add("active");
      });
    });

    document.getElementById("saveTokenBtn").addEventListener("click", () => {
      optionalText(els.tokenInput, "Preview token", 500);
      localStorage.setItem("paperintel.previewToken", els.tokenInput.value.trim());
      els.output.innerHTML = `<span class="ok">Token saved.</span>`;
    });

    document.getElementById("clearTokenBtn").addEventListener("click", () => {
      localStorage.removeItem("paperintel.previewToken");
      els.tokenInput.value = "";
      els.output.innerHTML = `<span class="ok">Token cleared.</span>`;
    });

    document.getElementById("createSessionBtn").addEventListener("click", () => run("Creating session", async () => {
      const originalQuery = optionalText(els.initialQueryInput, "Initial query", 500);
      const payload = await request("/sessions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          persona: els.personaInput.value,
          original_query: originalQuery || null,
        }),
      });
      renderSession(payload);
      await refreshWorkspaces().catch(() => null);
      return payload;
    }));

    document.getElementById("loadSessionBtn").addEventListener("click", () => run("Loading session", refreshSession));
    document.getElementById("refreshWorkspacesBtn").addEventListener("click", () => run("Loading workspaces", refreshWorkspaces));

    document.getElementById("analyzeUrlBtn").addEventListener("click", () => run("Analyzing URL", async () => {
      const id = requireSession();
      const paperUrl = requireArxivUrl();
      return runJob(`/sessions/${encodeURIComponent(id)}/jobs/analyze-paper`, { paper_url: paperUrl });
    }));

    document.getElementById("queueUrlBtn").addEventListener("click", () => run("Queueing URL", async () => {
      const id = requireSession();
      const paperUrl = requireArxivUrl();
      return request(`/sessions/${encodeURIComponent(id)}/jobs/analyze-paper`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ paper_url: paperUrl }),
      });
    }));

    document.getElementById("analyzePdfBtn").addEventListener("click", () => run("Analyzing PDF", async () => {
      const id = requireSession();
      const file = requirePdfFile();
      const paperId = optionalText(els.pdfPaperIdInput, "Paper ID", 500);
      const form = new FormData();
      form.append("file", file);
      if (paperId) form.append("paper_id", paperId);
      form.append("skip_arxiv_metadata_fetch", "true");
      const job = await request(`/sessions/${encodeURIComponent(id)}/jobs/analyze-pdf`, {
        method: "POST",
        body: form,
      });
      return runJobStatus(job);
    }));

    document.getElementById("queuePdfBtn").addEventListener("click", () => run("Queueing PDF", async () => {
      const id = requireSession();
      const file = requirePdfFile();
      const paperId = optionalText(els.pdfPaperIdInput, "Paper ID", 500);
      const form = new FormData();
      form.append("file", file);
      if (paperId) form.append("paper_id", paperId);
      form.append("skip_arxiv_metadata_fetch", "true");
      return request(`/sessions/${encodeURIComponent(id)}/jobs/analyze-pdf`, {
        method: "POST",
        body: form,
      });
    }));

    document.getElementById("discoverBtn").addEventListener("click", () => run("Discovering papers", async () => {
      const id = requireSession();
      const topic = requireText(els.topicInput, "Topic", 500);
      return runJob(`/sessions/${encodeURIComponent(id)}/jobs/discover`, { topic });
    }));

    document.getElementById("selectBtn").addEventListener("click", () => run("Selecting papers", async () => {
      const id = requireSession();
      const selection = requireText(els.selectionInput, "Selection", 500);
      const payload = await request(`/sessions/${encodeURIComponent(id)}/select`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ selection }),
      });
      await refreshSession().catch(() => null);
      return payload;
    }));

    document.getElementById("analyzeSelectedBtn").addEventListener("click", () => run("Analyzing selected papers", async () => {
      const id = requireSession();
      return runJob(`/sessions/${encodeURIComponent(id)}/jobs/analyze-selected`);
    }));

    document.getElementById("queueSelectedBtn").addEventListener("click", () => run("Queueing selected papers", async () => {
      const id = requireSession();
      return request(`/sessions/${encodeURIComponent(id)}/jobs/analyze-selected`, { method: "POST" });
    }));

    document.getElementById("askBtn").addEventListener("click", () => run("Asking", async () => {
      const id = requireSession();
      const question = requireText(els.questionInput, "Question", 2000);
      return request(`/sessions/${encodeURIComponent(id)}/ask`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question }),
      });
    }));

    document.getElementById("compareBtn").addEventListener("click", () => run("Comparing papers", async () => {
      const id = requireSession();
      const prompt = optionalText(els.comparePromptInput, "Prompt", 2000);
      return runJob(`/sessions/${encodeURIComponent(id)}/jobs/compare`, { prompt: prompt || null });
    }));

    document.getElementById("synthesizeBtn").addEventListener("click", () => run("Synthesizing", async () => {
      const id = requireSession();
      const prompt = optionalText(els.comparePromptInput, "Prompt", 2000);
      return runJob(`/sessions/${encodeURIComponent(id)}/jobs/synthesize`, { prompt: prompt || null });
    }));

    document.getElementById("loadComparisonBtn").addEventListener("click", () => run("Loading comparison", async () => {
      const id = requireSession();
      return request(`/sessions/${encodeURIComponent(id)}/comparison`);
    }));

    els.workspaceList.addEventListener("click", (event) => {
      const item = event.target.closest(".workspace-item");
      if (!item || !sessionId()) return;
      const paperId = item.dataset.paperId;
      run("Loading workspace", () => request(`/sessions/${encodeURIComponent(sessionId())}/workspaces/${encodeURIComponent(paperId)}`));
    });

    request("/health")
      .then((payload) => {
        els.healthBadge.textContent = payload.status || "healthy";
        els.healthBadge.classList.add("ok");
      })
      .catch(() => {
        els.healthBadge.textContent = "degraded";
        els.healthBadge.classList.add("error");
      });
  </script>
</body>
</html>
"""
