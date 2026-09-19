"use strict";

// Document text is attacker-controlled: an ingested file can contain anything, including
// markup and instruction-like text. Every value that originates from a document is placed
// with textContent or createTextNode. innerHTML is never used with server data.

const $ = (id) => document.getElementById(id);

const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (response.status === 204) return null;
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      (body && (body.detail || body.error)) || `${response.status} ${response.statusText}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body;
}

function busy(button, on, label) {
  button.disabled = on;
  if (on) {
    button.dataset.label = button.textContent;
    button.textContent = "";
    button.append(el("span", "spinner"), document.createTextNode(" " + (label || "working…")));
  } else if (button.dataset.label) {
    button.textContent = button.dataset.label;
  }
}

function showError(message) {
  const out = $("output");
  out.replaceChildren();
  const banner = el("div", "banner error");
  banner.append(el("h3", null, "Request failed"), el("p", null, message));
  out.append(banner);
}

/* ---------------------------------------------------------------- status */

async function refreshStatus() {
  const bar = $("status");
  try {
    const r = await api("/ready");
    const pills = [
      [`${r.documents} docs · ${r.chunks} chunks`, ""],
      [`bm25 ${r.lexical_terms} terms`, r.ready ? "ok" : "bad"],
      [
        r.dense_ready ? `dense ${r.dense_vectors}×${r.dense_dimension}` : "dense unavailable",
        r.dense_ready ? "ok" : "bad",
      ],
      [r.embedding_model, ""],
      [`cfg ${r.config_fingerprint}`, ""],
    ];
    bar.replaceChildren(...pills.map(([t, c]) => el("span", "pill " + c, t)));
    if (!r.dense_ready && r.dense_error) bar.lastChild.title = r.dense_error;
  } catch (e) {
    bar.replaceChildren(el("span", "pill bad", "API unreachable"));
  }
}

/* ------------------------------------------------------------- documents */

async function refreshDocuments() {
  const list = $("documents");
  const docs = await api("/documents");
  $("doc-count").textContent = docs.length;

  if (!docs.length) {
    list.replaceChildren(el("p", "empty", "Nothing indexed yet."));
    return;
  }

  list.replaceChildren(
    ...docs.map((d) => {
      const card = el("div", "doc");
      card.append(el("div", "doc-title", d.title));
      card.append(
        el(
          "div",
          "doc-meta",
          `${d.original_filename} · ${d.media_type} · ${d.chunk_count} chunks · ${d.char_count} chars`
        )
      );
      card.append(el("div", "doc-meta", d.document_id.slice(0, 16)));

      const actions = el("div", "doc-actions");
      const view = el("button", "link", "View text");
      view.onclick = () => openDocument(d.document_id);

      const only = el("button", "link", "Search only this");
      only.onclick = () => {
        $("title-filter").value = d.title;
        $("query").focus();
      };

      const del = el("button", "danger", "Delete");
      del.onclick = async () => {
        if (!confirm(`Delete "${d.title}"? It will be removed from every index.`)) return;
        busy(del, true, "deleting");
        try {
          await api(`/documents/${d.document_id}`, { method: "DELETE" });
          // Clear the upload log: leaving "<file>: ingested" on screen after that same
          // document has been deleted reads as though it were still indexed.
          $("ingest-results").replaceChildren();
          await Promise.all([refreshDocuments(), refreshStatus()]);
        } catch (e) {
          busy(del, false);
          showError(e.message);
        }
      };

      actions.append(view, only, del);
      card.append(actions);
      return card;
    })
  );
}

async function openDocument(documentId, start, end) {
  const detail = await api(`/documents/${documentId}`);
  $("modal-title").textContent = detail.document.title;
  $("modal-meta").textContent =
    `${detail.document.original_filename} · ${detail.document.char_count} chars · ` +
    `${detail.document.chunk_count} chunks · ${detail.document.document_id}`;

  const target = $("modal-text");
  const text = detail.normalized_text;

  if (Number.isInteger(start) && Number.isInteger(end) && end > start && end <= text.length) {
    const mark = el("mark", null, text.slice(start, end));
    target.replaceChildren(
      document.createTextNode(text.slice(0, start)),
      mark,
      document.createTextNode(text.slice(end))
    );
    $("doc-modal").showModal();
    // Scroll the modal's own container, not the page: scrollIntoView walks every scrollable
    // ancestor and would drag the document behind the dialog out of position.
    const body = target.parentElement;
    body.scrollTop = Math.max(0, mark.offsetTop - body.clientHeight / 2);
  } else {
    target.replaceChildren(document.createTextNode(text));
    $("doc-modal").showModal();
  }
}

/* ---------------------------------------------------------------- ingest */

async function upload() {
  const input = $("files");
  const target = $("ingest-results");
  if (!input.files.length) {
    target.replaceChildren(el("div", "result-line warn", "Choose at least one file first."));
    return;
  }

  const form = new FormData();
  for (const file of input.files) form.append("files", file);

  const button = $("upload-btn");
  busy(button, true, "uploading");
  try {
    const result = await api("/documents", { method: "POST", body: form });
    target.replaceChildren(
      ...result.results.map((r) => {
        const cls =
          r.status === "ingested" ? "ok" : r.status === "rejected" ? "bad" : "warn";
        let message = `${r.filename}: ${r.status}`;
        if (r.status === "ingested") message += ` · ${r.chunk_count} chunks`;
        if (r.errors.length) message += ` · ${r.errors[0].message}`;
        return el("div", "result-line " + cls, message);
      })
    );
    input.value = "";
    await Promise.all([refreshDocuments(), refreshStatus()]);
  } catch (e) {
    target.replaceChildren(el("div", "result-line bad", e.message));
  } finally {
    busy(button, false);
  }
}

async function reindex() {
  const button = $("reindex-btn");
  busy(button, true, "rebuilding");
  try {
    const r = await api("/reindex", { method: "POST" });
    $("ingest-results").replaceChildren(
      el(
        "div",
        "result-line ok",
        `Rebuilt: ${r.lexical_chunks} chunks, ${r.lexical_terms} terms` +
          (r.dense_vectors ? `, ${r.dense_vectors} vectors @ ${r.dense_dimension}d` : "")
      )
    );
    await refreshStatus();
  } catch (e) {
    $("ingest-results").replaceChildren(el("div", "result-line bad", e.message));
  } finally {
    busy(button, false);
  }
}

/* --------------------------------------------------------------- queries */

function requestBody() {
  const filters = {};
  const title = $("title-filter").value.trim();
  const media = $("media-filter").value;
  if (title) filters.title_contains = title;
  if (media) filters.media_types = [media];

  return {
    text: $("query").value.trim(),
    mode: document.querySelector('input[name="mode"]:checked').value,
    top_k: Number($("topk").value),
    rrf_k: Number($("rrfk").value),
    filters: Object.keys(filters).length ? filters : null,
  };
}

function post(path, body) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function hitCard(hit) {
  const card = el("div", "hit");

  const head = el("div", "hit-head");
  head.append(el("span", "rank", "#" + hit.fused_rank), el("span", "hit-title", hit.title));
  if (hit.contains_instruction_like_text) {
    const badge = el("span", "badge warn", "instruction-like text");
    badge.title =
      "This passage contains text shaped like instructions to an assistant. " +
      "It is searchable, but it is never quoted as an answer.";
    head.append(badge);
  }
  head.append(el("span", "hit-score", "fused " + hit.fused_score.toFixed(6)));
  card.append(head);

  card.append(el("div", "snippet", hit.text));

  const table = el("table", "contrib");
  const header = el("tr");
  for (const label of ["retriever", "rank", "raw score", "RRF term 1/(k+rank)"]) {
    header.append(el("th", null, label));
  }
  table.append(header);

  for (const c of hit.contributions) {
    const row = el("tr");
    const tag = el("td");
    tag.append(el("span", "tag " + c.retriever, c.retriever));
    row.append(tag);
    row.append(el("td", null, "#" + c.rank));
    row.append(el("td", null, c.raw_score.toFixed(4)));
    row.append(el("td", null, c.rrf_term.toFixed(6)));
    table.append(row);
  }

  const total = el("tr", "total-row");
  total.append(el("td", null, "fused"), el("td", null, ""), el("td", null, ""));
  total.append(el("td", null, hit.fused_score.toFixed(6)));
  table.append(total);
  card.append(table);

  const open = el("button", "link", `Open source [${hit.start_offset}:${hit.end_offset}]`);
  open.style.marginTop = "9px";
  open.onclick = () => openDocument(hit.document_id, hit.start_offset, hit.end_offset);
  card.append(open);

  return card;
}

async function runSearch() {
  const body = requestBody();
  if (!body.text) return showError("Enter a query first.");

  const button = $("search-btn");
  busy(button, true, "searching");
  try {
    const r = await post("/search", body);
    const out = $("output");
    out.replaceChildren();

    const head = el("div", "results-head");
    head.append(el("strong", null, `${r.hit_count} result${r.hit_count === 1 ? "" : "s"}`));
    const parts = [`mode ${r.mode}`, `RRF k=${r.rrf_k}`];
    if (r.max_bm25_score !== null) parts.push(`best bm25 ${r.max_bm25_score.toFixed(4)}`);
    if (r.max_cosine_score !== null) parts.push(`best cosine ${r.max_cosine_score.toFixed(4)}`);
    head.append(el("span", "meta", parts.join(" · ")));
    out.append(head);

    if (!r.hit_count) {
      out.append(el("p", "empty", "No passage matched. Nothing was retrieved for this query."));
      return;
    }
    r.hits.forEach((hit) => out.append(hitCard(hit)));
  } catch (e) {
    showError(e.message);
  } finally {
    busy(button, false);
  }
}

async function runAnswer() {
  const body = requestBody();
  if (!body.text) return showError("Enter a question first.");

  const button = $("answer-btn");
  busy(button, true, "thinking");
  try {
    const r = await post("/answer", body);
    const out = $("output");
    out.replaceChildren();

    if (r.answered) {
      const banner = el("div", "banner answer");
      banner.append(el("h3", null, `Answer · ${r.provider} · ${r.mode}`));
      banner.append(el("p", null, r.text));
      out.append(banner);

      out.append(el("h2", null, `Citations (${r.citations.length})`));
      for (const c of r.citations) {
        const button2 = el("button", "citation");
        const head = el("div", "cite-head");
        head.append(
          el("span", "badge " + (c.validated ? "valid" : "invalid"), c.validated ? "verified" : "INVALID"),
          el("span", null, c.title)
        );
        if (c.locator) head.append(el("span", "badge", c.locator));
        head.append(el("span", null, `[${c.start_offset}:${c.end_offset}]`));
        button2.append(head, el("div", "cite-snippet", c.snippet));
        button2.onclick = () => openDocument(c.document_id, c.start_offset, c.end_offset);
        out.append(button2);
      }
    } else {
      const banner = el("div", "banner abstain");
      banner.append(el("h3", null, "Abstained · " + r.abstention.reason.replace(/_/g, " ")));
      banner.append(el("p", null, r.abstention.explanation));
      banner.append(
        el("p", null, `Evidence considered: ${r.abstention.evidence_considered} passage(s).`)
      );
      // conflict_note is already folded into the explanation for CONFLICTING_EVIDENCE;
      // only surface it separately when it would otherwise be lost.
      if (r.conflict_detected && r.conflict_note && !r.abstention.explanation.includes(r.conflict_note)) {
        banner.append(el("p", null, r.conflict_note));
      }
      out.append(banner);
    }

    if (r.evidence.length) {
      out.append(el("h2", null, `Evidence considered (${r.evidence.length})`));
      r.evidence.forEach((hit) => out.append(hitCard(hit)));
    }
  } catch (e) {
    showError(e.message);
  } finally {
    busy(button, false);
  }
}

/* ------------------------------------------------------------------ wire */

$("upload-btn").onclick = upload;
$("reindex-btn").onclick = reindex;
$("search-btn").onclick = runSearch;
$("answer-btn").onclick = runAnswer;
$("modal-close").onclick = () => $("doc-modal").close();

$("query").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) runAnswer();
});

refreshStatus();
refreshDocuments().catch((e) => showError(e.message));
