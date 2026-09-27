(() => {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => [...document.querySelectorAll(sel)];

  const state = {
    sessionId: null,
    busy: false,
    lastEvidence: [],
    lastConsult: null,
    lastAnswer: "",
    lastQuestion: "",
    evIndex: 0,
    activePanel: "answer",
    ritualTimer: null,
  };

  const els = {
    progress: $("#progress"),
    welcomeStage: $("#welcomeStage"),
    resultStage: $("#resultStage"),
    input: $("#input"),
    inputFollow: $("#inputFollow"),
    btnSend: $("#btnSend"),
    btnSendFollow: $("#btnSendFollow"),
    btnNew: $("#btnNew"),
    btnAskAgain: $("#btnAskAgain"),
    btnHistory: $("#btnHistory"),
    btnCloseHistory: $("#btnCloseHistory"),
    btnCloseTrace: $("#btnCloseTrace"),
    status: $("#statusLine"),
    statusFollow: $("#statusLineFollow"),
    questionText: $("#questionText"),
    answerBody: $("#answerBody"),
    retrieveStamp: $("#retrieveStamp"),
    consultLead: null,
    evidenceView: $("#evidenceView"),
    evidenceCount: $("#evidenceCount"),
    sessionList: $("#sessionList"),
    historyDrawer: $("#historyDrawer"),
    drawerMask: $("#drawerMask"),
    traceDrawer: $("#traceDrawer"),
    traceTitle: $("#traceTitle"),
    traceMeta: $("#traceMeta"),
    traceQuote: $("#traceQuote"),
    tracePageImg: $("#tracePageImg"),
    tracePreviewEmpty: $("#tracePreviewEmpty"),
    traceOpenFull: $("#traceOpenFull"),
    fetchingVeil: $("#fetchingVeil"),
    fetchingTitle: $("#fetchingTitle"),
    fetchingSub: $("#fetchingSub"),
    stateAuthority: $("#stateAuthority"),
    stateApplicability: $("#stateApplicability"),
    stateConsistency: $("#stateConsistency"),
    numAuthority: $("#numAuthority"),
    numApplicability: $("#numApplicability"),
    numConsistency: $("#numConsistency"),
    ritualBar: $("#ritualBar"),
    ritualMsg: $("#ritualMsg"),
    sanzangDetail: $("#sanzangDetail"),
    sanzangDetailTitle: $("#sanzangDetailTitle"),
    sanzangDetailBody: $("#sanzangDetailBody"),
    sanzangDoneHint: $("#sanzangDoneHint"),
    decisionBanner: $("#decisionBanner"),
    answerScoreboard: $("#answerScoreboard"),
    scoreOverall: $("#scoreOverall"),
    scoreAuth: $("#scoreAuth"),
    scoreAppl: $("#scoreAppl"),
    scoreCons: $("#scoreCons"),
    sourceAsideTitle: $("#sourceAsideTitle"),
    sourceAsideMeta: $("#sourceAsideMeta"),
    sourceAsideSeals: $("#sourceAsideSeals"),
    sourcePageNum: $("#sourcePageNum"),
    btnBackAnswer: $("#btnBackAnswer"),
    btnPrevEv: $("#btnPrevEv"),
    btnNextEv: $("#btnNextEv"),
  };

  const RITUAL_COPY = {
    ask: ["已收到问题", "正在整理表述"],
    fetch: ["取经中……", "正在检索专科指南与说明书"],
    judge: ["辨经中……", "正在评估权威、适用与一致"],
    gain: ["得据中……", "正在整理可核对证据"],
    verify: ["核经中……", "正在准备原文定位"],
    done: ["辨经完成", "可以查看结论与原文"],
  };

  const SANZANG_EXPLAIN = {
    authority: {
      title: "权威藏",
      body: "看这条依据是否来自指南、共识或说明书等可核对文献。",
    },
    applicability: {
      title: "适用藏",
      body: "看证据是否真正对应你的问题情境，例如肾功能、年龄或用药条件。",
    },
    consistency: {
      title: "一致藏",
      body: "看不同证据是否相互支持，有没有明显冲突的说法。",
    },
  };

  function setProgress(pct) {
    els.progress.style.width = `${pct}%`;
  }

  function setStatus(text) {
    const t = text || "就绪";
    if (els.status) els.status.textContent = t;
    if (els.statusFollow) els.statusFollow.textContent = t;
  }

  function clearRitualTimer() {
    if (state.ritualTimer) {
      clearInterval(state.ritualTimer);
      state.ritualTimer = null;
    }
  }

  function setRitual(step, msg) {
    if (!els.ritualBar) return;
    els.ritualBar.hidden = false;
    const order = ["ask", "fetch", "judge", "gain", "verify"];
    const idx = order.indexOf(step);
    $$(".ritual-step").forEach((el) => {
      const i = order.indexOf(el.dataset.ritual);
      el.classList.toggle("active", el.dataset.ritual === step);
      el.classList.toggle("done", i >= 0 && i < idx);
    });
    const copy = RITUAL_COPY[step] || RITUAL_COPY.fetch;
    els.ritualMsg.textContent = msg || copy[0];
    if (els.fetchingTitle) els.fetchingTitle.textContent = copy[0];
    if (els.fetchingSub) els.fetchingSub.textContent = copy[1];
    $$(".veil-dot").forEach((d) => {
      const key = d.dataset.v;
      const ki = order.indexOf(key);
      d.classList.toggle("on", ki >= 0 && ki <= idx);
    });
  }

  function startRitualWhileWaiting() {
    clearRitualTimer();
    const steps = ["fetch", "judge", "gain", "verify"];
    let i = 0;
    setRitual("ask");
    setTimeout(() => setRitual(steps[0]), 40);
    state.ritualTimer = setInterval(() => {
      i = Math.min(i + 1, steps.length - 1);
      setRitual(steps[i]);
      if (i >= steps.length - 1) clearRitualTimer();
    }, 280);
  }

  function finishRitual() {
    clearRitualTimer();
    setRitual("verify", RITUAL_COPY.done[0]);
  }

  function setBusy(busy) {
    state.busy = busy;
    [els.btnSend, els.btnSendFollow, els.btnNew, els.input, els.inputFollow].forEach((el) => {
      if (el) el.disabled = busy;
    });
    els.fetchingVeil.hidden = !busy;
    if (busy) startRitualWhileWaiting();
    else finishRitual();
  }

  function showWelcome() {
    els.welcomeStage.classList.remove("hidden");
    els.resultStage.classList.add("hidden");
    if (els.ritualBar) els.ritualBar.hidden = true;
  }

  function showResult() {
    els.welcomeStage.classList.add("hidden");
    els.resultStage.classList.remove("hidden");
  }

  function focusPanel(name) {
    state.activePanel = name;
    const map = {
      answer: ["answer"],
      sanzang: ["sanzang"],
      fetch: ["fetch"],
      source: ["source"],
      question: ["question"],
    };
    const focus = new Set(map[name] || ["answer"]);
    $$(".path-node").forEach((n) => n.classList.toggle("active", n.dataset.panel === name));
    $$(".panel-card[data-view]").forEach((p) => {
      const on = focus.has(p.dataset.view);
      p.classList.toggle("is-focus", on);
      p.classList.toggle("is-dim", !on);
    });
    const target = $(`.panel-card[data-view="${[...focus][0]}"]`);
    if (target) target.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  async function api(path, options = {}) {
    let res;
    try {
      res = await fetch(path, {
        headers: { "Content-Type": "application/json", ...(options.headers || {}) },
        ...options,
      });
    } catch (_) {
      throw new Error("暂时连不上服务，请确认已启动网页。");
    }
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || res.statusText);
    }
    return res.json();
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /** 轻量 Markdown → HTML，避免原文 ** 星号乱码 */
  function formatAnswerHtml(raw) {
    const text = String(raw || "").replace(/\r\n/g, "\n").trim();
    if (!text) return "—";
    const lines = text.split("\n");
    const html = [];
    let listType = null;

    const flushList = () => {
      if (listType) {
        html.push(listType === "ol" ? "</ol>" : "</ul>");
        listType = null;
      }
    };

    const inline = (s) =>
      escapeHtml(s)
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/`([^`]+)`/g, "<code>$1</code>");

    for (const line of lines) {
      const t = line.trim();
      if (!t) {
        flushList();
        continue;
      }
      const heading = t.match(/^(#{1,3})\s+(.+)$/) || t.match(/^\*\*(.+?)\*\*\s*$/);
      if (heading) {
        flushList();
        const title = heading[2] != null ? heading[2] : heading[1];
        html.push(`<h3>${inline(title.replace(/\*\*/g, ""))}</h3>`);
        continue;
      }
      const ol = t.match(/^(\d+)[\.、]\s+(.+)$/);
      if (ol) {
        if (listType !== "ol") {
          flushList();
          html.push("<ol>");
          listType = "ol";
        }
        html.push(`<li>${inline(ol[2])}</li>`);
        continue;
      }
      const ul = t.match(/^[-•·]\s+(.+)$/);
      if (ul) {
        if (listType !== "ul") {
          flushList();
          html.push("<ul>");
          listType = "ul";
        }
        html.push(`<li>${inline(ul[1])}</li>`);
        continue;
      }
      flushList();
      html.push(`<p>${inline(t)}</p>`);
    }
    flushList();
    return html.join("") || "—";
  }

  function scoreOf(consult, key) {
    const sz = consult?.sanzang?.[key];
    if (sz && typeof sz.score === "number") return Math.round(sz.score);
    const board = consult?.answer_score;
    if (board && typeof board[key] === "number") return Math.round(board[key]);
    return null;
  }

  function renderScoreboard(consult) {
    const board = consult?.answer_score || {};
    const a = board.authority ?? scoreOf(consult, "authority");
    const u = board.applicability ?? scoreOf(consult, "applicability");
    const k = board.consistency ?? scoreOf(consult, "consistency");
    let overall = board.overall;
    if (overall == null && [a, u, k].every((x) => typeof x === "number")) {
      overall = Math.round((a + u + k) / 3);
    }
    const ready = [overall, a, u, k].some((x) => typeof x === "number");
    els.answerScoreboard.hidden = !ready;
    els.scoreOverall.textContent = overall != null ? String(overall) : "—";
    els.scoreAuth.textContent = a != null ? String(a) : "—";
    els.scoreAppl.textContent = u != null ? String(u) : "—";
    els.scoreCons.textContent = k != null ? String(k) : "—";
  }

  function renderSanzang(consult) {
    const keys = [
      ["authority", els.stateAuthority, els.numAuthority, "#sealAuthority"],
      ["applicability", els.stateApplicability, els.numApplicability, "#sealApplicability"],
      ["consistency", els.stateConsistency, els.numConsistency, "#sealConsistency"],
    ];
    keys.forEach(([key, stateEl, numEl, sel], idx) => {
      const conf = consult?.sanzang?.[key] || {};
      const node = $(sel);
      setTimeout(() => {
        const score = typeof conf.score === "number" ? Math.round(conf.score) : null;
        numEl.textContent = score != null ? String(score) : "—";
        stateEl.textContent = conf.text || "待评估";
        node.classList.toggle("on", !!conf.on);
        node.classList.toggle("warn", !!conf.warn);
        node.classList.remove("lit");
        void node.offsetWidth;
        if (conf.on) node.classList.add("lit");
      }, idx * 100);
    });
    const allOn = keys.every(([k]) => consult?.sanzang?.[k]?.on);
    els.sanzangDoneHint.textContent = allOn ? "三藏已辨" : "已评估";

    if (consult?.decision_label) {
      els.decisionBanner.hidden = false;
      els.decisionBanner.textContent = consult.decision_label;
    } else if (consult?.status_text) {
      els.decisionBanner.hidden = false;
      els.decisionBanner.textContent = consult.status_text;
    } else {
      els.decisionBanner.hidden = true;
    }
    renderScoreboard(consult);
  }

  function renderEvidenceCard(c, idx) {
    const title = escapeHtml(c.title || "未标注来源");
    const tier = escapeHtml(c.tier_label || c.tier || "资料");
    const body = escapeHtml(c.summary || c.content || "");
    const a = Math.round(c.authority_score ?? 0);
    const u = Math.round(c.applicability_score ?? 0);
    const k = Math.round(c.consistency_score ?? 0);
    const overall = Math.round(c.overall_score ?? Math.round((a + u + k) / 3));
    const canOpen = !!c.has_source_file && !!c.file_path;
    const openBtn = canOpen
      ? `<button type="button" class="btn-trace open-source" data-idx="${idx}">查看原文</button>`
      : `<span class="page-hint tip">暂无本地原文</span>`;
    return `<article class="evidence-card" data-idx="${idx}">
      <div class="ev-head">
        <span class="ev-idx">证据 ${String(c.rank || idx + 1).padStart(2, "0")}</span>
        <span class="ev-overall">综 ${overall}</span>
      </div>
      <div class="ev-title">${title}</div>
      <div class="ev-tier-row"><span class="ev-tier">${tier}</span></div>
      <div class="ev-quote">${body}</div>
      <div class="ev-scores">
        <div><span>权威</span><strong>${a}</strong></div>
        <div><span>适用</span><strong>${u}</strong></div>
        <div><span>一致</span><strong>${k}</strong></div>
      </div>
      <div class="ev-actions">${openBtn}</div>
    </article>`;
  }

  function renderEvidence(consult, evidence) {
    state.lastEvidence = evidence || [];
    state.lastConsult = consult;
    const list = evidence || [];
    if (!list.length) {
      els.evidenceCount.textContent = "暂无证据";
      els.evidenceView.innerHTML =
        consult && consult.retrieved === false
          ? '<p class="empty-state tip">本题未检索专科库，因此没有指南或说明书依据。</p>'
          : '<p class="empty-state tip">取证完成后，这里会显示依据。</p>';
      return;
    }
    state.evIndex = Math.min(state.evIndex, list.length - 1);
    showEvidenceAt(state.evIndex);
  }

  function showEvidenceAt(idx) {
    const list = state.lastEvidence || [];
    if (!list.length) return;
    state.evIndex = (idx + list.length) % list.length;
    const c = list[state.evIndex];
    els.evidenceView.innerHTML = renderEvidenceCard(c, state.evIndex);
    els.evidenceCount.textContent = `${state.evIndex + 1} / ${list.length}`;
    const card = els.evidenceView.querySelector(".evidence-card");
    card?.classList.add("active-ev");
    card?.addEventListener("click", (e) => {
      if (e.target.closest(".open-source")) return;
      card.classList.toggle("expanded");
    });
    els.evidenceView.querySelector(".open-source")?.addEventListener("click", (e) => {
      e.stopPropagation();
      openSource(c, e.currentTarget);
    });
  }

  function fillSourceAside(item) {
    if (!item) {
      els.sourceAsideTitle.textContent = "从证据中打开原文";
      els.sourceAsideMeta.innerHTML = "";
      els.sourceAsideSeals.innerHTML = "";
      return;
    }
    els.sourceAsideTitle.textContent = item.title || "证据";
    const a = Math.round(item.authority_score ?? 0);
    const u = Math.round(item.applicability_score ?? 0);
    const k = Math.round(item.consistency_score ?? 0);
    els.sourceAsideMeta.innerHTML = `
      <dt>来源</dt><dd>${escapeHtml(item.tier_label || item.tier || "资料")}</dd>
      <dt>评分</dt><dd>权威 ${a} · 适用 ${u} · 一致 ${k}</dd>
    `;
    els.sourceAsideSeals.innerHTML = `
      <span class="mini-seal ${a >= 55 ? "on" : ""}">权威 ${a}</span>
      <span class="mini-seal ${u >= 55 ? "on" : ""}">适用 ${u}</span>
      <span class="mini-seal ${k >= 55 ? "on" : ""}">一致 ${k}</span>
    `;
  }

  async function openSource(item, btn) {
    if (!item || !item.file_path) return;
    const old = btn.textContent;
    btn.disabled = true;
    btn.textContent = "定位中…";
    focusPanel("source");
    fillSourceAside(item);
    setStatus("正在定位原文…");
    setProgress(70);
    try {
      const data = await api("/api/source/open", {
        method: "POST",
        body: JSON.stringify({
          file_path: item.file_path,
          title: item.title || "",
          quote: item.quote || item.content || item.summary || "",
          chunk_id: item.id || "",
        }),
      });
      els.traceTitle.textContent = data.title || item.title || "原文";
      els.traceMeta.textContent = data.page
        ? `第 ${data.page} 页`
        : data.file_name || "—";
      els.traceQuote.textContent = data.quote || item.summary || "—";
      els.sourcePageNum.textContent = data.page ? `第 ${data.page} 页` : "原文页";
      if (data.page_img_url) {
        els.tracePageImg.src = data.page_img_url;
        els.tracePageImg.classList.remove("hidden");
        els.tracePreviewEmpty.classList.add("hidden");
      } else {
        els.tracePageImg.classList.add("hidden");
        els.tracePreviewEmpty.classList.remove("hidden");
        els.tracePreviewEmpty.innerHTML = '<p class="tip">已找到文件。若预览未显示，可打开完整页查看。</p>';
      }
      if (data.view_url) {
        els.traceOpenFull.href = data.view_url;
        els.traceOpenFull.style.display = "inline-flex";
      }
      setStatus("原文已定位");
      setProgress(100);
    } catch (err) {
      setStatus(err.message || "原文打开失败");
      els.tracePreviewEmpty.classList.remove("hidden");
      els.tracePreviewEmpty.innerHTML = `<p class="tip">${escapeHtml(err.message || "打开失败")}</p>`;
    } finally {
      btn.disabled = false;
      btn.textContent = old;
      setTimeout(() => setProgress(0), 400);
    }
  }

  function renderResultView({ question, answer, consult, evidence }) {
    state.lastQuestion = question || "";
    state.lastAnswer = answer || "";
    els.questionText.textContent = question || "—";
    els.answerBody.innerHTML = formatAnswerHtml(answer || "—");
    if (consult?.retrieved) {
      els.retrieveStamp.textContent = "已取证";
      els.retrieveStamp.classList.remove("warn");
    } else if (consult?.retrieved === false) {
      els.retrieveStamp.textContent = "未检索";
      els.retrieveStamp.classList.add("warn");
    } else {
      els.retrieveStamp.textContent = "已完成";
      els.retrieveStamp.classList.remove("warn");
    }
    renderSanzang(consult);
    renderEvidence(consult, evidence);
    showResult();
    focusPanel("answer");
  }

  async function ensureSession() {
    if (state.sessionId) return state.sessionId;
    const data = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ title: "新问询" }),
    });
    state.sessionId = data.id || data.session_id;
    return state.sessionId;
  }

  async function sendMessage(text, fromFollow) {
    const q = (text || "").trim();
    if (!q || state.busy) return;
    setBusy(true);
    setProgress(12);
    setStatus("取经中…");
    showResult();
    els.questionText.textContent = q;
    els.answerBody.textContent = "…";
    els.answerScoreboard.hidden = true;
    try {
      const sid = await ensureSession();
      setProgress(35);
      const data = await api("/api/chat", {
        method: "POST",
        body: JSON.stringify({ session_id: sid, message: q }),
      });
      state.sessionId = data.session_id || sid;
      setProgress(85);
      renderResultView({
        question: q,
        answer: data.answer,
        consult: data.consult,
        evidence: data.evidence,
      });
      setStatus("完成");
      setProgress(100);
      if (fromFollow && els.inputFollow) els.inputFollow.value = "";
      if (!fromFollow && els.input) els.input.value = "";
    } catch (err) {
      setStatus(err.message || "失败");
      els.answerBody.textContent = err.message || "请求失败";
    } finally {
      setBusy(false);
      setTimeout(() => setProgress(0), 500);
    }
  }

  async function refreshHistory() {
    const data = await api("/api/sessions");
    const list = data.sessions || [];
    els.sessionList.innerHTML = list
      .map(
        (s) => `<button type="button" class="session-item" data-id="${s.id}">
          <div class="t">${escapeHtml(s.title || "问询")}</div>
          <div class="m tip">${s.message_count || 0} 条记录</div>
        </button>`
      )
      .join("") || '<p class="empty-state tip">还没有历史记录</p>';
    els.sessionList.querySelectorAll(".session-item").forEach((btn) => {
      btn.addEventListener("click", async () => {
        state.sessionId = btn.dataset.id;
        closeDrawers();
        try {
          const detail = await api(`/api/sessions/${state.sessionId}`);
          const hist = detail.history || [];
          const lastUser = [...hist].reverse().find((m) => m.role === "user");
          const lastAsst = [...hist].reverse().find((m) => m.role === "assistant");
          renderResultView({
            question: lastUser?.content || detail.title || "—",
            answer: lastAsst?.content || detail.answer || "—",
            consult: detail.consult,
            evidence: detail.evidence,
          });
        } catch (err) {
          setStatus(err.message || "加载失败");
        }
      });
    });
  }

  function openHistory() {
    els.drawerMask.hidden = false;
    els.historyDrawer.classList.add("open");
    els.historyDrawer.setAttribute("aria-hidden", "false");
    refreshHistory().catch(() => {});
  }

  function closeDrawers() {
    els.drawerMask.hidden = true;
    els.historyDrawer.classList.remove("open");
    els.traceDrawer.classList.remove("open");
    els.historyDrawer.setAttribute("aria-hidden", "true");
    els.traceDrawer.setAttribute("aria-hidden", "true");
  }

  async function boot() {
    try {
      const h = await api("/api/health");
      setStatus(h.ok ? "就绪" : h.error || "服务未就绪");
    } catch (err) {
      setStatus(err.message || "服务未启动");
    }
  }

  els.btnSend?.addEventListener("click", () => sendMessage(els.input.value, false));
  els.btnSendFollow?.addEventListener("click", () => sendMessage(els.inputFollow.value, true));
  els.input?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) sendMessage(els.input.value, false);
  });
  els.btnNew?.addEventListener("click", () => {
    state.sessionId = null;
    state.evIndex = 0;
    showWelcome();
    setStatus("就绪");
  });
  els.btnAskAgain?.addEventListener("click", () => {
    state.sessionId = null;
    showWelcome();
  });
  els.btnHistory?.addEventListener("click", openHistory);
  els.btnCloseHistory?.addEventListener("click", closeDrawers);
  els.btnCloseTrace?.addEventListener("click", closeDrawers);
  els.drawerMask?.addEventListener("click", closeDrawers);
  els.btnBackAnswer?.addEventListener("click", () => focusPanel("answer"));
  els.btnPrevEv?.addEventListener("click", () => showEvidenceAt(state.evIndex - 1));
  els.btnNextEv?.addEventListener("click", () => showEvidenceAt(state.evIndex + 1));
  $$(".path-node").forEach((n) => n.addEventListener("click", () => focusPanel(n.dataset.panel)));

  $$(".sanzang-seal").forEach((seal) => {
    seal.addEventListener("click", () => {
      const key = seal.dataset.key;
      const info = SANZANG_EXPLAIN[key];
      const live = state.lastConsult?.sanzang?.[key];
      const score = typeof live?.score === "number" ? `当前评分：${Math.round(live.score)}。` : "";
      els.sanzangDetail.hidden = false;
      els.sanzangDetailTitle.textContent = info.title;
      els.sanzangDetailBody.textContent = `${info.body}${score ? "\n\n" + score : ""}${
        live?.text ? "\n" + live.text : ""
      }`;
      focusPanel("sanzang");
    });
  });

  $$(".example-card").forEach((card) => {
    card.addEventListener("click", () => {
      els.input.value = card.dataset.text || "";
      els.input.focus();
    });
  });

  boot();
})();
