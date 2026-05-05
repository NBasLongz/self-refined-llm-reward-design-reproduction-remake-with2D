/* Self-Refine-RL frontend wired to the Flask backend. */

const $ = (id) => document.getElementById(id);
const cv = $("mazeCanvas");
const cx = cv.getContext("2d");

let CFG = {
  default_episodes: 350,
  default_eval_trials: 100,
  max_refinement_iterations: 5,
  success_threshold: 0.95,
};
let maze = null;
let bot = { x: 0, y: 0 };
let trail = [];
let paused = false;
let stopped = false;
let speed = 1;
let timer = null;
let seconds = 0;
let lastMetrics = null;
let previousCode = "";
let tableState = { initial: null, refined: null, manual: null, iterations: 0 };
let selectedTaskId = "lava_maze";
let allTasks = [];
let boardYaw = 0;
let draggingBoard = false;
let dragStartX = 0;
let dragStartYaw = 0;

function pct(value) {
  if (value === null || value === undefined) return "-";
  return `${Math.round(value * 100)}%`;
}

function fmtSec(s) {
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${r < 10 ? "0" : ""}${r}`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitIfPaused() {
  while (paused && !stopped) {
    await sleep(80);
  }
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${path} failed (${res.status}): ${text}`);
  }
  return res.json();
}

function taskQuery() {
  return `task_id=${encodeURIComponent(selectedTaskId)}`;
}

function togTh() {
  drawMaze();
}

function colors() {
  return {
    pathTop: "#f8fbff",
    pathEdge: "#d7e2f1",
    pathLine: "rgba(122,145,176,.34)",
    wallTop: "#aeb8c8",
    wallLeft: "#6e7a8d",
    wallRight: "#8793a6",
    wallHi: "#c9d3e2",
    lavaTop: "#ffd6dd",
    lavaCore: "#ef476f",
    lavaGlow: "rgba(239,71,111,.34)",
    goalTop: "#c9f3df",
    goalLeft: "#48b98b",
    goalRight: "#6fd2a5",
    goalText: "#087456",
    bot: "#2563eb",
    botSide: "#1746b3",
    botLight: "#dbeafe",
    shadow: "rgba(35,50,75,.18)",
    trail: "rgba(37,99,235,.18)",
    trailDot: "rgba(37,99,235,.22)",
    trailActive: "rgba(37,99,235,.38)",
    outline: "rgba(31,48,77,.12)",
    boardSideL: "#d3deec",
    boardSideR: "#c0cedf",
    boardTop: "rgba(255,255,255,.58)",
  };
}

function posKey(pos) {
  return `${pos[0]},${pos[1]}`;
}

function setStatus(text, cls, progress, tag = text) {
  const pill = $("sPill");
  pill.innerHTML = `<span class="d"></span>${text}`;
  pill.className = `sp s${cls}`;
  $("inTag").textContent = tag;
  const fill = $("pFill");
  fill.style.width = `${progress}%`;
  fill.style.background =
    cls === "w" ? "var(--warn)" : cls === "e" ? "var(--danger)" : "var(--accent)";
}

function tag(id, text, bg = "var(--border)", fg = "var(--text3)") {
  const el = $(id);
  el.textContent = text;
  el.style.background = bg;
  el.style.color = fg;
}

function log(message, cls = "") {
  const box = $("lBox");
  const line = document.createElement("div");
  line.className = `le ${cls}`;
  line.textContent = message;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

function toast(message, cls = "i") {
  const item = document.createElement("div");
  item.className = `tst tst-${cls}`;
  item.textContent = message;
  $("tbox").appendChild(item);
  setTimeout(() => item.remove(), 2400);
}

function updateClock() {
  $("mTi").textContent = fmtSec(seconds);
}

function updateMetrics(iteration, metrics) {
  if (metrics) lastMetrics = metrics;
  const m = metrics || lastMetrics || {};
  $("mIt").textContent = `${iteration} / ${CFG.max_refinement_iterations}`;
  $("mWi").textContent = pct(m.success_rate ?? 0);
  $("mAv").textContent = m.average_steps === undefined ? "-" : Number(m.average_steps).toFixed(1);
  $("mAr").textContent = m.arpd === null || m.arpd === undefined ? "-" : `${Number(m.arpd).toFixed(1)}%`;
}

function updateTable() {
  $("tInit").textContent = pct(tableState.initial);
  $("tRef").textContent = pct(tableState.refined);
  $("tMan").textContent = pct(tableState.manual);
  $("tIter").textContent = tableState.iterations === null ? "-" : String(tableState.iterations);
}

function renderExperimentRows(rows) {
  const body = $("expRows");
  if (!rows || rows.length === 0) {
    body.innerHTML = `<tr><td colspan="5">No results</td></tr>`;
    return;
  }
  body.innerHTML = rows.map((item) => {
    const row = item.table_row || item;
    const cls = (v) => (v >= CFG.success_threshold ? "good" : "bad");
    return `<tr>
      <td>${escapeHtml(row.task)}</td>
      <td class="${cls(row.R_initial)}">${pct(row.R_initial)}</td>
      <td class="${cls(row.R_refined)}">${pct(row.R_refined)}</td>
      <td class="${cls(row.R_manual)}">${pct(row.R_manual)}</td>
      <td>${row.iterations}</td>
    </tr>`;
  }).join("");
}

function resetTimeline() {
  $("tlB").innerHTML = `
    <div class="idle-tl">
      <div class="idle-tl-card"><div class="num">1</div><div class="lbl">Initial Design</div><div class="desc">LLM writes a Python reward function.</div></div>
      <div class="idle-tl-card"><div class="num">2</div><div class="lbl">Evaluation</div><div class="desc">Q-Learning trains locally on CPU and returns metrics.</div></div>
      <div class="idle-tl-card"><div class="num">3</div><div class="lbl">Self-Refine</div><div class="desc">Low Success Rate becomes feedback for the next reward.</div></div>
    </div>`;
}

function addTimeline(row) {
  const box = $("tlB");
  const existingIdle = box.querySelector(".idle-tl");
  if (existingIdle) box.innerHTML = "";
  const item = document.createElement("div");
  item.className = `tn ${row.active ? "a" : ""} ${row.ok ? "ok" : ""}`;
  item.innerHTML = `
    <div class="tp"></div>
    <div class="tl-lb">${escapeHtml(row.label)}</div>
    <div class="tl-fl">${escapeHtml(row.flow)}</div>
    <div class="tl-st">${escapeHtml(row.stats || "")}</div>`;
  box.appendChild(item);
  box.scrollTop = box.scrollHeight;
}

function scrTL() {
  $("tlB").scrollTo({ top: 0, behavior: "smooth" });
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function syntaxClass(line) {
  const trimmed = line.trim();
  if (trimmed.startsWith("def ")) return "sf";
  if (trimmed.startsWith("\"\"\"") || trimmed.startsWith("'''")) return "ss";
  if (trimmed.startsWith("#")) return "sc";
  if (/^(if|elif|else|return)\b/.test(trimmed)) return "sk";
  if (/[-+]?\d+(\.\d+)?/.test(trimmed)) return "sn";
  return "sv";
}

function renderCode(code, version, oldCode = "") {
  const wrap = $("cWrap");
  wrap.innerHTML = "";
  const oldLines = new Set(oldCode.split(/\r?\n/).map((line) => line.trim()).filter(Boolean));
  const lines = code.split(/\r?\n/);
  lines.forEach((line, index) => {
    const div = document.createElement("div");
    div.className = "c-ln";
    const trimmed = line.trim();
    if (oldCode && trimmed && !oldLines.has(trimmed)) div.classList.add("la", "fx-sl");
    div.innerHTML = `<span class="c-n">${index + 1}</span><span class="c-t ${syntaxClass(line)}">${escapeHtml(line)}</span>`;
    wrap.appendChild(div);
  });
  $("cdVer").textContent = version;
  tag("cdTag", version, "var(--accent-dim)", "var(--accent)");
}

function buildMazeGrid() {
  if (!maze) return [];
  const wallSet = new Set(maze.walls.map(posKey));
  const lavaSet = new Set(maze.lava.map(posKey));
  const grid = [];
  for (let y = 0; y < maze.height; y += 1) {
    const row = [];
    for (let x = 0; x < maze.width; x += 1) {
      const key = `${x},${y}`;
      if (wallSet.has(key)) row.push("wall");
      else if (lavaSet.has(key)) row.push("lava");
      else if (maze.goal[0] === x && maze.goal[1] === y) row.push("goal");
      else row.push("path");
    }
    grid.push(row);
  }
  return grid;
}

function drawMaze() {
  if (!maze) return;
  const c = colors();
  const grid = buildMazeGrid();
  const parent = cv.parentElement;
  const parentWidth = Math.max(320, parent.clientWidth - 20);
  const parentHeight = Math.max(300, parent.clientHeight - 20);
  const tileByWidth = Math.floor((parentWidth - 34) / ((maze.width + maze.height) / 2));
  const tileByHeight = Math.floor((parentHeight - 58) / ((maze.width + maze.height) * 0.27 + 1.8));
  const tileW = Math.max(40, Math.min(64, tileByWidth, tileByHeight));
  const tileH = Math.floor(tileW * 0.54);
  const blockH = Math.floor(tileH * 0.72);
  cv.width = parentWidth;
  cv.height = parentHeight;
  const originX = cv.width / 2;
  const boardH = (maze.width + maze.height) * tileH / 2 + blockH;
  const originY = Math.max(26, Math.floor((cv.height - boardH) / 2) + blockH);
  cx.clearRect(0, 0, cv.width, cv.height);
  drawBackdrop();
  cx.save();
  cx.shadowColor = "rgba(31,48,77,.09)";
  cx.shadowBlur = 18;
  cx.shadowOffsetY = 12;
  drawDiamond(originX, originY + (maze.width + maze.height) * tileH / 4 + 20, cv.width * 0.34, cv.width * 0.17, "rgba(31,48,77,.08)");
  cx.restore();
  drawBoardBase(tileW, tileH, originX, originY, blockH, c);

  const cells = [];
  for (let y = 0; y < maze.height; y += 1) {
    for (let x = 0; x < maze.width; x += 1) {
      const screen = worldToScreen(x, y, tileW, tileH, originX, originY);
      cells.push({ x, y, sx: screen.sx, sy: screen.sy });
    }
  }
  cells.sort((a, b) => a.sy - b.sy);

  for (const cellInfo of cells) {
      const { x, y, sx, sy } = cellInfo;
      const cell = grid[y][x];
      if (cell === "wall") {
        drawFlatTile(sx, sy, tileW, tileH, c.pathTop, c.pathLine);
        drawWallBlock(sx, sy - tileH * 0.04, tileW * 0.9, tileH * 0.9, Math.floor(blockH * 0.66), c);
      } else if (cell === "lava") {
        drawFlatTile(sx, sy, tileW, tileH, c.pathTop, c.pathLine);
        drawLavaTile(sx, sy, tileW, tileH, c);
      } else {
        drawFlatTile(sx, sy, tileW, tileH, c.pathTop, c.pathLine);
        if (cell === "goal") {
          drawGoalTile(sx, sy, tileW, tileH, c);
        }
      }
  }

  drawTrailDots(tileW, tileH, originX, originY, c);

  const { sx: bx, sy: by } = worldToScreen(bot.x, bot.y, tileW, tileH, originX, originY);
  drawBot3d(bx, by - tileH * 0.1, tileW, tileH, c);
}

function drawBackdrop() {
  const grd = cx.createLinearGradient(0, 0, 0, cv.height);
  grd.addColorStop(0, "#ffffff");
  grd.addColorStop(1, "#eef4fb");
  cx.fillStyle = grd;
  cx.fillRect(0, 0, cv.width, cv.height);
  cx.fillStyle = "rgba(37,99,235,.05)";
  for (let i = 0; i < 18; i += 1) {
    const x = (i * 97) % cv.width;
    const y = (i * 53) % cv.height;
    cx.beginPath();
    cx.arc(x, y, 1.2, 0, Math.PI * 2);
    cx.fill();
  }
}

function drawBoardBase(tileW, tileH, originX, originY, blockH, c) {
  const corners = [
    worldToScreen(-0.5, -0.5, tileW, tileH, originX, originY),
    worldToScreen(maze.width - 0.5, -0.5, tileW, tileH, originX, originY),
    worldToScreen(maze.width - 0.5, maze.height - 0.5, tileW, tileH, originX, originY),
    worldToScreen(-0.5, maze.height - 0.5, tileW, tileH, originX, originY),
  ];
  const depth = Math.max(8, blockH * 0.36);
  const shadowOffset = depth + tileH * 0.18;

  cx.save();
  cx.fillStyle = "rgba(31,48,77,.10)";
  cx.beginPath();
  cx.moveTo(corners[0].sx, corners[0].sy + shadowOffset);
  for (let i = 1; i < corners.length; i += 1) {
    cx.lineTo(corners[i].sx, corners[i].sy + shadowOffset);
  }
  cx.closePath();
  cx.fill();
  cx.restore();

  const edges = [
    [corners[0], corners[1]],
    [corners[1], corners[2]],
    [corners[2], corners[3]],
    [corners[3], corners[0]],
  ];
  const topToBottom = edges.map(([a, b]) => ((a.sy + b.sy) / 2));
  const frontIdx = topToBottom.indexOf(Math.max(...topToBottom));
  const leftIdx = (frontIdx + 3) % 4;
  const rightIdx = frontIdx;

  drawBoardSide(edges[leftIdx][0], edges[leftIdx][1], depth, c.boardSideL);
  drawBoardSide(edges[rightIdx][0], edges[rightIdx][1], depth, c.boardSideR);

  cx.beginPath();
  cx.moveTo(corners[0].sx, corners[0].sy);
  for (let i = 1; i < corners.length; i += 1) {
    cx.lineTo(corners[i].sx, corners[i].sy);
  }
  cx.closePath();
  cx.fillStyle = c.boardTop;
  cx.fill();
  cx.strokeStyle = "rgba(31,48,77,.10)";
  cx.lineWidth = 1;
  cx.stroke();
}

function drawBoardSide(a, b, depth, fill) {
  cx.beginPath();
  cx.moveTo(a.sx, a.sy);
  cx.lineTo(b.sx, b.sy);
  cx.lineTo(b.sx, b.sy + depth);
  cx.lineTo(a.sx, a.sy + depth);
  cx.closePath();
  cx.fillStyle = fill;
  cx.fill();
  cx.strokeStyle = "rgba(31,48,77,.12)";
  cx.lineWidth = 1;
  cx.stroke();
}

function drawTrailDots(tileW, tileH, originX, originY, c) {
  if (!trail.length) return;
  trail.forEach((p, i) => {
    const { sx, sy } = worldToScreen(p.x, p.y, tileW, tileH, originX, originY);
    const age = i / Math.max(1, trail.length - 1);
    cx.fillStyle = i === trail.length - 1 ? c.trailActive : c.trailDot;
    cx.strokeStyle = "rgba(255,255,255,.92)";
    cx.lineWidth = Math.max(1.5, tileW * 0.035);
    cx.beginPath();
    cx.ellipse(
      sx,
      sy + tileH * 0.16,
      Math.max(5, tileW * (0.10 + 0.03 * age)),
      Math.max(3, tileH * (0.08 + 0.02 * age)),
      0,
      0,
      Math.PI * 2,
    );
    cx.fill();
    cx.stroke();
    if (i === trail.length - 1) {
      cx.fillStyle = "#2563eb";
      cx.beginPath();
      cx.arc(sx, sy + tileH * 0.16, Math.max(2.5, tileW * 0.04), 0, Math.PI * 2);
      cx.fill();
    }
  });
}

function worldToScreen(x, y, tileW, tileH, originX, originY) {
  const cx0 = (maze.width - 1) / 2;
  const cy0 = (maze.height - 1) / 2;
  const px = x - cx0;
  const py = y - cy0;
  const rad = boardYaw * Math.PI / 180;
  const rx = px * Math.cos(rad) - py * Math.sin(rad);
  const ry = px * Math.sin(rad) + py * Math.cos(rad);
  return isoToScreen(rx + cx0, ry + cy0, tileW, tileH, originX, originY);
}

function isoToScreen(x, y, tileW, tileH, originX, originY) {
  return {
    sx: originX + (x - y) * tileW / 2,
    sy: originY + (x + y) * tileH / 2,
  };
}

function drawDiamond(cx0, cy0, tileW, tileH, fill, stroke) {
  cx.beginPath();
  cx.moveTo(cx0, cy0 - tileH / 2);
  cx.lineTo(cx0 + tileW / 2, cy0);
  cx.lineTo(cx0, cy0 + tileH / 2);
  cx.lineTo(cx0 - tileW / 2, cy0);
  cx.closePath();
  cx.fillStyle = fill;
  cx.fill();
  if (stroke) {
    cx.strokeStyle = stroke;
    cx.lineWidth = 1;
    cx.stroke();
  }
}

function drawFlatTile(cx0, cy0, tileW, tileH, fill, stroke) {
  const grd = cx.createLinearGradient(cx0, cy0 - tileH / 2, cx0, cy0 + tileH / 2);
  grd.addColorStop(0, "#ffffff");
  grd.addColorStop(1, fill);
  drawDiamond(cx0, cy0, tileW, tileH, grd, stroke);
  cx.strokeStyle = "rgba(255,255,255,.75)";
  cx.lineWidth = 0.8;
  cx.beginPath();
  cx.moveTo(cx0, cy0 - tileH / 2 + 1);
  cx.lineTo(cx0 + tileW / 2 - 2, cy0);
  cx.stroke();
}

function drawWallBlock(cx0, cy0, tileW, tileH, h, c) {
  cx.save();
  cx.shadowColor = "rgba(31,48,77,.20)";
  cx.shadowBlur = 10;
  cx.shadowOffsetY = 6;
  drawPrism(cx0, cy0, tileW, tileH, h, c.wallTop, c.wallLeft, c.wallRight, "rgba(31,48,77,.18)");
  drawDiamond(cx0, cy0 - h - 1, tileW * 0.7, tileH * 0.7, "rgba(255,255,255,.18)");
  cx.restore();
}

function drawLavaTile(cx0, cy0, tileW, tileH, c) {
  cx.save();
  cx.shadowColor = c.lavaGlow;
  cx.shadowBlur = 14;
  const grd = cx.createRadialGradient(cx0, cy0, 2, cx0, cy0, tileW * 0.42);
  grd.addColorStop(0, c.lavaCore);
  grd.addColorStop(0.58, c.lavaTop);
  grd.addColorStop(1, "rgba(255,214,221,.72)");
  drawDiamond(cx0, cy0, tileW * 0.82, tileH * 0.82, grd, "rgba(208,68,90,.42)");
  cx.fillStyle = "rgba(255,255,255,.28)";
  cx.beginPath();
  cx.ellipse(cx0 - tileW * 0.08, cy0 - tileH * 0.04, tileW * 0.12, tileH * 0.06, -0.2, 0, Math.PI * 2);
  cx.fill();
  cx.restore();
}

function drawGoalTile(cx0, cy0, tileW, tileH, c) {
  drawDiamond(cx0, cy0, tileW * 0.9, tileH * 0.9, c.goalTop, "rgba(15,159,122,.45)");
  cx.save();
  cx.shadowColor = "rgba(15,159,122,.22)";
  cx.shadowBlur = 12;
  cx.fillStyle = c.goalText;
  cx.font = `900 ${Math.max(12, tileW * 0.32)}px Arial, sans-serif`;
  cx.textAlign = "center";
  cx.textBaseline = "middle";
  cx.fillText("G", cx0, cy0 + tileH * 0.02);
  cx.restore();
}


function drawPrism(cx0, cy0, tileW, tileH, h, top, left, right, stroke) {
  const topY = cy0 - h;
  cx.beginPath();
  cx.moveTo(cx0 - tileW / 2, cy0);
  cx.lineTo(cx0, cy0 + tileH / 2);
  cx.lineTo(cx0, cy0 + tileH / 2 - h);
  cx.lineTo(cx0 - tileW / 2, cy0 - h);
  cx.closePath();
  cx.fillStyle = left;
  cx.fill();

  cx.beginPath();
  cx.moveTo(cx0 + tileW / 2, cy0);
  cx.lineTo(cx0, cy0 + tileH / 2);
  cx.lineTo(cx0, cy0 + tileH / 2 - h);
  cx.lineTo(cx0 + tileW / 2, cy0 - h);
  cx.closePath();
  cx.fillStyle = right;
  cx.fill();

  drawDiamond(cx0, topY, tileW, tileH, top, stroke);
}

function drawBot3d(x, y, tileW, tileH, c) {
  const w = tileW * 0.55;
  const h = tileH * 1.02;
  cx.fillStyle = c.shadow;
  cx.beginPath();
  cx.ellipse(x, y + tileH * 0.54, tileW * 0.34, tileH * 0.18, 0, 0, Math.PI * 2);
  cx.fill();

  cx.fillStyle = c.botSide;
  roundedRect(cx, x - w / 2, y - h * 0.1, w, h * 0.75, 7);
  cx.fill();
  const body = cx.createLinearGradient(x, y - h * 0.38, x, y + h * 0.45);
  body.addColorStop(0, "#4f8df7");
  body.addColorStop(1, c.bot);
  cx.fillStyle = body;
  roundedRect(cx, x - w / 2, y - h * 0.34, w, h * 0.78, 7);
  cx.fill();
  cx.strokeStyle = "rgba(23,70,179,.35)";
  cx.lineWidth = 1.5;
  cx.stroke();
  cx.fillStyle = c.botLight;
  cx.beginPath();
  cx.arc(x - w * 0.18, y - h * 0.07, Math.max(2, tileW * 0.04), 0, Math.PI * 2);
  cx.arc(x + w * 0.18, y - h * 0.07, Math.max(2, tileW * 0.04), 0, Math.PI * 2);
  cx.fill();
  cx.strokeStyle = "rgba(255,255,255,.55)";
  cx.lineWidth = 1;
  cx.beginPath();
  cx.moveTo(x - w * 0.22, y - h * 0.25);
  cx.lineTo(x + w * 0.18, y - h * 0.25);
  cx.stroke();
}

function roundedRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

async function loadMaze() {
  maze = await api(`/api/maze?${taskQuery()}`);
  bot = { x: maze.start[0], y: maze.start[1] };
  trail = [];
  $("sSt").textContent = `0/${maze.max_steps}`;
  $("sCo").textContent = "0";
  $("sOp").textContent = maze.optimal_path_length ?? "-";
  $("sBo").textContent = "Idle";
  drawMaze();
}

async function loadTasks() {
  allTasks = await api("/api/tasks");
  const select = $("selTask");
  select.innerHTML = "";
  allTasks.forEach((task) => {
    const opt = document.createElement("option");
    opt.value = task.id;
    opt.textContent = `${task.name} | optimal ${task.optimal_path_length ?? "-"}`;
    opt.title = task.description;
    if (task.id === selectedTaskId) {
      opt.selected = true;
      $("inpTask").value = task.description;
    }
    select.appendChild(opt);
  });
  select.addEventListener("change", async () => {
    if (timer) return;
    selectedTaskId = select.value;
    const task = allTasks.find(t => t.id === selectedTaskId);
    if (task) {
      $("inpTask").value = task.description;
    }
    await loadMaze();
    tag("mzTag", selectedTaskId);
    log(`Selected task: ${select.options[select.selectedIndex].textContent}`, "i");
  });
}

async function loadCurrentReward() {
  const result = await api("/api/reward");
  previousCode = result.code || "";
  if (previousCode) renderCode(previousCode, "current_reward.py");
}

async function animateTrajectory(trajectory, maxSteps) {
  trail = [];
  let collisions = 0;
  if (!trajectory || trajectory.length === 0) return;
  bot = { x: trajectory[0].x, y: trajectory[0].y };
  drawMaze();
  for (let i = 1; i < trajectory.length; i += 1) {
    await waitIfPaused();
    if (stopped) return;
    const point = trajectory[i];
    trail.push({ x: bot.x, y: bot.y });
    bot = { x: point.x, y: point.y };
    if (point.event === "lava") collisions += 1;
    $("sSt").textContent = `${i}/${maxSteps}`;
    $("sCo").textContent = String(collisions);
    $("sBo").textContent = point.event === "goal" ? "Goal" : point.event === "lava" ? "Lava" : "Move";
    drawMaze();
    await sleep(Math.max(35, 180 / speed));
  }
}

async function runOneIteration(iteration, mode) {
  const label = iteration === 0 ? "R_initial" : `R_refined_${iteration}`;
  const progressBase = iteration === 0 ? 8 : 25 + iteration * 10;
  setStatus(iteration === 0 ? "LLM viet reward" : "LLM refine reward", "g", progressBase, "Generating");
  tag("mzTag", `Iteration ${iteration}`, "var(--accent-dim)", "var(--accent)");
  log(`${label}: generating reward code...`, "i");

  const generationPayload =
    mode === "refine"
      ? {
          mode: "refine",
          task_id: selectedTaskId,
          metrics: lastMetrics,
          previous_code: previousCode,
          instruction: $("inpTask").value,
        }
      : { mode: "initial", task_id: selectedTaskId, instruction: $("inpTask").value };
  const generation = await api("/api/generate-reward", {
    method: "POST",
    body: JSON.stringify(generationPayload),
  });

  renderCode(generation.code, label, previousCode);
  previousCode = generation.code;
  if (generation.warning) {
    log(`WARNING: ${generation.warning}`, "w");
  } else {
    log(`${label}: reward source=${generation.source}, latency=${generation.latency_sec.toFixed(3)}s`, "i");
  }

  setStatus("Q-Learning dang train", "r", Math.min(90, progressBase + 12), "Running");
  const run = await api("/api/run-agent", {
    method: "POST",
    body: JSON.stringify({
      episodes: CFG.default_episodes,
      eval_trials: CFG.default_eval_trials,
      seed: 42 + iteration,
      task_id: selectedTaskId,
    }),
  });

  const m = run.evaluation;
  updateMetrics(iteration, m);
  await animateTrajectory(m.trajectory, maze.max_steps);

  if (iteration === 0) tableState.initial = m.success_rate;
  tableState.refined = m.success_rate;
  tableState.iterations = iteration;
  updateTable();

  const ok = m.success_rate >= CFG.success_threshold;
  addTimeline({
    label,
    active: !ok,
    ok,
    flow: `SR=${pct(m.success_rate)} | AvgReward=${m.average_reward.toFixed(2)} | AvgSteps=${m.average_steps.toFixed(1)} | Lava=${m.lava_hits} | Wall=${m.wall_hits}`,
    stats: `CPU train=${run.training.cpu_training_time_sec.toFixed(3)}s, eval=${m.cpu_eval_time_sec.toFixed(3)}s, ARPD=${m.arpd === null ? "-" : m.arpd.toFixed(1) + "%"}`,
  });
  log(`${label}: SR=${pct(m.success_rate)}, avg_steps=${m.average_steps.toFixed(1)}, ARPD=${m.arpd === null ? "-" : m.arpd.toFixed(1) + "%"}`, ok ? "o" : "e");
  return ok;
}

async function runManualBaseline() {
  setStatus("Dang chay R_manual", "r", 96, "Manual");
  const manual = await api("/api/manual-baseline", {
    method: "POST",
    body: JSON.stringify({
      episodes: CFG.default_episodes,
      eval_trials: CFG.default_eval_trials,
      seed: 2023,
      task_id: selectedTaskId,
    }),
  });
  tableState.manual = manual.evaluation.success_rate;
  updateTable();
  addTimeline({
    label: "R_manual baseline",
    ok: manual.evaluation.success_rate >= CFG.success_threshold,
    flow: `SR=${pct(manual.evaluation.success_rate)} | AvgSteps=${manual.evaluation.average_steps.toFixed(1)}`,
    stats: "Human-designed dense reward baseline for Table 1 comparison.",
  });
  log(`R_manual: SR=${pct(manual.evaluation.success_rate)}, avg_steps=${manual.evaluation.average_steps.toFixed(1)}`, "i");
}

async function go() {
  if (timer) return;
  stopped = false;
  paused = false;
  $("btnRun").disabled = true;
  $("btnP").disabled = false;
  $("btnP").textContent = "Tam dung";
  seconds = 0;
  timer = setInterval(() => {
    seconds += 1;
    updateClock();
  }, 1000);
  $("lBox").innerHTML = "";
  resetTimeline();
  tableState = { initial: null, refined: null, manual: null, iterations: 0 };
  updateTable();

  try {
    await loadMaze();
    log(`Starting task ${selectedTaskId}: LLM reward -> Q-Learning -> feedback.`, "i");
    let ok = await runOneIteration(0, "initial");
    let iteration = 0;
    while (!ok && iteration < CFG.max_refinement_iterations && !stopped) {
      iteration += 1;
      setStatus("Phan tich feedback", "w", 25 + iteration * 10, "Refining");
      log("Feedback sent to LLM: low SR and objective metrics.", "i");
      ok = await runOneIteration(iteration, "refine");
    }

    if (!stopped) {
      await runManualBaseline();
      setStatus(ok ? "Hoan tat" : "Can refine them", ok ? "d" : "e", 100, "Done");
      tag("mzTag", ok ? "Success" : "Needs more work", ok ? "var(--accent-dim)" : "var(--danger-dim)", ok ? "var(--accent)" : "var(--danger)");
      $("spoS").textContent = `R_initial=${pct(tableState.initial)} | R_refined=${pct(tableState.refined)} | R_manual=${pct(tableState.manual)} | Iter=${tableState.iterations}`;
      $("spo").classList.add("show");
    }
  } catch (err) {
    console.error(err);
    setStatus("Loi", "e", 100, "Error");
    log(err.message, "e");
    toast("Experiment failed. Check backend console.", "e");
  } finally {
    clearInterval(timer);
    timer = null;
    $("btnRun").disabled = false;
    $("btnP").disabled = true;
  }
}

async function runAll() {
  if (timer) return;
  stopped = false;
  $("btnRun").disabled = true;
  $("btnRunAll").disabled = true;
  $("lBox").innerHTML = "";
  resetTimeline();
  setStatus("Dang chay tat ca task", "r", 10, "Run All");
  log("Running all challenge tasks to reproduce a Table 1-style result.", "i");
  $("expRows").innerHTML = `<tr><td colspan="5">Running...</td></tr>`;
  const start = Date.now();
  try {
    const result = await api("/api/run-all", {
      method: "POST",
      body: JSON.stringify({
        episodes: CFG.default_episodes,
        eval_trials: CFG.default_eval_trials,
        max_iterations: CFG.max_refinement_iterations,
      }),
    });
    renderExperimentRows(result.results);
    setStatus("Hoan tat bang", "d", 100, "Done");
    log(`Run-all completed in ${(result.wall_time_sec || ((Date.now() - start) / 1000)).toFixed(2)}s for ${result.results.length} tasks.`, "o");
    addTimeline({
      label: "Table 1 reproduction",
      ok: true,
      flow: `${result.results.length} tasks | SR columns: R_initial, R_refined, R_manual | max iter=${CFG.max_refinement_iterations}`,
      stats: `Wall time=${result.wall_time_sec.toFixed(2)}s, eval_trials=${result.eval_trials}`,
    });
  } catch (err) {
    console.error(err);
    setStatus("Loi", "e", 100, "Error");
    log(err.message, "e");
  } finally {
    $("btnRun").disabled = false;
    $("btnRunAll").disabled = false;
  }
}

function closeSpo() {
  $("spo").classList.remove("show");
}

function togP() {
  paused = !paused;
  $("btnP").textContent = paused ? "Tiep tuc" : "Tam dung";
  if (paused) setStatus("Tam dung", "i", Number($("pFill").style.width.replace("%", "")) || 0, "Paused");
}

function cySp() {
  const values = [0.5, 1, 2, 4];
  const index = (values.indexOf(speed) + 1) % values.length;
  speed = values[index];
  $("btnSp").textContent = `${speed}x`;
}

function reMz() {
  drawMaze();
}

function setBoardYaw(value) {
  boardYaw = Math.max(-180, Math.min(180, Number(value) || 0));
  const slider = $("rotSlider");
  const label = $("rotVal");
  if (slider) slider.value = String(Math.round(boardYaw));
  if (label) label.textContent = `${Math.round(boardYaw)} deg`;
  drawMaze();
}

function rst() {
  stopped = true;
  paused = false;
  clearInterval(timer);
  timer = null;
  seconds = 0;
  lastMetrics = null;
  tableState = { initial: null, refined: null, manual: null, iterations: 0 };
  updateClock();
  updateMetrics(0, null);
  updateTable();
  $("lBox").innerHTML = "";
  $("btnRun").disabled = false;
  $("btnRunAll").disabled = false;
  $("btnP").disabled = true;
  $("btnP").textContent = "Tam dung";
  setStatus("San sang", "i", 0, "Status");
  tag("mzTag", "Ready");
  closeSpo();
  resetTimeline();
  loadMaze().catch((err) => log(err.message, "e"));
}

async function bootstrap() {
  try {
    CFG = await api("/api/config");
    await loadTasks();
    await loadMaze();
    await loadCurrentReward();
    resetTimeline();
    updateMetrics(0, null);
    updateTable();
    setupBoardRotationControls();
  } catch (err) {
    log(err.message, "e");
  }
}

function setupBoardRotationControls() {
  const slider = $("rotSlider");
  if (slider) {
    slider.addEventListener("input", () => setBoardYaw(slider.value));
  }
  cv.addEventListener("pointerdown", (event) => {
    draggingBoard = true;
    dragStartX = event.clientX;
    dragStartYaw = boardYaw;
    cv.setPointerCapture(event.pointerId);
    cv.style.cursor = "grabbing";
  });
  cv.addEventListener("pointermove", (event) => {
    if (!draggingBoard) return;
    setBoardYaw(dragStartYaw + (event.clientX - dragStartX) * 0.7);
  });
  const stopDrag = (event) => {
    draggingBoard = false;
    cv.style.cursor = "grab";
    try { cv.releasePointerCapture(event.pointerId); } catch (_) {}
  };
  cv.addEventListener("pointerup", stopDrag);
  cv.addEventListener("pointercancel", stopDrag);
  cv.style.cursor = "grab";
}

bootstrap();
