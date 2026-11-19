import { createDataset, DATASETS } from "./data.js";
import { MLP, MODEL_PRESETS } from "./network.js";
import { LineChart } from "./charts.js";

const byId = (id) => document.getElementById(id);

const elements = {
  datasetTabs: byId("datasetTabs"),
  datasetHint: byId("datasetHint"),
  noiseRange: byId("noiseRange"),
  noiseValue: byId("noiseValue"),
  sampleRange: byId("sampleRange"),
  sampleValue: byId("sampleValue"),
  modelOptions: byId("modelOptions"),
  lrRange: byId("lrRange"),
  lrValue: byId("lrValue"),
  epochRange: byId("epochRange"),
  epochValueMax: byId("epochValueMax"),
  demoButton: byId("demoButton"),
  trainButton: byId("trainButton"),
  resetButton: byId("resetButton"),
  regenerateButton: byId("regenerateButton"),
  challengeValue: byId("challengeValue"),
  challengeText: byId("challengeText"),
  challengeBar: byId("challengeBar"),
  heroDataset: byId("heroDataset"),
  heroModel: byId("heroModel"),
  heroAcc: byId("heroAcc"),
  vizTitle: byId("vizTitle"),
  showTestToggle: byId("showTestToggle"),
  canvasWrap: byId("canvasWrap"),
  decisionCanvas: byId("decisionCanvas"),
  hoverTooltip: byId("hoverTooltip"),
  epochValue: byId("epochValue"),
  epochTotal: byId("epochTotal"),
  trainAccValue: byId("trainAccValue"),
  testAccValue: byId("testAccValue"),
  lossValue: byId("lossValue"),
  lossChart: byId("lossChart"),
  accuracyChart: byId("accuracyChart"),
  cm00: byId("cm00"),
  cm01: byId("cm01"),
  cm10: byId("cm10"),
  cm11: byId("cm11"),
  cmTotal: byId("cmTotal"),
  architectureValue: byId("architectureValue"),
  paramsValue: byId("paramsValue"),
  dataCountValue: byId("dataCountValue"),
  seedValue: byId("seedValue"),
  modelTip: byId("modelTip"),
  exportImageButton: byId("exportImageButton"),
  exportJsonButton: byId("exportJsonButton"),
  toast: byId("toast"),
};

const state = {
  kind: "moons",
  noise: 0.15,
  sampleCount: 400,
  modelKey: "small",
  learningRate: 0.01,
  maxEpochs: 600,
  seed: 20261119,
  initCounter: 0,
  dataset: null,
  model: null,
  history: [],
  epoch: 0,
  training: false,
  paused: false,
  runToken: 0,
  trainMetrics: null,
  testMetrics: null,
  showTest: true,
  hover: null,
};

const charts = {};
const canvasContext = elements.decisionCanvas.getContext("2d");
const contourCanvas = document.createElement("canvas");
const contourResolution = 84;
contourCanvas.width = contourResolution;
contourCanvas.height = contourResolution;
const contourContext = contourCanvas.getContext("2d", { willReadFrequently: true });
const contourImage = contourContext.createImageData(contourResolution, contourResolution);
const contourField = new Float32Array(contourResolution * contourResolution);
const canvasSize = { width: 0, height: 0, pixelRatio: 0 };
const challengeTarget = 0.95;
let toastTimer = 0;

function formatPercent(value) {
  if (!Number.isFinite(value)) return "--";
  return `${(value * 100).toFixed(1)}%`;
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    elements.toast.classList.remove("show");
  }, 2800);
}

function modelSeed() {
  return (state.seed + 0x9e3779b9 + Math.imul(state.initCounter, 0x85ebca6b)) >>> 0;
}

function buildModel() {
  const preset = MODEL_PRESETS[state.modelKey] || MODEL_PRESETS.small;
  state.model = new MLP(preset.hidden, { seed: modelSeed() });
  state.epoch = 0;
  state.history = [];
  state.trainMetrics = state.model.evaluate(state.dataset.train);
  state.testMetrics = state.model.evaluate(state.dataset.test);
}

function resetModel(options = {}) {
  const { freshInit = false, quiet = false } = options;
  state.runToken += 1;
  state.training = false;
  state.paused = false;
  if (freshInit) state.initCounter += 1;
  buildModel();
  updateAll();
  if (!quiet) showToast("模型已重置，使用新的随机初始化。");
}

function regenerateData(options = {}) {
  const { quiet = false, toastMessage = "已生成新的数据集。" } = options;
  state.runToken += 1;
  state.training = false;
  state.paused = false;
  state.initCounter = 0;
  state.dataset = createDataset(state.kind, state.sampleCount, state.noise, state.seed);
  buildModel();
  updateAll();
  if (!quiet) showToast(toastMessage);
}

function launchTraining() {
  state.runToken += 1;
  const token = state.runToken;
  window.requestAnimationFrame(() => trainingFrame(token));
}

function startTraining() {
  if (!state.model) return;
  if (state.epoch >= state.maxEpochs) {
    state.initCounter += 1;
    buildModel();
  }
  state.training = true;
  state.paused = false;
  elements.hoverTooltip.hidden = true;
  updateControlState();
  launchTraining();
}

function pauseTraining() {
  state.paused = true;
  state.runToken += 1;
  updateControlState();
  showToast("训练已暂停，可以继续调整参数。");
}

function resumeTraining() {
  state.paused = false;
  updateControlState();
  launchTraining();
}

function toggleTraining() {
  if (!state.training) {
    startTraining();
  } else if (state.paused) {
    resumeTraining();
  } else {
    pauseTraining();
  }
}

function finishTraining() {
  state.training = false;
  state.paused = false;
  state.runToken += 1;
  updateControlState();
  updateMetrics();
  const accuracy = state.testMetrics ? state.testMetrics.accuracy : 0;
  if (accuracy >= challengeTarget) {
    showToast(`训练完成，测试准确率 ${formatPercent(accuracy)}，挑战达成！`);
  } else {
    showToast(`训练完成，测试准确率 ${formatPercent(accuracy)}。可以换模型或增加轮数再试。`);
  }
}

function trainingFrame(token) {
  if (token !== state.runToken || !state.training || state.paused) return;

  const frameStart = performance.now();
  let epochsThisFrame = 0;

  do {
    state.model.trainEpoch(state.dataset.train, state.learningRate);
    state.epoch += 1;
    state.trainMetrics = state.model.evaluate(state.dataset.train);
    state.testMetrics = state.model.evaluate(state.dataset.test);
    state.history.push({
      epoch: state.epoch,
      trainLoss: state.trainMetrics.loss,
      testLoss: state.testMetrics.loss,
      trainAcc: state.trainMetrics.accuracy,
      testAcc: state.testMetrics.accuracy,
    });
    epochsThisFrame += 1;
  } while (
    state.epoch < state.maxEpochs &&
    epochsThisFrame < 3 &&
    performance.now() - frameStart < 13
  );

  renderCanvas();
  drawCharts();
  updateMetrics();
  updateControlState();

  if (state.epoch >= state.maxEpochs) {
    finishTraining();
    return;
  }

  window.requestAnimationFrame(() => trainingFrame(token));
}

function runDemo() {
  state.runToken += 1;
  state.training = false;
  state.paused = false;
  state.modelKey = state.kind === "spiral" ? "deep" : "small";
  state.learningRate = 0.012;
  state.maxEpochs = 600;
  state.seed = (state.seed + 1) >>> 0;
  state.initCounter = 0;
  syncControls();
  regenerateData({ quiet: true });
  startTraining();
  showToast(`演示模式：${DATASETS[state.kind].name} + ${MODEL_PRESETS[state.modelKey].name}`);
}

function updateControlState() {
  if (state.training && !state.paused) {
    elements.trainButton.textContent = "暂停训练";
  } else if (state.training && state.paused) {
    elements.trainButton.textContent = "继续训练";
  } else if (state.epoch >= state.maxEpochs) {
    elements.trainButton.textContent = "重新训练";
  } else {
    elements.trainButton.textContent = "开始训练";
  }
}

function updateMetrics() {
  const hasRun = state.epoch > 0;
  elements.epochValue.textContent = String(state.epoch);
  elements.epochTotal.textContent = `/ ${state.maxEpochs}`;
  elements.trainAccValue.textContent = hasRun && state.trainMetrics ? formatPercent(state.trainMetrics.accuracy) : "--";
  elements.testAccValue.textContent = hasRun && state.testMetrics ? formatPercent(state.testMetrics.accuracy) : "--";
  elements.lossValue.textContent = hasRun && state.testMetrics ? state.testMetrics.loss.toFixed(3) : "--";

  const accuracy = hasRun && state.testMetrics ? state.testMetrics.accuracy : 0;
  elements.heroAcc.textContent = hasRun ? formatPercent(accuracy) : "--";
  elements.challengeValue.textContent = `${Math.round(accuracy * 100)} / 95%`;
  elements.challengeBar.style.width = `${Math.min(100, (accuracy / challengeTarget) * 100)}%`;
  elements.challengeText.classList.toggle("success", accuracy >= challengeTarget);

  if (accuracy >= challengeTarget) {
    elements.challengeText.textContent = `挑战达成！测试准确率达到 ${formatPercent(accuracy)}。`;
  } else if (hasRun && state.modelKey === "linear") {
    elements.challengeText.textContent = `当前 ${formatPercent(accuracy)}。线性模型的边界是一条直线，换成 MLP 才能弯曲。`;
  } else if (hasRun) {
    const gap = Math.max(0, challengeTarget * 100 - accuracy * 100);
    elements.challengeText.textContent = `当前 ${formatPercent(accuracy)}，距离目标还差 ${gap.toFixed(1)} 个百分点。`;
  } else {
    elements.challengeText.textContent = "先训练模型，再挑战 95% 测试准确率。";
  }

  if (hasRun && state.testMetrics) {
    const confusion = state.testMetrics.confusion;
    elements.cm00.textContent = String(confusion[0][0]);
    elements.cm01.textContent = String(confusion[0][1]);
    elements.cm10.textContent = String(confusion[1][0]);
    elements.cm11.textContent = String(confusion[1][1]);
    elements.cmTotal.textContent = `${state.dataset.test.length} 个测试样本`;
  } else {
    elements.cm00.textContent = "0";
    elements.cm01.textContent = "0";
    elements.cm10.textContent = "0";
    elements.cm11.textContent = "0";
    elements.cmTotal.textContent = `${state.dataset.test.length} 个测试样本`;
  }
}

function updateModelCard() {
  if (!state.model || !state.dataset) return;
  const preset = MODEL_PRESETS[state.modelKey] || MODEL_PRESETS.small;
  elements.heroDataset.textContent = state.dataset.name;
  elements.heroModel.textContent = preset.name;
  elements.vizTitle.textContent = state.dataset.name;
  elements.datasetHint.textContent = state.dataset.hint;
  elements.architectureValue.textContent = state.model.architecture();
  elements.paramsValue.textContent = state.model.countParameters().toLocaleString("zh-CN");
  elements.dataCountValue.textContent = `${state.dataset.train.length} / ${state.dataset.test.length}`;
  elements.seedValue.textContent = String(state.dataset.seed);
  elements.modelTip.textContent = state.modelKey === "linear" ? MODEL_PRESETS.linear.tip : state.dataset.tip;
}

function updateAll() {
  updateModelCard();
  updateMetrics();
  updateControlState();
  drawCharts();
  renderCanvas();
}

function drawCharts() {
  if (!state.dataset) return;
  charts.loss.draw(state.history, state.maxEpochs);
  charts.accuracy.draw(state.history, state.maxEpochs);
}

function resizeCanvas() {
  const rectangle = elements.decisionCanvas.getBoundingClientRect();
  if (!rectangle.width || !rectangle.height) return false;
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.round(rectangle.width);
  const height = Math.round(rectangle.height);

  if (width !== canvasSize.width || height !== canvasSize.height || pixelRatio !== canvasSize.pixelRatio) {
    canvasSize.width = width;
    canvasSize.height = height;
    canvasSize.pixelRatio = pixelRatio;
    elements.decisionCanvas.width = Math.round(width * pixelRatio);
    elements.decisionCanvas.height = Math.round(height * pixelRatio);
    canvasContext.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  }
  return true;
}

function mapX(value, width, domain) {
  return ((value + domain) / (domain * 2)) * width;
}

function mapY(value, height, domain) {
  return ((domain - value) / (domain * 2)) * height;
}

function drawGrid(width, height, domain) {
  canvasContext.save();
  canvasContext.strokeStyle = "rgba(180, 210, 245, 0.085)";
  canvasContext.lineWidth = 1;
  canvasContext.setLineDash([2, 5]);

  for (const value of [-1, 0, 1]) {
    const x = mapX(value, width, domain);
    const y = mapY(value, height, domain);
    canvasContext.beginPath();
    canvasContext.moveTo(x, 0);
    canvasContext.lineTo(x, height);
    canvasContext.moveTo(0, y);
    canvasContext.lineTo(width, y);
    canvasContext.stroke();
  }
  canvasContext.restore();
}

function drawBoundary(width, height, domain) {
  canvasContext.save();
  canvasContext.strokeStyle = "rgba(238, 248, 255, 0.76)";
  canvasContext.lineWidth = 1.15;
  canvasContext.lineJoin = "round";
  canvasContext.lineCap = "round";
  canvasContext.shadowColor = "rgba(220, 244, 255, 0.5)";
  canvasContext.shadowBlur = 6;
  canvasContext.beginPath();

  const cellWidth = width / contourResolution;
  const cellHeight = height / contourResolution;

  for (let row = 0; row < contourResolution - 1; row += 1) {
    for (let column = 0; column < contourResolution - 1; column += 1) {
      const topLeft = contourField[row * contourResolution + column];
      const topRight = contourField[row * contourResolution + column + 1];
      const bottomRight = contourField[(row + 1) * contourResolution + column + 1];
      const bottomLeft = contourField[(row + 1) * contourResolution + column];
      const caseIndex =
        (topLeft > 0 ? 1 : 0) |
        (topRight > 0 ? 2 : 0) |
        (bottomRight > 0 ? 4 : 0) |
        (bottomLeft > 0 ? 8 : 0);

      if (caseIndex === 0 || caseIndex === 15) continue;

      const topX = (column + topLeft / (topLeft - topRight || 1e-9)) * cellWidth;
      const topY = row * cellHeight;
      const rightX = (column + 1) * cellWidth;
      const rightY = (row + topRight / (topRight - bottomRight || 1e-9)) * cellHeight;
      const bottomX = (column + bottomLeft / (bottomLeft - bottomRight || 1e-9)) * cellWidth;
      const bottomY = (row + 1) * cellHeight;
      const leftX = column * cellWidth;
      const leftY = (row + topLeft / (topLeft - bottomLeft || 1e-9)) * cellHeight;

      const segment = (startX, startY, endX, endY) => {
        canvasContext.moveTo(startX, startY);
        canvasContext.lineTo(endX, endY);
      };

      if (caseIndex === 1 || caseIndex === 14) segment(topX, topY, leftX, leftY);
      else if (caseIndex === 2 || caseIndex === 13) segment(topX, topY, rightX, rightY);
      else if (caseIndex === 3 || caseIndex === 12) segment(leftX, leftY, rightX, rightY);
      else if (caseIndex === 4 || caseIndex === 11) segment(rightX, rightY, bottomX, bottomY);
      else if (caseIndex === 5) {
        segment(topX, topY, leftX, leftY);
        segment(rightX, rightY, bottomX, bottomY);
      } else if (caseIndex === 6 || caseIndex === 9) {
        segment(topX, topY, bottomX, bottomY);
      } else if (caseIndex === 7 || caseIndex === 8) {
        segment(leftX, leftY, bottomX, bottomY);
      } else if (caseIndex === 10) {
        segment(topX, topY, rightX, rightY);
        segment(bottomX, bottomY, leftX, leftY);
      }
    }
  }

  canvasContext.stroke();
  canvasContext.restore();
}

function drawPoint(point, width, height, domain) {
  const x = mapX(point.x, width, domain);
  const y = mapY(point.y, height, domain);
  const radius = Math.max(2.2, width / 260);
  const fill = point.label === 0 ? "rgba(85, 214, 255, 0.92)" : "rgba(255, 107, 214, 0.92)";

  canvasContext.save();
  canvasContext.beginPath();
  canvasContext.arc(x, y, radius, 0, Math.PI * 2);
  canvasContext.fillStyle = fill;
  canvasContext.fill();
  canvasContext.strokeStyle = "rgba(4, 10, 18, 0.78)";
  canvasContext.lineWidth = 0.8;
  canvasContext.stroke();
  canvasContext.restore();
}

function drawTestPoint(point, width, height, domain) {
  const x = mapX(point.x, width, domain);
  const y = mapY(point.y, height, domain);
  const radius = Math.max(3.6, width / 180);
  const color = point.label === 0 ? "#55d6ff" : "#ff6bd6";

  canvasContext.save();
  canvasContext.beginPath();
  canvasContext.moveTo(x, y - radius);
  canvasContext.lineTo(x + radius, y);
  canvasContext.lineTo(x, y + radius);
  canvasContext.lineTo(x - radius, y);
  canvasContext.closePath();
  canvasContext.fillStyle = "rgba(5, 12, 22, 0.86)";
  canvasContext.fill();
  canvasContext.strokeStyle = color;
  canvasContext.lineWidth = 1.25;
  canvasContext.stroke();
  canvasContext.restore();
}

function drawHover(width, height, domain) {
  if (!state.hover) return;
  const { dataX, dataY, label } = state.hover;
  const x = mapX(dataX, width, domain);
  const y = mapY(dataY, height, domain);

  canvasContext.save();
  canvasContext.strokeStyle = "rgba(255, 255, 255, 0.82)";
  canvasContext.lineWidth = 1;
  canvasContext.beginPath();
  canvasContext.arc(x, y, 8, 0, Math.PI * 2);
  canvasContext.stroke();
  canvasContext.beginPath();
  canvasContext.moveTo(x - 13, y);
  canvasContext.lineTo(x + 13, y);
  canvasContext.moveTo(x, y - 13);
  canvasContext.lineTo(x, y + 13);
  canvasContext.stroke();
  canvasContext.beginPath();
  canvasContext.arc(x, y, 3, 0, Math.PI * 2);
  canvasContext.fillStyle = label === 0 ? "#55d6ff" : "#ff6bd6";
  canvasContext.fill();
  canvasContext.restore();
}

function renderCanvas() {
  if (!state.model || !state.dataset) return;
  if (!resizeCanvas()) return;

  const width = canvasSize.width;
  const height = canvasSize.height;
  const domain = state.dataset.domain;

  for (let row = 0; row < contourResolution; row += 1) {
    const y = domain - ((row + 0.5) / contourResolution) * domain * 2;
    for (let column = 0; column < contourResolution; column += 1) {
      const x = -domain + ((column + 0.5) / contourResolution) * domain * 2;
      const probability = state.model.predictProba([x, y])[1];
      const confidence = Math.abs(probability - 0.5) * 2;
      const red = Math.round(35 + probability * 220);
      const green = Math.round(170 - probability * 78);
      const blue = Math.round(255 - probability * 91);
      const alpha = Math.round(28 + confidence * 90);
      const index = (row * contourResolution + column) * 4;
      contourImage.data[index] = red;
      contourImage.data[index + 1] = green;
      contourImage.data[index + 2] = blue;
      contourImage.data[index + 3] = alpha;
      contourField[row * contourResolution + column] = probability - 0.5;
    }
  }

  contourContext.putImageData(contourImage, 0, 0);
  canvasContext.clearRect(0, 0, width, height);
  canvasContext.save();
  canvasContext.imageSmoothingEnabled = true;
  canvasContext.drawImage(contourCanvas, 0, 0, width, height);
  canvasContext.restore();

  drawGrid(width, height, domain);
  drawBoundary(width, height, domain);

  const vignette = canvasContext.createRadialGradient(
    width * 0.5,
    height * 0.5,
    width * 0.1,
    width * 0.5,
    height * 0.5,
    width * 0.78,
  );
  vignette.addColorStop(0, "rgba(2, 6, 12, 0)");
  vignette.addColorStop(1, "rgba(2, 6, 12, 0.48)");
  canvasContext.fillStyle = vignette;
  canvasContext.fillRect(0, 0, width, height);

  for (const point of state.dataset.train) drawPoint(point, width, height, domain);
  if (state.showTest) {
    for (const point of state.dataset.test) drawTestPoint(point, width, height, domain);
  }
  drawHover(width, height, domain);
}

function handlePointerMove(event) {
  if (!state.model || !state.dataset) return;
  const rectangle = elements.decisionCanvas.getBoundingClientRect();
  const x = event.clientX - rectangle.left;
  const y = event.clientY - rectangle.top;
  if (x < 0 || y < 0 || x > rectangle.width || y > rectangle.height) {
    handlePointerLeave();
    return;
  }

  const domain = state.dataset.domain;
  const dataX = -domain + (x / rectangle.width) * domain * 2;
  const dataY = domain - (y / rectangle.height) * domain * 2;
  const probabilities = state.model.predictProba([dataX, dataY]);
  const label = probabilities[1] >= probabilities[0] ? 1 : 0;
  const confidence = Math.max(probabilities[0], probabilities[1]);

  state.hover = { dataX, dataY, label, confidence, probabilities };
  elements.hoverTooltip.innerHTML = `<strong>预测：类别 ${label === 0 ? "A" : "B"}</strong>置信度 ${(confidence * 100).toFixed(1)}%<br>P(A) ${(probabilities[0] * 100).toFixed(1)}% · P(B) ${(probabilities[1] * 100).toFixed(1)}%`;
  elements.hoverTooltip.hidden = false;

  const tooltipWidth = 150;
  const tooltipHeight = 62;
  let left = x + 15;
  let top = y + 15;
  if (left + tooltipWidth > rectangle.width) left = x - tooltipWidth - 15;
  if (top + tooltipHeight > rectangle.height) top = y - tooltipHeight - 15;
  elements.hoverTooltip.style.left = `${Math.max(8, left)}px`;
  elements.hoverTooltip.style.top = `${Math.max(8, top)}px`;
  renderCanvas();
}

function handlePointerLeave() {
  state.hover = null;
  elements.hoverTooltip.hidden = true;
  renderCanvas();
}

function syncControls() {
  elements.noiseRange.value = String(state.noise);
  elements.noiseValue.textContent = state.noise.toFixed(2);
  elements.sampleRange.value = String(state.sampleCount);
  elements.sampleValue.textContent = String(state.sampleCount);
  elements.lrRange.value = String(state.learningRate);
  elements.lrValue.textContent = state.learningRate.toFixed(3);
  elements.epochRange.value = String(state.maxEpochs);
  elements.epochValueMax.textContent = String(state.maxEpochs);
  elements.showTestToggle.checked = state.showTest;

  elements.datasetTabs.querySelectorAll("button").forEach((button) => {
    button.classList.toggle("active", button.dataset.dataset === state.kind);
  });

  elements.modelOptions.querySelectorAll("input[name='model']").forEach((input) => {
    input.checked = input.value === state.modelKey;
    input.closest(".model-option").classList.toggle("active", input.checked);
  });
}

function bindEvents() {
  elements.datasetTabs.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-dataset]");
    if (!button || button.dataset.dataset === state.kind) return;
    state.kind = button.dataset.dataset;
    state.seed = (state.seed + 1) >>> 0;
    state.initCounter = 0;
    syncControls();
    regenerateData({ toastMessage: `已切换到${DATASETS[state.kind].name}数据集。` });
  });

  elements.noiseRange.addEventListener("input", (event) => {
    state.noise = Number(event.target.value);
    elements.noiseValue.textContent = state.noise.toFixed(2);
  });

  elements.noiseRange.addEventListener("change", () => {
    state.seed = (state.seed + 1) >>> 0;
    state.initCounter = 0;
    regenerateData({ toastMessage: "噪声强度已更新。" });
  });

  elements.sampleRange.addEventListener("input", (event) => {
    state.sampleCount = Number(event.target.value);
    elements.sampleValue.textContent = String(state.sampleCount);
  });

  elements.sampleRange.addEventListener("change", () => {
    state.seed = (state.seed + 1) >>> 0;
    state.initCounter = 0;
    regenerateData({ toastMessage: "样本数量已更新。" });
  });

  elements.modelOptions.addEventListener("change", (event) => {
    if (event.target.name !== "model") return;
    state.modelKey = event.target.value;
    state.initCounter = 0;
    syncControls();
    resetModel({ quiet: true });
    showToast(`已切换到${MODEL_PRESETS[state.modelKey].name}。`);
  });

  elements.lrRange.addEventListener("input", (event) => {
    state.learningRate = Number(event.target.value);
    elements.lrValue.textContent = state.learningRate.toFixed(3);
  });

  elements.epochRange.addEventListener("input", (event) => {
    state.maxEpochs = Number(event.target.value);
    elements.epochValueMax.textContent = String(state.maxEpochs);
    updateMetrics();
    drawCharts();
  });

  elements.demoButton.addEventListener("click", runDemo);
  elements.trainButton.addEventListener("click", toggleTraining);
  elements.resetButton.addEventListener("click", () => resetModel({ freshInit: true }));
  elements.regenerateButton.addEventListener("click", () => {
    state.seed = (state.seed + 1) >>> 0;
    state.initCounter = 0;
    regenerateData();
  });

  elements.showTestToggle.addEventListener("change", (event) => {
    state.showTest = event.target.checked;
    renderCanvas();
  });

  elements.decisionCanvas.addEventListener("pointermove", handlePointerMove);
  elements.decisionCanvas.addEventListener("pointerleave", handlePointerLeave);
  elements.exportImageButton.addEventListener("click", exportScreenshot);
  elements.exportJsonButton.addEventListener("click", exportExperiment);

  window.addEventListener("resize", () => {
    state.hover = null;
    elements.hoverTooltip.hidden = true;
    renderCanvas();
    drawCharts();
  });

  if (window.ResizeObserver) {
    const observer = new ResizeObserver(() => {
      state.hover = null;
      elements.hoverTooltip.hidden = true;
      renderCanvas();
    });
    observer.observe(elements.canvasWrap);
  }
}

function downloadBlob(blob, fileName) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportScreenshot() {
  elements.decisionCanvas.toBlob((blob) => {
    if (!blob) {
      showToast("截图导出失败，请重试。");
      return;
    }
    downloadBlob(blob, `neural-playground-${state.kind}-epoch-${state.epoch}.png`);
    showToast("决策边界截图已导出。");
  }, "image/png");
}

function exportExperiment() {
  if (!state.model || !state.dataset) return;
  const payload = {
    project: "Neural Playground",
    exportedAt: new Date().toISOString(),
    dataset: {
      kind: state.dataset.kind,
      name: state.dataset.name,
      noise: state.dataset.noise,
      seed: state.dataset.seed,
      trainSamples: state.dataset.train.length,
      testSamples: state.dataset.test.length,
    },
    model: {
      preset: state.modelKey,
      name: MODEL_PRESETS[state.modelKey].name,
      architecture: state.model.architecture(),
      parameters: state.model.countParameters(),
      learningRate: state.learningRate,
      epochs: state.epoch,
    },
    metrics: {
      train: state.trainMetrics,
      test: state.testMetrics,
    },
    history: state.history,
  };
  downloadBlob(
    new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }),
    `neural-playground-${state.kind}-${state.seed}.json`,
  );
  showToast("实验数据已导出为 JSON。");
}

function initialize() {
  charts.loss = new LineChart(elements.lossChart, "loss");
  charts.accuracy = new LineChart(elements.accuracyChart, "accuracy");
  bindEvents();
  syncControls();
  state.dataset = createDataset(state.kind, state.sampleCount, state.noise, state.seed);
  buildModel();
  updateAll();
}

initialize();
