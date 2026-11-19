const COLORS = {
  grid: "rgba(148, 178, 225, 0.1)",
  axis: "rgba(148, 178, 225, 0.35)",
  text: "rgba(169, 186, 212, 0.75)",
  train: "#55d6ff",
  test: "#ff6bd6",
  placeholder: "rgba(113, 131, 160, 0.75)",
};

function formatTick(value, type) {
  if (type === "accuracy") return `${Math.round(value * 100)}%`;
  if (value >= 10) return value.toFixed(0);
  return value.toFixed(2);
}

export class LineChart {
  constructor(canvas, type) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.type = type;
    this.width = 0;
    this.height = 0;
    this.pixelRatio = 1;
    this.padding = { left: 35, right: 9, top: 13, bottom: 22 };
  }

  resize() {
    const rectangle = this.canvas.getBoundingClientRect();
    if (rectangle.width === 0 || rectangle.height === 0) return;

    const pixelRatio = window.devicePixelRatio || 1;
    const width = Math.round(rectangle.width);
    const height = Math.round(rectangle.height);

    if (width !== this.width || height !== this.height || pixelRatio !== this.pixelRatio) {
      this.width = width;
      this.height = height;
      this.pixelRatio = pixelRatio;
      this.canvas.width = Math.round(width * pixelRatio);
      this.canvas.height = Math.round(height * pixelRatio);
      this.ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    }
  }

  draw(history, maxEpoch) {
    this.resize();
    if (!this.width || !this.height) return;

    const context = this.ctx;
    context.clearRect(0, 0, this.width, this.height);

    const plotLeft = this.padding.left;
    const plotTop = this.padding.top;
    const plotWidth = this.width - this.padding.left - this.padding.right;
    const plotHeight = this.height - this.padding.top - this.padding.bottom;

    if (plotWidth <= 0 || plotHeight <= 0) return;

    const values = [];
    for (const point of history) {
      if (this.type === "accuracy") {
        values.push(point.trainAcc, point.testAcc);
      } else {
        values.push(point.trainLoss, point.testLoss);
      }
    }

    const yMax = this.type === "accuracy"
      ? 1
      : Math.max(0.7, ...values.map((value) => (Number.isFinite(value) ? value : 0))) * 1.08;

    context.save();
    context.strokeStyle = COLORS.grid;
    context.lineWidth = 1;
    context.font = "8px Cascadia Code, Consolas, monospace";
    context.fillStyle = COLORS.text;

    for (let tick = 0; tick <= 4; tick += 1) {
      const ratio = tick / 4;
      const y = plotTop + plotHeight - ratio * plotHeight;
      context.beginPath();
      context.moveTo(plotLeft, y);
      context.lineTo(plotLeft + plotWidth, y);
      context.stroke();
      context.textAlign = "right";
      context.textBaseline = "middle";
      context.fillText(formatTick(yMax * ratio, this.type), plotLeft - 6, y);
    }

    for (let tick = 0; tick <= 2; tick += 1) {
      const ratio = tick / 2;
      const x = plotLeft + ratio * plotWidth;
      context.beginPath();
      context.moveTo(x, plotTop);
      context.lineTo(x, plotTop + plotHeight);
      context.stroke();
      context.textAlign = tick === 0 ? "left" : tick === 2 ? "right" : "center";
      context.textBaseline = "top";
      context.fillText(String(Math.round(maxEpoch * ratio)), x, plotTop + plotHeight + 7);
    }

    context.strokeStyle = COLORS.axis;
    context.beginPath();
    context.moveTo(plotLeft, plotTop);
    context.lineTo(plotLeft, plotTop + plotHeight);
    context.lineTo(plotLeft + plotWidth, plotTop + plotHeight);
    context.stroke();
    context.restore();

    if (history.length === 0) {
      context.save();
      context.setLineDash([4, 5]);
      context.strokeStyle = "rgba(148, 178, 225, 0.16)";
      context.beginPath();
      context.moveTo(plotLeft + 8, plotTop + plotHeight * 0.48);
      context.lineTo(plotLeft + plotWidth - 8, plotTop + plotHeight * 0.48);
      context.stroke();
      context.setLineDash([]);
      context.fillStyle = COLORS.placeholder;
      context.font = "9px Segoe UI, Microsoft YaHei, sans-serif";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText("?????????", plotLeft + plotWidth / 2, plotTop + plotHeight * 0.48);
      context.restore();
      return;
    }

    const lastEpoch = history[history.length - 1].epoch;
    const xMax = Math.max(1, maxEpoch, lastEpoch);
    const drawSeries = (key, color) => {
      context.save();
      context.beginPath();
      context.lineWidth = 1.8;
      context.lineJoin = "round";
      context.lineCap = "round";
      context.strokeStyle = color;
      context.shadowColor = color;
      context.shadowBlur = 7;

      history.forEach((point, index) => {
        const x = plotLeft + (point.epoch / xMax) * plotWidth;
        const rawValue = Math.min(yMax, Math.max(0, point[key]));
        const y = plotTop + plotHeight - (rawValue / yMax) * plotHeight;
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      context.stroke();
      context.restore();

      const last = history[history.length - 1];
      const x = plotLeft + (last.epoch / xMax) * plotWidth;
      const rawValue = Math.min(yMax, Math.max(0, last[key]));
      const y = plotTop + plotHeight - (rawValue / yMax) * plotHeight;
      context.save();
      context.fillStyle = color;
      context.shadowColor = color;
      context.shadowBlur = 9;
      context.beginPath();
      context.arc(x, y, 2.2, 0, Math.PI * 2);
      context.fill();
      context.restore();
    };

    drawSeries(this.type === "accuracy" ? "trainAcc" : "trainLoss", COLORS.train);
    drawSeries(this.type === "accuracy" ? "testAcc" : "testLoss", COLORS.test);
  }
}
