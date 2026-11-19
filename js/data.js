import { gaussian, mulberry32, shuffleInPlace } from "./random.js";

export const DATASETS = {
  moons: {
    name: "双月牙",
    hint: "两个交错的月牙",
    tip: "双月牙需要弯曲的决策边界。线性模型只能画直线，MLP 才能把两类分得更开。",
  },
  circles: {
    name: "同心圆",
    hint: "一圈包住另一圈",
    tip: "同心圆考验模型对距离的理解。浅层网络通常就能学得不错。",
  },
  spiral: {
    name: "螺旋",
    hint: "两条相互缠绕的臂",
    tip: "螺旋是更难的数据集。试试深层 MLP 和更多训练轮数，观察边界如何卷曲。",
  },
  xor: {
    name: "XOR",
    hint: "对角线上的两类",
    tip: "XOR 是经典的线性不可分问题。没有隐藏层的模型无法正确分开四个区域。",
  },
};

function makePoint(x, y, label) {
  return { x, y, label };
}

function moonsPoint(label, noise, random) {
  const angle = Math.PI * random();
  let x = Math.cos(angle);
  let y = Math.sin(angle);

  if (label === 1) {
    x = 1 - x;
    y = 0.5 - y;
  }

  x -= 0.5;
  y -= 0.25;
  x += gaussian(random) * noise * 0.34;
  y += gaussian(random) * noise * 0.34;
  return makePoint(x, y, label);
}

function circlesPoint(label, noise, random) {
  const radius = label === 0 ? 0.52 : 1.08;
  const angle = Math.PI * 2 * random();
  const jitter = noise * 0.24;
  return makePoint(
    Math.cos(angle) * radius + gaussian(random) * jitter,
    Math.sin(angle) * radius + gaussian(random) * jitter,
    label,
  );
}

function spiralPoint(label, noise, random) {
  const progress = 0.12 + 0.88 * random();
  const radius = 1.22 * progress;
  const angle = Math.PI * 2.9 * progress + label * Math.PI;
  const jitter = noise * 0.18;
  return makePoint(
    Math.cos(angle) * radius + gaussian(random) * jitter,
    Math.sin(angle) * radius + gaussian(random) * jitter,
    label,
  );
}

function xorPoint(noise, random) {
  const x = random() * 2 - 1 + gaussian(random) * noise * 0.16;
  const y = random() * 2 - 1 + gaussian(random) * noise * 0.16;
  let label = x * y >= 0 ? 0 : 1;
  if (random() < noise * 0.18) label = 1 - label;
  return makePoint(x, y, label);
}

function generateRaw(kind, count, noise, random) {
  const points = [];
  const evenCount = count % 2 === 0 ? count : count + 1;

  for (let index = 0; index < evenCount; index += 1) {
    if (kind === "circles") {
      points.push(circlesPoint(index % 2, noise, random));
    } else if (kind === "spiral") {
      points.push(spiralPoint(index % 2, noise, random));
    } else if (kind === "xor") {
      points.push(xorPoint(noise, random));
    } else {
      points.push(moonsPoint(index % 2, noise, random));
    }
  }

  return points;
}

export function createDataset(kind, sampleCount, noise, seed) {
  const definition = DATASETS[kind] || DATASETS.moons;
  const train = generateRaw(kind, Math.max(80, Math.round(sampleCount)), noise, mulberry32(seed));
  const testCount = Math.max(80, Math.round(train.length * 0.35));
  const test = generateRaw(kind, testCount, noise, mulberry32(seed + 7919));

  shuffleInPlace(train, mulberry32(seed + 101));
  shuffleInPlace(test, mulberry32(seed + 202));

  return {
    kind,
    name: definition.name,
    hint: definition.hint,
    tip: definition.tip,
    domain: 1.55,
    noise,
    seed,
    train,
    test,
  };
}
