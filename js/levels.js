import { DATASETS } from "./data.js";
import { MODEL_PRESETS } from "./network.js";

const CH1 = "第 1 章 · 先看懂问题";
const CH2 = "第 2 章 · 看懂网络内部";
const CH3 = "第 3 章 · 学会调参";
const CH4 = "第 4 章 · 独立挑战";

const datasetButton = (kind) => ({ target: "dataset", value: kind, label: `先切到「${DATASETS[kind].name}」数据集` });
const modelOption = (key) => ({ target: "model", value: key, label: `先选择「${MODEL_PRESETS[key].name}」` });

export const LEVELS = [
  {
    id: "train-first",
    chapter: CH1,
    title: "让机器从例子里学会一件事",
    question: "你不需要写规则。给机器一堆带答案的例子，它会自己调整参数，把两类点分开。",
    setup: { kind: "moons", modelKey: "small", noise: 0.15, sampleCount: 400, maxEpochs: 600, learningRate: 0.01 },
    steps: [
      { text: "点「开始训练」，让模型先学一会儿", check: (s) => s.epoch >= 10 },
      { text: "等训练走完，看类 A 和类 B 之间出现了一条边界", check: (s) => s.finished && s.testAcc >= 0.9 },
    ],
    conclusion: "刚才发生的事就是「训练」：模型先随机猜，再看自己错在哪里，然后微调参数。没有人告诉它「月牙要弯着分」，这条边界是它自己从数据里学出来的。",
  },
  {
    id: "line-fails",
    chapter: CH1,
    title: "一条直线，切不开一个圈",
    question: "同心圆里，一类点包住另一类点。现在只用最简单的线性模型，它能分开吗？",
    setup: { kind: "circles", modelKey: "linear", noise: 0.12, sampleCount: 320, maxEpochs: 300, learningRate: 0.01 },
    focus: (s) => {
      if (s.kind !== "circles") return datasetButton("circles");
      if (s.modelKey !== "linear") return modelOption("linear");
      if (!s.finished && !s.training) return { target: "train", label: "点「开始训练」" };
      return null;
    },
    steps: [
      { text: "保持「同心圆 + 线性分类器」，点「开始训练」", check: (s) => s.finished && s.kind === "circles" && s.modelKey === "linear" },
      { text: "看测试准确率，是不是只有一半左右？", hint: "准确率接近 50%，说明模型基本在瞎猜。", check: (s) => s.finished && s.testAcc < 0.8 },
    ],
    conclusion: "线性模型只能画一条直线。直线无论怎么摆，都不可能把一个圆分成里外两部分。这种数据叫「线性不可分」——这不是模型没训练好，而是它的能力上限。",
  },
  {
    id: "add-hidden-layer",
    chapter: CH1,
    title: "加一层隐藏层，边界就弯了",
    question: "换成一个带隐藏层的小型 MLP，同样的数据，结果会不一样吗？",
    setup: { kind: "circles", modelKey: "small", noise: 0.12, sampleCount: 320, maxEpochs: 600, learningRate: 0.012 },
    steps: [
      { text: "点「开始训练」，这次让小型 MLP 来学", check: (s) => s.finished && s.kind === "circles" && s.modelKey !== "linear" },
      { text: "测试准确率超过 95%", check: (s) => s.testAcc >= 0.95 },
    ],
    conclusion: "隐藏层里的神经元会先把平面做一次「变形」，再在变形后的空间里画直线。回到原来的平面，这条线就弯了。深度学习的「深」，说的就是这种一层层的变换。",
  },
  {
    id: "neuron-probability",
    chapter: CH2,
    title: "一个点进去，两个概率出来",
    question: "训练好的模型看到一个新点，会给出什么答案？把鼠标移到图上试试。",
    setup: { kind: "moons", modelKey: "small", noise: 0.15, sampleCount: 400, maxEpochs: 600, learningRate: 0.01 },
    focus: (s, c) => {
      if (s.epoch < 100) return { target: "train", label: "先训练到 100 轮以上" };
      if (!c.hovered) return { target: "canvas", label: "把鼠标移到图上" };
      return null;
    },
    steps: [
      { text: "训练至少 100 轮，让模型有点本事", check: (s) => s.epoch >= 100 },
      { text: "把鼠标移到图上任意位置，读出 P(A) 和 P(B)", hint: "两个概率加起来永远是 100%。越接近 100%，模型越有把握。", check: (s, c) => c.hovered },
      { text: "我看到了：模型输出的是概率，不是硬编码的答案", manual: true },
    ],
    conclusion: "网络最后用 softmax 把两个分数变成概率。所谓「模型预测是 A」，其实就是 P(A) 比 P(B) 大。概率越接近 50%，说明这个位置越模糊，模型越没把握。",
  },
  {
    id: "read-architecture",
    chapter: CH2,
    title: "读懂 2 → 16 → 8 → 2",
    question: "模型信息卡片里有一串数字。它们不是随便写的，每一层都在做一件事。",
    setup: { kind: "moons", modelKey: "deep", noise: 0.15, sampleCount: 400, maxEpochs: 400, learningRate: 0.01 },
    focus: (s) => (s.modelKey !== "deep" ? modelOption("deep") : null),
    steps: [
      { text: "在「模型结构」里选中「深层 MLP」", check: (s) => s.modelKey === "deep" },
      { text: "看模型信息：结构变成 2 → 16 → 8 → 2，参数量超过 150", hint: "2 是输入的 x 和 y，16 和 8 是两层隐藏神经元，最后的 2 是两类概率。", check: (s) => s.modelKey === "deep" && s.params >= 150 },
      { text: "我明白了：数字代表每一层的神经元个数", manual: true },
    ],
    conclusion: "参数就是神经元之间的连接权重。层数越多、神经元越多，模型能表达的曲线越复杂，但需要的样本和计算也越多。这就是「模型容量」。",
  },
  {
    id: "epochs",
    chapter: CH2,
    title: "训练轮数：同一批数据看几遍",
    question: "损失曲线一开始掉得很快，后来变平。为什么？",
    setup: { kind: "moons", modelKey: "small", noise: 0.15, sampleCount: 400, maxEpochs: 600, learningRate: 0.01 },
    openAdvanced: true,
    focus: (s) => (!s.finished && !s.training ? { target: "train", label: "训练到 600 轮结束" } : null),
    steps: [
      { text: "确认最大训练轮数是 600，然后训练到结束", check: (s) => s.finished && s.maxEpochs >= 600 },
      { text: "观察损失曲线：前面陡，后面平", manual: true },
    ],
    conclusion: "一轮（epoch）就是把训练集完整看一遍。刚开始错误多，改得快；后来能改的地方越来越少，曲线就平了。所以不是轮数越多越好，够用就停。",
  },
  {
    id: "learning-rate",
    chapter: CH3,
    title: "学习率：每一步迈多大",
    question: "同样是 200 轮，把学习率调小十倍，结果会有什么不同？",
    setup: { kind: "moons", modelKey: "small", noise: 0.15, sampleCount: 400, maxEpochs: 200, learningRate: 0.01 },
    openAdvanced: true,
    focus: (s, c) => {
      const low = c.runs.some((r) => r.finished && r.learningRate <= 0.002);
      const high = c.runs.some((r) => r.finished && r.learningRate >= 0.02);
      if (!low) return { target: "lr", label: "把学习率调到 0.002 左右" };
      if (!high) return { target: "lr", label: "再把学习率调到 0.02 左右" };
      return null;
    },
    steps: [
      { text: "把学习率调到 0.002 以下，训练 200 轮", hint: "训练结束后，把学习率滑块往左拖到 0.002。", check: (s, c) => c.runs.some((r) => r.finished && r.learningRate <= 0.002) },
      { text: "再把学习率调到 0.02 以上，重新训练 200 轮", hint: "同一批数据、同一个模型，只改学习率，方便对比。", check: (s, c) => c.runs.some((r) => r.finished && r.learningRate >= 0.02) },
      { text: "对比两次的损失曲线和收敛速度", manual: true },
    ],
    conclusion: "学习率决定每次更新参数时迈多大一步。太小，200 轮走不到好位置；太大，可能来回跳过最低点。调参的第一件事，往往就是找一个合适的学习率。",
  },
  {
    id: "data-quality",
    chapter: CH3,
    title: "标签错了，模型再强也没用",
    question: "训练集里有 30% 的答案本身标错了。模型还能考高分吗？",
    setup: { kind: "moons", modelKey: "deep", noise: 0.2, labelNoise: 0.3, sampleCount: 320, maxEpochs: 600, learningRate: 0.01 },
    focus: (s, c) => {
      const base = c.runs.find((r) => r.finished && r.labelNoise >= 0.25);
      if (!base) return { target: "train", label: "先用当前设置训练一次" };
      const fixed = c.runs.some((r) => r.finished && r.labelNoise <= 0.02 && (r.testAcc >= base.testAcc + 0.05 || r.testAcc >= 0.95));
      if (!fixed) return { target: "labelNoise", label: "把错误标签比例调回 0" };
      return null;
    },
    steps: [
      { text: "用当前设置训练到结束（30% 的训练标签是错的）", check: (s, c) => c.runs.some((r) => r.finished && r.labelNoise >= 0.25) },
      { text: "看测试准确率：是不是卡在 90% 左右，怎么都上不去", hint: "测试集是干净的，但模型是按错误答案学的。无论网络多深、训练多久，都绕不过错误标签带来的上限。", check: (s, c) => c.runs.some((r) => r.finished && r.labelNoise >= 0.25 && r.testAcc < 0.95) },
      { text: "把错误标签比例调回 0，再训练一次，测试准确率超过 95%", check: (s, c) => {
        const base = c.runs.find((r) => r.finished && r.labelNoise >= 0.25);
        if (!base) return false;
        return c.runs.some((r) => r.finished && r.labelNoise <= 0.02 && (r.testAcc >= base.testAcc + 0.05 || r.testAcc >= 0.95));
      } },
    ],
    conclusion: "模型是跟着训练数据学的。标签里混进 30% 的错误，它就会学到这些错误规律，测试准确率被压在一个上不去的高度。把错误标签清掉，同样的模型立刻回到 99% 以上。所以调参之前，先检查数据。",
  },
  {
    id: "spiral",
    chapter: CH4,
    title: "螺旋：把边界卷起来",
    question: "两条互相缠绕的螺旋，是最难的分法。现在轮到你自己调了。",
    setup: { kind: "spiral", modelKey: "deep", noise: 0.15, sampleCount: 480, maxEpochs: 800, learningRate: 0.015 },
    focus: (s) => (!s.finished && !s.training ? { target: "train", label: "点「开始训练」" } : null),
    steps: [
      { text: "训练到结束", check: (s) => s.finished },
      { text: "测试准确率达到 97% 以上", hint: "如果卡住了，试试增加轮数、把学习率调到 0.01 附近，或者多训练一次。", check: (s) => s.testAcc >= 0.97 },
    ],
    conclusion: "螺旋几乎要求边界贴着数据卷进去。这需要足够的隐藏层、足够的轮数，也需要一点耐心调参。能把螺旋分开，说明你已经掌握了这个实验台的核心。",
  },
  {
    id: "final",
    chapter: CH4,
    title: "毕业挑战：自己设计一次实验",
    question: "不限数据集、不限模型。把测试准确率推到 99% 以上，并说清楚你为什么选这套配置。",
    setup: {},
    focus: (s) => (!s.finished && !s.training ? { target: "train", label: "自由选择配置后开始训练" } : null),
    steps: [
      { text: "任选数据集和模型，测试准确率达到 99%", check: (s) => s.testAcc >= 0.99 },
      { text: "选一句最准确的解释", quiz: {
        options: [
          "数据点更多，所以必须用深层网络",
          "螺旋的边界要跟着两条臂弯曲缠绕，浅层模型表达不了这么复杂的形状",
          "只要把学习率调大，任何模型都能分开螺旋",
        ],
        answer: 1,
        wrongHint: "再想想：决定需要多深的是边界的复杂程度，不是数据点的数量。",
      } },
    ],
    conclusion: "你已经走完了「看懂问题 → 看懂网络 → 学会调参 → 独立设计实验」这条路。以后再遇到新的数据，你会先问：它线性可分吗？需要多大容量？训练集够不够？——这就是机器学习工程师的思考方式。",
  },
];

export const RANKS = [
  { min: 0, name: "观察员" },
  { min: 200, name: "实验员" },
  { min: 400, name: "调参师" },
  { min: 600, name: "网络训练师" },
  { min: 800, name: "神经网络理解者" },
];

const STORAGE_KEY = "neural-playground:levels:v1";

function loadProgress() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    const index = Number.isInteger(parsed.index) ? Math.min(Math.max(parsed.index, 0), LEVELS.length - 1) : 0;
    const completed = Array.isArray(parsed.completed) ? parsed.completed.filter((id) => LEVELS.some((level) => level.id === id)) : [];
    return { index, completed, xp: completed.length * 100 };
  } catch (error) {
    return null;
  }
}

function saveProgress(progress) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({
      index: progress.index,
      completed: progress.completed,
    }));
  } catch (error) {
    /* localStorage 不可用时只保留当前会话进度 */
  }
}

export class Tutor {
  constructor(options) {
    this.getSnapshot = options.getSnapshot;
    this.applySetup = options.applySetup;
    this.onNavigate = options.onNavigate || (() => {});
    this.root = options.root;
    this.badge = options.badge;
    this.refs = {
      chapter: document.getElementById("tutorChapter"),
      title: document.getElementById("tutorTitle"),
      index: document.getElementById("tutorIndex"),
      question: document.getElementById("tutorQuestion"),
      steps: document.getElementById("tutorSteps"),
      hint: document.getElementById("tutorHint"),
      hintButton: document.getElementById("tutorHintButton"),
      setupButton: document.getElementById("tutorSetupButton"),
      skipButton: document.getElementById("tutorSkipButton"),
      action: document.getElementById("tutorAction"),
      nextButton: document.getElementById("tutorNextButton"),
      mapButton: document.getElementById("tutorMapButton"),
      map: document.getElementById("tutorMap"),
      result: document.getElementById("tutorResult"),
      exitButton: document.getElementById("tutorExitButton"),
      resumeButton: document.getElementById("tutorResume"),
      body: document.getElementById("tutorBody"),
      badgeText: document.getElementById("tutorBadgeText"),
      badgeRank: document.getElementById("tutorBadgeRank"),
    };

    const saved = loadProgress();
    this.progress = saved || { index: 0, completed: [], xp: 0 };
    this.session = { manual: new Set(), quiz: new Set(), quizWrong: null, hovered: false, runs: [], hintOpen: false };
    this.suspended = false;
    this.signature = "";
    this.bound = false;

    this.bind();
    this.enter(this.progress.index, { applySetup: true, quiet: true });
    if (this.progress.completed.length === LEVELS.length) {
      this.suspended = true;
      this.render();
    }
  }

  bind() {
    if (this.bound) return;
    this.bound = true;

    this.refs.hintButton.addEventListener("click", () => {
      this.session.hintOpen = !this.session.hintOpen;
      this.refresh(true);
    });

    this.refs.setupButton.addEventListener("click", () => {
      this.applyLevelSetup();
      this.session.manual.clear();
      this.refresh(true);
    });

    this.refs.skipButton.addEventListener("click", () => {
      if (this.progress.index >= LEVELS.length - 1) {
        this.suspended = true;
        this.render();
        return;
      }
      this.session.manual.clear();
      this.session.hintOpen = false;
      this.enter(Math.min(this.progress.index + 1, LEVELS.length - 1), { applySetup: true, scroll: true });
    });

    this.refs.nextButton.addEventListener("click", () => {
      if (this.progress.index >= LEVELS.length - 1) {
        this.suspended = true;
        this.render();
        this.onNavigate({ type: "graduated" });
        return;
      }
      this.session.manual.clear();
      this.session.hintOpen = false;
      this.enter(this.progress.index + 1, { applySetup: true, scroll: true });
    });

    this.refs.action.addEventListener("click", () => {
      const trainButton = document.getElementById("trainButton");
      if (trainButton) trainButton.click();
      this.refresh(true);
    });

    this.refs.mapButton.addEventListener("click", () => {
      const open = this.refs.map.hidden;
      this.refs.map.hidden = !open;
      this.refs.mapButton.setAttribute("aria-expanded", String(open));
    });

    this.refs.steps.addEventListener("click", (event) => {
      const quizButton = event.target.closest("button[data-quiz-step]");
      if (quizButton) {
        const stepIndex = Number(quizButton.dataset.quizStep);
        const optionIndex = Number(quizButton.dataset.quizOption);
        const step = this.level.steps[stepIndex];
        if (step && step.quiz && optionIndex === step.quiz.answer) {
          this.session.quiz.add(stepIndex);
          this.session.quizWrong = null;
        } else {
          this.session.quizWrong = stepIndex;
        }
        this.refresh(true);
        return;
      }
      const button = event.target.closest("button[data-manual-step]");
      if (!button) return;
      this.session.manual.add(Number(button.dataset.manualStep));
      this.refresh(true);
    });

    this.refs.map.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-level]");
      if (!button) return;
      this.session.manual.clear();
      this.session.hintOpen = false;
      this.refs.map.hidden = true;
      this.refs.mapButton.setAttribute("aria-expanded", "false");
      this.enter(Number(button.dataset.level), { applySetup: true, scroll: true });
    });

    this.refs.exitButton.addEventListener("click", () => {
      this.suspended = true;
      this.render();
    });

    this.refs.resumeButton.addEventListener("click", () => {
      this.suspended = false;
      this.refresh(true);
    });

    this.badge.addEventListener("click", () => {
      if (!this.root.isConnected) return;
      this.focusCard();
    });
  }

  focusCard() {
    if (!this.root) return;
    this.suspended = false;
    this.refresh(true);
    this.root.scrollIntoView({ behavior: "smooth", block: "center" });
    this.root.classList.remove("tutor-flash");
    window.requestAnimationFrame(() => this.root.classList.add("tutor-flash"));
    window.setTimeout(() => this.root.classList.remove("tutor-flash"), 1400);
  }

  get level() {
    return LEVELS[this.progress.index];
  }

  rank() {
    let name = RANKS[0].name;
    for (const item of RANKS) {
      if (this.progress.xp >= item.min) name = item.name;
    }
    return name;
  }

  enter(index, options = {}) {
    this.progress.index = Math.min(Math.max(index, 0), LEVELS.length - 1);
    this.session.manual = new Set();
    this.session.quiz = new Set();
    this.session.quizWrong = null;
    this.session.hovered = false;
    this.session.hintOpen = false;
    this.suspended = false;
    saveProgress(this.progress);
    if (options.applySetup) this.applyLevelSetup(Boolean(options.quiet));
    const panel = this.root.closest(".controls");
    if (panel && panel.scrollHeight > panel.clientHeight) panel.scrollTo({ top: 0, behavior: "smooth" });
    this.signature = "";
    this.refresh(true);
    this.onNavigate({ type: "level", index: this.progress.index });
  }

  applyLevelSetup(quiet = false) {
    const level = this.level;
    if (!level.setup || !Object.keys(level.setup).length) return;
    this.applySetup(level.setup, level, { quiet });
    if (level.openAdvanced) {
      const advanced = document.querySelector(".advanced-params");
      if (advanced) advanced.open = true;
    }
  }

  noteHover() {
    if (this.session.hovered) return;
    this.session.hovered = true;
    this.refresh();
  }

  recordRun(run) {
    this.session.runs.push({ ...run });
    if (this.session.runs.length > 24) this.session.runs.shift();
    this.refresh();
  }

  stepDone(step, index, snapshot) {
    if (step.quiz) return this.session.quiz.has(index);
    if (step.manual) return this.session.manual.has(index);
    try {
      return Boolean(step.check(snapshot, this.session));
    } catch (error) {
      return false;
    }
  }

  refresh(force = false) {
    if (!this.root) return;
    const snapshot = this.getSnapshot();
    const level = this.level;
    const doneFlags = level.steps.map((step, index) => this.stepDone(step, index, snapshot));
    const complete = doneFlags.every(Boolean);
    if (complete) this.completeLevel();
    const signature = [level.id, doneFlags.join(","), this.suspended, this.session.hintOpen, this.progress.completed.length, this.session.hovered].join("|");
    if (!force && signature === this.signature) {
      this.highlightNext(snapshot, complete);
      return;
    }
    this.signature = signature;
    this.render(snapshot, doneFlags, complete);
    this.highlightNext(snapshot, complete);
  }

  completeLevel() {
    const level = this.level;
    if (this.progress.completed.includes(level.id)) return;
    this.progress.completed.push(level.id);
    this.progress.xp = this.progress.completed.length * 100;
    saveProgress(this.progress);
    this.onNavigate({ type: "complete", index: this.progress.index, level });
  }

  render(snapshot = this.getSnapshot(), doneFlags = [], complete = false) {
    const level = this.level;
    const refs = this.refs;
    this.root.classList.toggle("is-suspended", this.suspended);
    if (refs.body) refs.body.hidden = this.suspended;
    if (this.suspended && refs.action) refs.action.hidden = true;
    if (refs.resumeButton) {
      refs.resumeButton.hidden = !this.suspended;
      refs.resumeButton.textContent = `继续学习 · 第 ${this.progress.index + 1} 关`;
    }

    refs.chapter.textContent = level.chapter;
    refs.title.textContent = level.title;
    refs.index.textContent = `${this.progress.index + 1} / ${LEVELS.length}`;
    refs.question.textContent = level.question;
    refs.steps.innerHTML = level.steps.map((step, index) => {
      const done = doneFlags[index];
      const action = step.manual && !done
        ? `<button class="tutor-confirm" type="button" data-manual-step="${index}">我看到了</button>`
        : "";
      let quiz = "";
      if (step.quiz && !done) {
        const options = step.quiz.options.map((option, optionIndex) => `<button class="tutor-quiz-option" type="button" data-quiz-step="${index}" data-quiz-option="${optionIndex}">${option}</button>`).join("");
        const wrong = this.session.quizWrong === index ? `<p class="tutor-quiz-hint">${step.quiz.wrongHint}</p>` : "";
        quiz = `<div class="tutor-quiz">${options}${wrong}</div>`;
      }
      return `<li class="tutor-step${done ? " is-done" : ""}"><span class="tutor-check" aria-hidden="true">${done ? "✓" : index + 1}</span><span class="tutor-step-text">${step.text}</span>${action}${quiz}</li>`;
    }).join("");

    const hint = level.steps.find((step) => step.hint && !doneFlags[level.steps.indexOf(step)]);
    refs.hintButton.hidden = !hint;
    refs.hint.hidden = !hint || !this.session.hintOpen;
    refs.hint.textContent = hint ? hint.hint : "";

    refs.result.hidden = !complete;
    if (complete) {
      refs.result.innerHTML = `<strong>本关结论</strong><p>${level.conclusion}</p>`;
    }

    refs.nextButton.hidden = !complete;
    refs.nextButton.textContent = this.progress.index >= LEVELS.length - 1 ? "完成全部关卡" : "下一关 →";
    refs.skipButton.hidden = complete;

    refs.badgeText.textContent = `${this.progress.completed.length} / ${LEVELS.length}`;
    refs.badgeRank.textContent = this.rank();
    this.badge.classList.toggle("is-complete", this.progress.completed.length === LEVELS.length);

    this.renderMap(doneFlags, complete);
  }

  renderMap() {
    const current = this.progress.index;
    this.refs.map.innerHTML = LEVELS.map((level, index) => {
      const done = this.progress.completed.includes(level.id);
      const classes = ["tutor-map-item"];
      if (done) classes.push("is-done");
      if (index === current) classes.push("is-current");
      return `<button class="${classes.join(" ")}" type="button" data-level="${index}">
        <span class="tutor-map-index">${done ? "✓" : index + 1}</span>
        <span class="tutor-map-copy"><strong>${level.title}</strong><small>${level.chapter}</small></span>
      </button>`;
    }).join("") + `<div class="tutor-map-footer"><span>${this.progress.xp} / ${LEVELS.length * 100} XP · ${this.rank()}</span><button type="button" id="tutorResetProgress">重置进度</button></div>`;
    const reset = this.refs.map.querySelector("#tutorResetProgress");
    if (reset) {
      reset.addEventListener("click", () => {
        if (!window.confirm("重置全部关卡进度？已完成的关卡会重新锁上。")) return;
        this.progress = { index: 0, completed: [], xp: 0 };
        saveProgress(this.progress);
        this.enter(0, { applySetup: true, scroll: true });
      });
    }
  }

  highlightNext(snapshot, complete) {
    if (this.previousTarget) {
      this.previousTarget.classList.remove("tutor-target");
      this.previousTarget = null;
    }
    if (complete || this.suspended) {
      this.renderAction(null);
      return;
    }
    const action = typeof this.level.focus === "function" ? this.level.focus(snapshot, this.session) : this.defaultFocus(snapshot);
    this.renderAction(action);
    if (!action) return;
    const element = this.resolveTarget(action);
    if (!element) return;
    let target = element;
    if (action.target === "model") {
      const label = element.closest(".model-option");
      if (label) target = label;
    }
    target.classList.add("tutor-target");
    this.previousTarget = target;
  }

  defaultFocus(snapshot) {
    const setup = this.level.setup || {};
    if (setup.kind && snapshot.kind !== setup.kind) return datasetButton(setup.kind);
    if (setup.modelKey && snapshot.modelKey !== setup.modelKey) return modelOption(setup.modelKey);
    if (!snapshot.finished && !snapshot.training) return { target: "train", label: "点「开始训练」" };
    return null;
  }

  renderAction(action) {
    if (!this.refs.action) return;
    this.refs.action.hidden = !(action && action.target === "train");
  }

  resolveTarget(action) {
    if (action.target === "dataset") return this.root.ownerDocument.querySelector(`#datasetTabs [data-dataset="${action.value}"]`);
    if (action.target === "model") return this.root.ownerDocument.querySelector(`#modelOptions input[value="${action.value}"]`);
    if (action.target === "train") return document.getElementById("trainButton");
    if (action.target === "lr") return document.getElementById("lrRange")?.closest(".range-field") || null;
    if (action.target === "samples") return document.getElementById("sampleRange")?.closest(".range-field") || null;
    if (action.target === "labelNoise") return document.getElementById("labelNoiseRange")?.closest(".range-field") || null;
    if (action.target === "epochs") return document.getElementById("epochRange")?.closest(".range-field") || null;
    if (action.target === "canvas") return document.getElementById("canvasWrap");
    return null;
  }
}
