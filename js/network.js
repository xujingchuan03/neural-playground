import { clamp, gaussian, mulberry32, shuffleInPlace } from "./random.js";

export const MODEL_PRESETS = {
  linear: {
    name: "线性分类器",
    hidden: [],
    short: "线性",
    description: "没有隐藏层，只能形成线性决策边界。",
    tip: "线性模型的边界是一条直线。面对双月牙、同心圆和螺旋时，它通常只能取得一半左右的准确率。",
  },
  small: {
    name: "小型 MLP",
    hidden: [8],
    short: "小 MLP",
    description: "一个 8 神经元隐藏层，能学习弯曲边界。",
    tip: "小型 MLP 足以处理大多数二维分类数据，是准确率和训练速度之间的平衡选择。",
  },
  deep: {
    name: "深层 MLP",
    hidden: [16, 8],
    short: "深 MLP",
    description: "两个隐藏层，适合更复杂的螺旋数据。",
    tip: "深层网络能表达更复杂的边界，但也更容易在小数据上过拟合，建议同时观察测试集曲线。",
  },
};

function zeros(size) {
  return new Float64Array(size);
}

function softmax(values) {
  let maximum = -Infinity;
  for (let index = 0; index < values.length; index += 1) {
    if (values[index] > maximum) maximum = values[index];
  }

  const output = new Float64Array(values.length);
  let total = 0;
  for (let index = 0; index < values.length; index += 1) {
    const value = Math.exp(values[index] - maximum);
    output[index] = value;
    total += value;
  }

  for (let index = 0; index < output.length; index += 1) {
    output[index] /= total;
  }
  return output;
}

export class MLP {
  constructor(hiddenLayers, options = {}) {
    this.hiddenLayers = [...hiddenLayers];
    this.layerSizes = [2, ...hiddenLayers, 2];
    this.seed = (options.seed === undefined ? 1 : options.seed) >>> 0;
    this.random = mulberry32(this.seed);
    this.weights = [];
    this.biases = [];
    this.weightMomentum = [];
    this.weightVelocity = [];
    this.biasMomentum = [];
    this.biasVelocity = [];
    this.adamStep = 0;
    this.initialize();
  }

  initialize() {
    for (let layer = 0; layer < this.layerSizes.length - 1; layer += 1) {
      const inputSize = this.layerSizes[layer];
      const outputSize = this.layerSizes[layer + 1];
      const scale = Math.sqrt(2 / (inputSize + outputSize));
      const weights = new Float64Array(inputSize * outputSize);

      for (let index = 0; index < weights.length; index += 1) {
        weights[index] = gaussian(this.random) * scale;
      }

      this.weights.push(weights);
      this.biases.push(zeros(outputSize));
      this.weightMomentum.push(zeros(weights.length));
      this.weightVelocity.push(zeros(weights.length));
      this.biasMomentum.push(zeros(outputSize));
      this.biasVelocity.push(zeros(outputSize));
    }
  }

  forward(input) {
    const activations = [new Float64Array([input[0], input[1]])];
    const preActivations = [];
    let current = activations[0];

    for (let layer = 0; layer < this.weights.length; layer += 1) {
      const inputSize = this.layerSizes[layer];
      const outputSize = this.layerSizes[layer + 1];
      const weights = this.weights[layer];
      const biases = this.biases[layer];
      const next = new Float64Array(outputSize);

      for (let output = 0; output < outputSize; output += 1) {
        let value = biases[output];
        for (let inputIndex = 0; inputIndex < inputSize; inputIndex += 1) {
          value += weights[output * inputSize + inputIndex] * current[inputIndex];
        }
        next[output] = value;
      }

      preActivations.push(next);

      if (layer === this.weights.length - 1) {
        current = softmax(next);
      } else {
        const activated = new Float64Array(outputSize);
        for (let output = 0; output < outputSize; output += 1) {
          activated[output] = Math.tanh(next[output]);
        }
        current = activated;
      }

      activations.push(current);
    }

    return { output: current, activations, preActivations };
  }

  predictProba(input) {
    return this.forward(input).output;
  }

  predict(input) {
    const probabilities = this.predictProba(input);
    return probabilities[1] >= probabilities[0] ? 1 : 0;
  }

  evaluate(samples) {
    if (samples.length === 0) {
      return { loss: 0, accuracy: 0, confusion: [[0, 0], [0, 0]] };
    }

    let totalLoss = 0;
    let correct = 0;
    const confusion = [[0, 0], [0, 0]];

    for (const sample of samples) {
      const probabilities = this.predictProba([sample.x, sample.y]);
      const probability = clamp(probabilities[sample.label], 1e-9, 1);
      const predicted = probabilities[1] >= probabilities[0] ? 1 : 0;
      totalLoss -= Math.log(probability);
      if (predicted === sample.label) correct += 1;
      confusion[sample.label][predicted] += 1;
    }

    return {
      loss: totalLoss / samples.length,
      accuracy: correct / samples.length,
      confusion,
    };
  }

  trainEpoch(samples, learningRate, batchSize = 32) {
    shuffleInPlace(samples, this.random);
    let steps = 0;

    for (let start = 0; start < samples.length; start += batchSize) {
      const batch = samples.slice(start, start + batchSize);
      this.step(batch, learningRate);
      steps += 1;
    }

    return steps;
  }

  step(batch, learningRate) {
    const weightGradients = this.weights.map((weights) => zeros(weights.length));
    const biasGradients = this.biases.map((biases) => zeros(biases.length));

    for (const sample of batch) {
      const forward = this.forward([sample.x, sample.y]);
      const probabilities = forward.output;
      let delta = new Float64Array(2);
      delta[0] = probabilities[0] - (sample.label === 0 ? 1 : 0);
      delta[1] = probabilities[1] - (sample.label === 1 ? 1 : 0);

      for (let layer = this.layerSizes.length - 2; layer >= 0; layer -= 1) {
        const inputSize = this.layerSizes[layer];
        const outputSize = this.layerSizes[layer + 1];
        const previousActivation = forward.activations[layer];
        const weightGradient = weightGradients[layer];
        const biasGradient = biasGradients[layer];

        for (let output = 0; output < outputSize; output += 1) {
          const error = delta[output];
          biasGradient[output] += error;
          for (let inputIndex = 0; inputIndex < inputSize; inputIndex += 1) {
            weightGradient[output * inputSize + inputIndex] += error * previousActivation[inputIndex];
          }
        }

        if (layer > 0) {
          const previousDelta = new Float64Array(inputSize);
          const activation = forward.activations[layer];
          const weights = this.weights[layer];

          for (let inputIndex = 0; inputIndex < inputSize; inputIndex += 1) {
            let sum = 0;
            for (let output = 0; output < outputSize; output += 1) {
              sum += weights[output * inputSize + inputIndex] * delta[output];
            }
            previousDelta[inputIndex] = sum * (1 - activation[inputIndex] * activation[inputIndex]);
          }
          delta = previousDelta;
        }
      }
    }

    this.applyAdam(weightGradients, biasGradients, batch.length, learningRate);
  }

  applyAdam(weightGradients, biasGradients, batchSize, learningRate) {
    this.adamStep += 1;
    const betaOne = 0.9;
    const betaTwo = 0.999;
    const epsilon = 1e-8;
    const correctionOne = 1 - Math.pow(betaOne, this.adamStep);
    const correctionTwo = 1 - Math.pow(betaTwo, this.adamStep);

    for (let layer = 0; layer < this.weights.length; layer += 1) {
      const weights = this.weights[layer];
      const biases = this.biases[layer];
      const weightGradient = weightGradients[layer];
      const biasGradient = biasGradients[layer];
      const weightMomentum = this.weightMomentum[layer];
      const weightVelocity = this.weightVelocity[layer];
      const biasMomentum = this.biasMomentum[layer];
      const biasVelocity = this.biasVelocity[layer];

      for (let index = 0; index < weights.length; index += 1) {
        const gradient = weightGradient[index] / batchSize;
        weightMomentum[index] = betaOne * weightMomentum[index] + (1 - betaOne) * gradient;
        weightVelocity[index] = betaTwo * weightVelocity[index] + (1 - betaTwo) * gradient * gradient;
        const momentumHat = weightMomentum[index] / correctionOne;
        const velocityHat = weightVelocity[index] / correctionTwo;
        weights[index] -= learningRate * momentumHat / (Math.sqrt(velocityHat) + epsilon);
      }

      for (let index = 0; index < biases.length; index += 1) {
        const gradient = biasGradient[index] / batchSize;
        biasMomentum[index] = betaOne * biasMomentum[index] + (1 - betaOne) * gradient;
        biasVelocity[index] = betaTwo * biasVelocity[index] + (1 - betaTwo) * gradient * gradient;
        const momentumHat = biasMomentum[index] / correctionOne;
        const velocityHat = biasVelocity[index] / correctionTwo;
        biases[index] -= learningRate * momentumHat / (Math.sqrt(velocityHat) + epsilon);
      }
    }
  }

  countParameters() {
    let total = 0;
    for (const weights of this.weights) total += weights.length;
    for (const biases of this.biases) total += biases.length;
    return total;
  }

  architecture() {
    return this.layerSizes.join(" → ");
  }
}
