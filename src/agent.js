import * as tf from '@tensorflow/tfjs'

/**
 * Replay Buffer to store and sample agent experiences (transitions).
 */
class ReplayBuffer {
  constructor(capacity = 100) {
    this.capacity = capacity;
    this.buffer = [];
  }

  // Push a single experience into memory
  push(experience) {
    if (this.buffer.length >= this.capacity) {
      this.buffer.shift(); // Remove oldest memory if full
    }
    this.buffer.push(experience);
  }

  /**
   * Push a batch of experiences into memory
   * @param {Array} experiences - Array of experience objects: { state, action, reward, nextState, done }
   */
  pushBatch(experiences) {
    for (const exp of experiences) {
      this.push(exp);
    }
  }

  // Sample a random batch of experiences for training
  sample(batchSize) {
    const batch = [];
    const currentSize = this.buffer.length;
    const size = Math.min(batchSize, currentSize);
    
    // Simple random sampling without replacement
    const indices = [];
    while (indices.length < size) {
      const randomIndex = Math.floor(Math.random() * currentSize);
      if (!indices.includes(randomIndex)) {
        indices.push(randomIndex);
        batch.push(this.buffer[randomIndex]);
      }
    }
    return batch;
  }

  size() {
    return this.buffer.length;
  }
}

export class DQN {
    constructor(options = {}) {
        this.stateSize = options.stateSize || 9;     // 9 float inputs
        this.actionSize = options.actionSize || 6;   // 6 discrete actions (1 to 6)
        
        // Hyperparameters
        this.gamma = options.gamma || 0.95;          // Discount factor
        this.epsilon = options.epsilon || 1.0;       // Initial exploration rate
        this.epsilonMin = options.epsilonMin || 0.01; // Minimum exploration rate
        this.epsilonDecay = options.epsilonDecay || 0.9999; // Decay rate per episode
        this.learningRate = options.learningRate || 0.001;

        // Initialize Experience Replay Memory
        this.memory = new ReplayBuffer(options.memoryCapacity || 10000);

        // Build neural networks
        this.model = this.createModel();
        this.targetModel = this.createModel();
        this.updateTargetModel(); // Sync weights

        this.optimizer = tf.train.adam(this.learningRate);
    }

    createModel() {
        const model = tf.sequential();
        
        model.add(tf.layers.dense({
            units: 24,
            activation: 'relu',
            inputShape: [this.stateSize]
        }));
        
        model.add(tf.layers.dense({
            units: 24,
            activation: 'relu'
        }));
        
        model.add(tf.layers.dense({
            units: this.actionSize,
            activation: 'linear'
        }));

        return model;
    }

    updateTargetModel() {
        const weights = this.model.getWeights();
        const targetWeights = weights.map(w => w.clone());
        this.targetModel.setWeights(targetWeights);
    }

    act(state) {
        this.decayEpsilon()
        if (Math.random() < this.epsilon) {
        return Math.floor(Math.random() * this.actionSize) + 1; // 1 to 6
        }
        

        return tf.tidy(() => {
        const stateTensor = tf.tensor2d([state]);
        const predictions = this.model.predict(stateTensor);
        const actionIndex = predictions.argMax(1).dataSync()[0];
        return actionIndex + 1; // 1 to 6
        });
    }

    decayEpsilon() {
        if (this.epsilon > this.epsilonMin) {
        this.epsilon *= this.epsilonDecay;
        }
    }

    /**
     * Helper method to push a batch of experiences directly into the DQN's memory
     * @param {Array} experiences - Array of transitions
     */
    pushMemory(experiences) {
        this.memory.push(experiences);
    }

    /**
     * Train the model by sampling a batch from its memory buffer
     * @param {number} batchSize - Number of samples to train on
     */
    async replay(batchSize = 32) {
        if (this.memory.size() < batchSize) return; // Not enough memories yet

        const batch = this.memory.sample(batchSize);

        const states = batch.flatMap(exp => exp[0]);
        const nextStates = batch.flatMap(exp => exp[3]);


        const stateTensor = tf.tensor2d(states, [batch.length, this.stateSize]);
        const nextStateTensor = tf.tensor2d(nextStates, [batch.length, this.stateSize]);

        const currentQ = this.model.predict(stateTensor);
        const nextQ = this.targetModel.predict(nextStateTensor);

        const currentQData = currentQ.arraySync();
        const nextQData = nextQ.arraySync();

        for (let i = 0; i < batch.length; i++) {
        const { action, reward, done } = batch[i];
        const actionIndex = action - 1; // Map 1-6 to 0-5

        let target = reward;
        if (!done) {
            const maxNextQ = Math.max(...nextQData[i]);
            target = reward + this.gamma * maxNextQ;
        }

        currentQData[i][actionIndex] = target;
        }

        const targetTensor = tf.tensor2d(currentQData);

        await this.optimizer.minimize(() => {
        const prediction = this.model.predict(stateTensor);
        return tf.losses.meanSquaredError(targetTensor, prediction);
        });

        // Clean up tensors
        stateTensor.dispose();
        nextStateTensor.dispose();
        currentQ.dispose();
        nextQ.dispose();
        targetTensor.dispose();
    }
}
