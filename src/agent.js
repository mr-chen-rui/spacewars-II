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
    this.epsilonDecay = options.epsilonDecay || 0.995; // Decay rate per episode
    this.learningRate = options.learningRate || 0.001;

    // Initialize Experience Replay Memory
    this.memory = new ReplayBuffer(options.memoryCapacity || 10000);

    // Build neural networks
    this.model = this.createModel();
    this.targetModel = this.createModel();
    this.updateTargetModel(); // Sync weights

    this.optimizer = tf.train.adam(this.learningRate);

    this.replaySteps = 0;
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
  replay(batchSize = 32) {
    this.replaySteps += 1;
    if (this.replaySteps == 10){
      this.updateTargetModel();
      this.replaySteps = 0;
    }

    if (this.memory.size() < batchSize) return;

    // Wrap the entire replay step in tf.tidy to prevent memory leaks
    tf.tidy(() => {
      this.replaySteps += 1;
      if (this.replaySteps == 100){
        this.replaySteps = 0;
        this.updateTargetModel()
      }
      const batch = this.memory.sample(batchSize);

      // 1. Unpack batch data into separate arrays
      const states = batch.map(exp => exp[0]);
      const actions = batch.map(exp => exp[1] - 1); // Map actions 1-6 to indices 0-5
      const rewards = batch.map(exp => exp[2]);
      const nextStates = batch.map(exp => exp[3]);
      const dones = batch.map(exp => exp[4] ? 1 : 0);

      // 2. Convert batch arrays to Tensors
      const stateTensor = tf.tensor2d(states, [batchSize, this.stateSize]);
      const nextStateTensor = tf.tensor2d(nextStates, [batchSize, this.stateSize]);
      const actionTensor = tf.tensor1d(actions, 'int32');
      const rewardTensor = tf.tensor1d(rewards);
      const doneTensor = tf.tensor1d(dones);

      // 3. Compute target Q-values using the target model
      const nextQ = this.targetModel.apply(nextStateTensor, { training: false });
      const maxNextQ = nextQ.max(1); // Get max Q-value for next state [batchSize]

      // Target = reward + gamma * maxNextQ * (1 - done)
      const notDoneTensor = tf.scalar(1).sub(doneTensor);
      const targetQ = rewardTensor.add(maxNextQ.mul(this.gamma).mul(notDoneTensor));

      let lossValue = 0;

      // 4. Minimize loss using the optimizer
      this.optimizer.minimize(() => {
          // Forward pass through the main model
          const preds = this.model.apply(stateTensor, { training: true });

          // Use one-hot encoding to extract only the Q-values of the actions actually taken
          const actionMask = tf.oneHot(actionTensor, this.actionSize);
          const predictedQ = tf.sum(preds.mul(actionMask), 1); // Shape: [batchSize]

          // Compute Mean Squared Error strictly between target Q and chosen action Q
          const loss = tf.losses.meanSquaredError(targetQ, predictedQ);
          
          // Extract the raw number safely before tidy cleans it up
          lossValue = loss.dataSync()[0];

          return loss;
      });
    // console.log("Training Loss:", lossValue);
    });
  }
}
