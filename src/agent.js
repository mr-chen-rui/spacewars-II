import * as tfjs from "@tensorflow/tfjs"
import * as model from "./model"
import {Input, Output, Vector2} from "./utils"

const GAMMA = 0.99;
const BATCH_SIZE = 32;
const MEMORY_SIZE = 10000;
const LEARNING_RATE = 0.0005;

class DQNAgent{ // why does DQN stand for Deep Q learNing
    constructor(model) {
        this.model = model;
        this.memory = [];
    }

    remember(state, action, reward, nextState, done){ // Make the model add a snapshot to memory
        if (this.memory.length >= MEMORY_SIZE) {
            this.memory.shift(); // Remove oldest memory if full
        }
        this.memory.push({ state, action, reward, nextState, done });
    }

    async trainStep() { // this part is by Gemini ------------------------------------
        // Only train if we have gathered enough memory samples
        if (this.memory.length < BATCH_SIZE) return;

        // Pick a random sample batch from our experience memory bank
        const batch = [];
        for (let i = 0; i < BATCH_SIZE; i++) {
            const randomIndex = Math.floor(Math.random() * this.memory.length);
            batch.push(this.memory[randomIndex]);
        }

        // Extract arrays to batch tensors
        const states = batch.map(b => b.state);
        const nextStates = batch.map(b => b.nextState);

        // Minimize memory leaks using tf.tidy()
        const loss = tf.tidy(() => { 
            const stateTensor = tf.tensor2d(states, [BATCH_SIZE, 9]);
            const nextStateTensor = tf.tensor2d(nextStates, [BATCH_SIZE, 9]);

            // Predict current Q-values and next Q-values
            const currentQs = this.model.predict(stateTensor);
            const nextQs = this.model.predict(nextStateTensor);

            // Extract raw JS arrays out of tensors to manipulate target values
            const currentQsArray = currentQs.arraySync();
            const nextQsArray = nextQs.arraySync();

            // Bellman Equation implementation: Update targeted values 
            for (let i = 0; i < BATCH_SIZE; i++) {
                const targetAction = batch[i].action; // Integer (0 to 5)
                
                if (batch[i].done) {
                    // If the game ended on this step, target is just the final reward
                    currentQsArray[i][targetAction] = batch[i].reward;
                } else {
                    // Otherwise, target is Reward + (Gamma * Max predicted future reward)
                    const maxFutureQ = Math.max(...nextQsArray[i]);
                    currentQsArray[i][targetAction] = batch[i].reward + (GAMMA * maxFutureQ);
                }
            }

            // Convert updated values back to a tensor target
            const targetTensor = tf.tensor2d(currentQsArray, [BATCH_SIZE, 6]);

        // Compute loss to update network weights via a single gradient step
            return {
                stateTensor,
                targetTensor
            };
        });

        // Fit the model weights with our calculated targets
        const info = await this.model.fit(loss.stateTensor, loss.targetTensor, {
            epochs: 1,
            verbose: 0 // Keep console clean during quick frames
        });

        // Cleanup batch tensors from GPU/CPU memory
        loss.stateTensor.dispose();
        loss.targetTensor.dispose();
        
        return info.history.loss[0];
  }
}

// I decided to implement epsilon-greedy exploration 

var epsilon = 1.0;
const min_epsilon = 0.05;
const decay = 0.995;

function chooseAction(agent, input) {
    if (Math.random < epsilon){ // explore
        return Math.floor(Math.random() * 6)
    }
    else{
        return tf.tidy(() => { // be greedy
            const inputTensor = tf.tensor2d([stateArray], [1, 9]);
            const predictions = agent.model.predict(input.tensor);
            return predictions.argMax(1).dataSync()[0];
        });
    }
}