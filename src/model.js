import * as tfjs from "@tensorflow/tfjs"
import {Input, Output, Vector2} from './utils'
// trying to run npm install @tensorflow/tfjs but I dont have sudo access

// no clue how to do PPO because I have smol brain


// ok so there are 9 inputs: player x, player y, player velocity x, player velocity y, player rotation, gravity, target position x, and target position y
// and there are 6 possible outputs: 
// nothing, turn left, turn right, accelerate, accelerate while turning left, accelerate while turning right

const INPUT_SIZE = 9;
const OUTPUT_SIZE = 6;

// this makes a sequential model which means it looks like this (from what I understand):
// Input layer -> Layer 1 -> ... -> Output layer
// so i guess the layers are just tensors with weights and somehow multiplying the inputs by the layers and squishing them gives you outputs??

const model = tfjs.sequential();

// dense means every neuron is connected to every one in the previous layer
// so the numbers in the tensor are the neurons i think
model.add(tfjs.layers.dense({
    inputShape: [INPUT_SIZE],
    units: 64, // how many neurons i guess
    activation: 'relu' // ok did some research relu is just a curve to minimize 'linearity' and there are also sigmoid and gelu
}))

model.add(tfjs.layers.dense({
    units: 64,
    activation: 'relu'
}))

model.add(tfjs.layers.dense({
    units: OUTPUT_SIZE,
    activation: 'linear' // linear this time because we have discrete outputs
}))

model.compile({
    optimizer: tfjs.train.adam(0.0005),
    loss: 'meanSquaredError'
})

function getAction(input){ // actually the input will be the output class but I'm too lazy to change that
    return tfjs.tidy(()=>{ // gets rid of tensors in memory
        const inputTensor = input.tensor;
        const logits = model.predict(inputTensor);

        const actionIndex = logits.argMax(1).dataSync()[0];

        return new Input(actionIndex >= 3, actionIndex === 1 || actionIndex === 4, actionIndex === 2 || actionIndex === 5)
    });
}