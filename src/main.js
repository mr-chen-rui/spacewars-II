import * as Game from "./game.js"
import * as Screen from "./renderer.js"
import * as Agent from "./agent.js"
import { Vector2, Input, Output, round } from "./utils.js"

/* 
    Hello! If you're reading this, you're probably the reviewer. This project went wayyy overtime and took me
    way too long to figure out how to use tfjs. This is an extremely simple game because I couldn't figure out how to 
    calculate whether a projectile hits a moving target in nonuniform gravity :(
*/

const canvas = document.getElementById("screen")
const main = document.getElementsByTagName("main")[0]

var canvasWidth = canvas.clientWidth
var canvasHeight = canvas.clientWidth/2

const ctx = canvas.getContext("2d");

var input = new Input(false, false, false);

var current_state = 0 // 0 means player is playing, 1 means training, 2 means AI playing

const scoredisplay = document.getElementById("score")
const fueldisplay = document.getElementById("fuel")
const timedisplay = document.getElementById("time")
const trainingdisplay = document.getElementById("training")

const playButton = document.getElementById("play")
const trainButton = document.getElementById("train")
const watchButton = document.getElementById("watch")

function resize_screen(event){ // Haven't connected this one yet
    canvas.setAttribute("width", main.clientWidth - 20)
    canvas.setAttribute("height", main.clientWidth/2)
    canvasWidth = main.clientWidth - 20
    canvasHeight = main.clientWidth/2
    ctx.clearRect(0,0,canvasWidth,canvasHeight)
    ctx.setTransform(1, 0, 0, 1, 0, 0) // reset scale because ctx.scale is cumulative
    ctx.scale(canvasWidth / 200, canvasHeight / 100)
}

onresize = resize_screen
resize_screen()


var keys = new Array();

window.addEventListener("keydown",
    function(e){
        keys[e.keyCode] = e.keyCode; // I know its deprecated but idk how else to use it
    },
false);

window.addEventListener('keyup',
    function(e){
        keys[e.keyCode] = false;
    },
false);

function getNumberArray(arr){
    var newArr = new Array();
    for(var i = 0; i < arr.length; i++){
        if(typeof arr[i] == "number"){
            newArr[newArr.length] = arr[i];
        }
    }
    return newArr;
}

function calculateInput(){
    var code = getNumberArray(keys);

    var accelerating = false;
    var left_turn = false;
    var right_turn = false;

    for (var event_code of code){
        if (event_code == 87 || event_code == 38){
            accelerating = true;
            
        }
        if (event_code == 65 || event_code == 37){
            left_turn = true;
            
        }
        if (event_code == 68 || event_code == 39){
            right_turn = true;
            
        }
    }

    return new Input(accelerating, left_turn, right_turn);
}

function draw(ctx){
    ctx.beginPath();

    Screen.drawBackground(ctx);
    Screen.drawTarget(ctx, Game.target.position);
    Screen.drawPlayer(ctx, Game.player.position, Game.player.rotation + 2 * Math.PI);

    timedisplay.textContent = `Time Elapsed: ${round(Game.timeElapsed)}`
    fueldisplay.textContent = `Fuel Used: ${round(Game.fuelUsed)}`
    scoredisplay.textContent = `Score: ${Game.getScore()}`
}

const agent = new Agent.DQN()

function physicsStep(){
    requestAnimationFrame(physicsStep)
    if (Game.isGameFinished() == false){

        if (current_state == 0){
            Game.addTimeElapsed(0.001);
        
            Game.player.physicsStep(calculateInput(), 1);
            draw(ctx);
        }

        if (current_state == 1){
            // Moved training code to its own async function

            
        }

        if (current_state == 2){
            Game.addTimeElapsed(0.001);

            const output = Game.getGameData();
            var input = agent.act(output.array);
            console.log(input)

            var accelerating = false;
            var right_turn = false;
            var left_turn = false;

            if (input == 1 || input == 4){
                right_turn = true;
            }

            else if (input == 2 || input == 5){
                left_turn = true;
            }

            if (input == 6 || input == 4 || input == 5){
                accelerating = true
            }

            input = new Input(accelerating, left_turn, right_turn)

            console.log(input)

            Game.physicsStep(input, 1);
            draw(ctx)
        }
        
    }

}

async function train(){
    Game.addTimeElapsed(0.001);

    const output = Game.getGameData();
    const input = agent.act(output.array);

    var accelerating = false;
    var right_turn = false;
    var left_turn = false;

    if (input == 1 || input == 4){
        right_turn = true;
    }

    else if (input == 2 || input == 5){
        left_turn = true;
    }

    if (input == 6 || input == 4 || input == 5){
        accelerating = true
    }

    Game.physicsStep(new Input(accelerating, left_turn, right_turn), 1);

    agent.pushMemory([output.array, input.array, Game.getFitness(1), Game.getGameData().array, Game.isGameFinished()]);
}

async function runTraining(){
    Game.resetWorld()
    
    current_state = 1

    var iterations = 0
    while (iterations < 1000){
        train()
        iterations += 1
    }

    for (var i = 0; i < 50; i++){
        agent.replay();
    }
}

function startGame(e){
    if (current_state != 1){
        Game.resetWorld();
        current_state = 0;

    }

}

async function trainAI(e){
    trainingdisplay.textContent = ""

    console.log("Training...")
    requestAnimationFrame(runTraining)
    await runTraining()
    trainingdisplay.textContent = "Finished Training"
}


function watchAI(e){
    Game.resetWorld();
    current_state = 2;
}

playButton.addEventListener("click", startGame);
trainButton.addEventListener("click", trainAI);
watchButton.addEventListener("click", watchAI);

physicsStep()