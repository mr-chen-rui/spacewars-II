import * as Game from "./game.js"
import * as Screen from "./renderer.js"
import { Vector2, Input, Output } from "./utils.js"

var canvas = document.getElementById("screen")
var main = document.getElementsByTagName("main")[0]

var canvasWidth = canvas.clientWidth
var canvasHeight = canvas.clientWidth/2

const ctx = canvas.getContext("2d");

var input = new Input(false, false, false);

var scoredisplay = document.getElementById("score")
var fueldisplay = document.getElementById("fuel")
var timedisplay = document.getElementById("time")



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
    var code = getNumberArray(keys)

    var accelerating = false
    var left_turn = false
    var right_turn = false

    for (var event_code of code){
        if (event_code == 87 || event_code == 38){
            accelerating = true
            Game.fuelUsed += 1;
        }
        if (event_code == 65 || event_code == 37){
            left_turn = true
            Game.fuelUsed += 0.1;
        }
        if (event_code == 68 || event_code == 39){
            right_turn = true
            Game.fuelUsed += 0.1;
        }
    }

    input = new Input(accelerating, left_turn, right_turn);
}

function draw(ctx){
    ctx.beginPath()

    Screen.drawBackground(ctx)
    Screen.drawTarget(ctx, Game.target.position)
    Screen.drawPlayer(ctx, Game.player.position, Game.player.rotation + 2 * Math.PI)

    console.log(Game.player.position)
}

function physicsStep(){
    Game.timeElapsed += 1;
    

    calculateInput()
    Game.player.physicsStep(input, 1)
    draw(ctx)

    
    requestAnimationFrame(physicsStep,1)

}

physicsStep()