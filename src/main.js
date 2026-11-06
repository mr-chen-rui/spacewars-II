import * as Game from "./game.js"

var canvas = document.getElementById("screen")
var html = document.getElementsByTagName("html")[0]

var canvasWidth = canvas.clientWidth
var canvasHeight = canvas.clientWidth/2

function resize_screen(){ // Haven't connected this one yet
    canvas.setAttribute("width", html.clientWidth)
    canvas.setAttribute("height", html.clientWidth/2)
    canvasWidth = html.clientWidth
    canvasHeight = html.clientWidth/2
}

function debug_draw(){
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0,canvasWidth ,canvasHeight)
    ctx.fillRect(0,0,100,100)
}

resize_screen()
debug_draw()