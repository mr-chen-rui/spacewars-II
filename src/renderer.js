import {Vector2, Input, Output} from "./utils"

const playerAsPoints = [new Vector2(0,1), new Vector2(-1,-1), new Vector2(0,-0.5), new Vector2(1,-1)];

var scale = 1000;

export function drawPlayer(ctx, pos, rot){
    var playerRotated = new Array(4)
    playerAsPoints.forEach((element) => {
        playerRotated[playerAsPoints.indexOf(element)] = element.rotateAroundOrigin(rot).multiply(scale).add(pos).gameSpaceToScreenSpace();
    })

    ctx.beginPath();
    playerRotated.forEach((element) => { // arrow functions so useful!!
        if (playerRotated.indexOf(element) == 0){
            ctx.moveTo(element.x, element.y);
        }
        else {
            ctx.lineTo(element.x, element.y)
        }
    })
    ctx.lineTo(playerRotated[0].x, playerRotated[0].y)
    ctx.stroke();
}

export function drawOrigin(ctx){

}

export function drawTarget(ctx, pos){

}

export function drawBackground(ctx){
    ctx.beginPath(); // Important to clear current path.
    ctx.fillStyle = "black";
    ctx.arc(100, 50, 40, 0, 360);
    ctx.fill();

    ctx.strokeStyle = `#8ed16d`;
    ctx.lineWidth = 0.2;
    for (var i = 10; i < 50; i += 10){
        ctx.beginPath();
        ctx.arc(100, 50, i, 0, Math.PI * 2);
        ctx.stroke();
    }

    ctx.beginPath();
    ctx.moveTo(100, 10);
    ctx.lineTo(100,90);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(60, 50);
    ctx.lineTo(140,50);
    ctx.stroke();
}