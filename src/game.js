import { Vector2, Input, Output, round } from './utils.js'


export var gameScreenRadius = 100;

export class Player{
    constructor(position, rotation){
        this.position = position;
        this.velocity = new Vector2();
        this.forces = new Vector2();
        this.rotation = rotation;
    }

    get direction() {
        if (this.rotation >= 2 * Math.PI){
            this.rotation -= 2 * Math.PI
        }
        if (this.rotation <= 2 * Math.PI){
            this.rotation += 2 * Math.PI
        }

        return new Vector2(Math.cos(this.rotation + Math.PI/2), Math.sin(this.rotation + Math.PI/2));
    }

    getGravity() {
        return this.position.multiply(-1).multiply(0.00001);
    }

    physicsStep(input, delta) {
        previousDistanceFromTarget = target.position.subtract(this.position).magnitude();
        this.forces = this.getGravity();
        fuelUsedSinceLastTick = 0;
        if (input.acceleration){
            this.forces = this.forces.add(this.direction.multiply(delta).multiply(0.001));
            addFuelUsed(1);
        }
        if (input.left_turn){
            this.rotation -= Math.PI/100;
            addFuelUsed(0.1);
        }
        if (input.right_turn){
            this.rotation += Math.PI/100;
            addFuelUsed(0.1);
        }

        // F=MA but we assume M is 1 so F=A

        this.velocity = this.velocity.add(this.forces);
        this.velocity = this.velocity.multiply(0.999)
        this.position = this.position.add(this.velocity);
    }
}

export class Target{
    constructor(){
        var distanceFromCenter = (gameScreenRadius * Math.sqrt(Math.random()));
        var theta = (Math.random() * Math.PI) - Math.PI/2

        this.position = new Vector2(Math.cos(theta), Math.sin(theta)).multiply(distanceFromCenter);
        console.log(this.position)
    }
}

export var player = new Player(new Vector2(-100,0), 0);
export var target = new Target();

export function isGameFinished(){
    return target.position.dist_to(player.position) < 10
}

export var fuelUsed = 0;
var fuelUsedSinceLastTick = 0;

function addFuelUsed(value){
    fuelUsed += value
    fuelUsedSinceLastTick += value
}

export var timeElapsed = 0;
export function addTimeElapsed(value){
    timeElapsed += value
}

export function resetWorld(){
    fuelUsed = 0;
    timeElapsed = 0;
    
    player = new Player(new Vector2(-100,0), 0);
    target = new Target();
}

export function physicsStep(input, delta = 0.1){
    player.physicsStep(input, delta);
}

export function getGameData(){
    return new Output(player.position, player.velocity, player.rotation, target.position, player.getGravity());
}

export function getScore(){
    return round((50 / Math.max(fuelUsed, 0.1) + 50 / Math.max(timeElapsed, 0.1)))
}

var previousDistanceFromTarget = 999;

export function getFitness(delta){
    return player.position.subtract(target.position).magnitude() * 10 - (fuelUsedSinceLastTick * delta) *0.1
}