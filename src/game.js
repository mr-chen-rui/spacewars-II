import { Vector2, Input, Output } from './utils'


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
        this.forces = this.getGravity();
        if (input.acceleration){
            this.forces = this.forces.add(this.direction.multiply(delta).multiply(0.001));
        }
        if (input.left_turn){
            this.rotation -= Math.PI/100;
        }
        if (input.right_turn){
            this.rotation += Math.PI/100;
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
        var theta = Math.random() * 2 * Math.PI;

        this.position = new Vector2(Math.cos(theta), Math.sin(theta)).multiply(distanceFromCenter);
        console.log(this.position)
    }
}

export var target = new Target();
export var player = new Player(new Vector2(-100,0), 0);

export function isGameFinished(){
    return target.position.dist_to(player.position) < 0.5
}

export var fuelUsed = 0;
export function addFuelUsed(value){
    fuelUsed += value
}

export var timeElapsed = 0;
export function addTimeElapsed(value){
    timeElapsed += value
}

export function resetWorld(){
    fuelUsed = 0;
    timeElapsed = 0;
    target = new Target();
    player = new Player(new Vector2(-100,0), 0);
}

export function physicsStep(input, delta = 0.1){
    player.physicsStep(input, delta);
}

export function getGameData(){
    return new Output(player.position, player.velocity, player.rotation, target.position, player.getGravity());
}

export function getScore(){
    return 
}

export function getFitness(fuelUsedSinceLastTick, delta){
    return target.position.subtract(player.position).magnitude - (fuelUsedSinceLastTick + delta) 
}