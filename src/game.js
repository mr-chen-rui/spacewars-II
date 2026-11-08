import { Vector2, Input, Output } from './utils'

export var totalFuelUsed = 0;
export var gameScreenRadius = 500;

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

    get alive(){
        return this.position.magnitude() < gameScreenRadius;
    }

    getGravity() {
        return this.position.multiply(-1).multiply(0.0001);
    }

    physicsStep(input, delta) {
        this.forces = this.getGravity();
        if (input.acceleration){
            this.forces = this.forces.add(this.direction.multiply(delta));
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
        var theta = Math.random() * 2 * Math.pi;

        this.position = new Vector2(Math.cos(theta), Math.sin(theta)).multiply(distanceFromCenter);
    }
}

export var target = new Target();
export var player = new Player(new Vector2(-100,0), 0);

export function resetWorld(){
    totalFuelUsed = 0;
    target = new Target();
    player = new Player(new Vector2(-100,0), 0);
}

export function physicsStep(input, delta = 0.1){
    player.physicsStep(input, delta);
}

export function getGameData(){
    return new Output(player.position, player.velocity, player.rotation, target.position, player.getGravity());
}