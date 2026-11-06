import { Vector2, Input, Output } from 'utils'

var totalFuelUsed = 0;
var gameScreenRadius = 500;

class Player{
    constructor(position, rotation){
        this.position = position;
        this.velocity = new Vector2();
        this.forces = new Vector2();
        this.rotation = rotation;
    }

    get direction() {return new Vector2(Math.cos(rotation), Math.sin(rotation))}



    getGravity() {
        return this.position.multiply(-1) * 0.1;
    }

    physicsStep(input, delta) {
        this.forces = this.getGravity();
        if (input.acceleration){
            this.forces = this.forces.add(this.direction * 2);
        }
        if (input.left_turn){
            this.rotation -= Math.PI/100;
        }
        if (input.right_turn){
            this.rotation += Math.PI/100;
        }
    }
}

class Target{
    constructor(){
        distanceFromCenter = (gameScreenRadius * Math.sqrt(Math.random()));
        theta = Math.random() * 2 * Math.pi;

        this.position = new Vector2(cos(theta), sin(theta)).multiply(distanceFromCenter);
    }
}

var target = new Target();
var player = new Player(new Vector2(-100,0), 0);

function resetWorld(){
    totalFuelUsed = 0;
    target = new Target();
    player = new Player(new Vector2(-100,0), 0);
}

function physicsStep(input, delta = 0.1){
    player.physicsStep(input, delta);
}

function getGameData(){
    return new Output(player.position, player.velocity, player.rotation, target.position, player.getGravity());
}