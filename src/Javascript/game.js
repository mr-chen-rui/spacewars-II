import { Vector2, Input, Output } from 'utils.js'

class Player{
    constructor(position, rotation){
        this.position = position
        this.velocity = new Vector2()
        this.forces = new Vector2()
        this.rotation = rotation
    }

    get direction() {return new Vector2(Math.cos(rotation), Math.sin(rotation))}



    getGravity() {
        return this.position.multiply(-1) * 0.1
    }

    physicsStep(input) {
        this.forces = this.getGravity()
        if (input.acceleration){
            this.forces = this.forces.add(this.direction * 2)
        }
        if (input.left_turn){
            this.rotation -= Math.PI/100
        }
        if (input.right_turn){
            this.rotation += Math.PI/100
        }
    }
}