export class Vector2 {
    constructor(x = 0, y = 0){
        this.x = x;
        this.y = y;

        Object.freeze(this);
    }

    add(vector){
        return new Vector2(this.x + vector.x, this.y + vector.y);
    }

    subtract(vector){
        return new Vector2(this.x - vector.x, this.y - vector.y);
    }

    multiply(scalar){
        return new Vector2(this.x * scalar, this.y * scalar);
    }

    divide(scalar){
        return new Vector2(this.x / scalar, this.y / scalar);
    }

    dist_to(vector){
        return Math.sqrt((this.x - vector.x,2)**2 + (this.x - vector.x,2)**2);
    }

    magnitude(){
        return Math.sqrt(this.x * this.x + this.y * this.y);
    }

    normalize(){
        return new Vector2(this.x / this.magnitude(), this.y / this.magnitude());
    }
}

export class Input{
    constructor(acceleration = false, left_turn = false, right_turn = false){
        this.acceleration = acceleration;
        this.left_turn = left_turn;
        this.right_turn = right_turn;
        Object.freeze(this);
    }
}

export class Output{
    constructor(playerPos, playerVelocity, playerRotation, targetPosition, gravity){
        this.playerPos = playerPos;
        this.playerVelocity = playerVelocity;
        this.playerRotation = playerRotation;
        this.targetPosition = targetPosition;
        this.gravity = gravity;
        Object.freeze(this);
    }
}