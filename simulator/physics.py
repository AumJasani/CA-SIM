import numpy as np

class PhysicsEngine:
    def __init__(self, mass, friction_coeff, restitution):
        self.mass = mass
        self.friction_coeff = friction_coeff
        self.restitution = restitution
        
        # Linear motion state
        self.momentum = np.array([0.0, 0.0])
        
        # Angular motion state
        self.angular_momentum = 0.0
        self.orientation = 0.0  # Robot's facing direction in radians
        
        # Linear motion limits
        self.max_acceleration = 2 * 50  # 2 m/s² * METER_TO_PIXEL
        
        # Angular motion limits (matching Sample2.py)
        self.max_angular_acceleration = np.radians(90)  # 90 degrees/s²
        self.max_angular_velocity = np.radians(180)  # 180 degrees/s

    def update(self, target_velocity, target_angular_velocity, dt):
        # Linear motion update (component-wise)
        Ax = np.clip(
            (target_velocity[0] - self.momentum[0] / self.mass) / dt,
            -self.max_acceleration,
            self.max_acceleration
        )
        Ay = np.clip(
            (target_velocity[1] - self.momentum[1] / self.mass) / dt,
            -self.max_acceleration,
            self.max_acceleration
        )
        
        # Angular motion update
        current_angular_velocity = self.angular_momentum
        alpha = np.clip(
            (target_angular_velocity - current_angular_velocity) / dt,
            -self.max_angular_acceleration,
            self.max_angular_acceleration
        )
        
        # Update linear momentum
        self.momentum[0] += Ax * self.mass * dt
        self.momentum[1] += Ay * self.mass * dt
        
        # Update angular momentum and orientation
        self.angular_momentum += alpha * dt
        self.angular_momentum = np.clip(
            self.angular_momentum,
            -self.max_angular_velocity,
            self.max_angular_velocity
        )
        self.orientation += self.angular_momentum * dt
        
        # Apply friction to both linear and angular motion
        self.momentum *= (1 - self.friction_coeff)
        self.angular_momentum *= (1 - self.friction_coeff)
        
        # Return current velocities and orientation
        return self.momentum / self.mass, self.angular_momentum, self.orientation

    def handle_collision(self, position, radius, obstacle):
        closest_x = max(obstacle.left, min(position[0], obstacle.right))
        closest_y = max(obstacle.top, min(position[1], obstacle.bottom))
        
        collision_vector = position - np.array([closest_x, closest_y])
        distance = np.linalg.norm(collision_vector)
        
        if distance < radius:
            collision_normal = collision_vector / distance
            p_normal = np.dot(self.momentum, collision_normal) * collision_normal
            p_tangent = self.momentum - p_normal
            p_normal_post = -self.restitution * p_normal
            self.momentum[:] = p_normal_post + p_tangent
            overlap = radius - distance
            return collision_normal * overlap
        return np.array([0.0, 0.0])
