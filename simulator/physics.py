import numpy as np

class PhysicsEngine:
    def __init__(self, mass, friction_coeff, restitution, size):
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

        # Energy tracking components
        self.translational_energy = 0.0
        self.rotational_energy = 0.0
        self.potential_energy = 0.0
        self.elastic_energy = 0.0
        self.dissipated_energy = 0.0
        
        # Moment of inertia (assuming circular disk)
        self.moment_of_inertia = 0.5 * mass * (size ** 2)

        # Contact parameters for collision
        self.contact_stiffness = 1e4  # N/m

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
            # Calculate pre-collision velocity
            v_before = self.momentum / self.mass
            v_before_normal = np.dot(v_before, collision_vector/distance)
            
            # Existing collision handling
            collision_normal = collision_vector / distance
            p_normal = np.dot(self.momentum, collision_normal) * collision_normal
            p_tangent = self.momentum - p_normal
            p_normal_post = -self.restitution * p_normal
            self.momentum[:] = p_normal_post + p_tangent
            
            # Calculate post-collision velocity and energy loss
            v_after = self.momentum / self.mass
            v_after_normal = np.dot(v_after, collision_vector/distance)
            
            # Energy loss calculation: E_loss = (1/2)m(v_before² - v_after²)
            self.dissipated_energy = 0.5 * self.mass * (v_before_normal**2 - v_after_normal**2)
            
            overlap = radius - distance
            return collision_normal * overlap
            
        return np.array([0.0, 0.0])

    def calculate_energies(self):
        # Translational Kinetic Energy: T_T = (1/2)mv²
        velocity = self.momentum / self.mass
        self.translational_energy = 0.5 * self.mass * np.dot(velocity, velocity)
        
        # Rotational Kinetic Energy: T_R = (1/2)Iω²
        self.rotational_energy = 0.5 * self.moment_of_inertia * (self.angular_momentum ** 2)
        
        # Total Kinetic Energy
        total_kinetic = self.translational_energy + self.rotational_energy
        
        return {
            'translational': self.translational_energy,
            'rotational': self.rotational_energy,
            'total_kinetic': total_kinetic,
            'elastic': self.elastic_energy,
            'dissipated': self.dissipated_energy
        }
