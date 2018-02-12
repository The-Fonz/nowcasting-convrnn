import numpy as np
import skimage.draw

from utils import moving_mnist


class Ball:
    def __init__(self, shape=(100,100), radius=(5,25), velocity=5,
                 gravity=0, bounce=True):
        """
        Synthetic dataset that generates a bouncing ball
        :param shape: Tuple of rows,cols of canvas in px
        :param velocity: Max velocity in px/frame
        :param radius: Radius of ball in U[radius] in px
        :param gravity: Acceleration of gravity in px/frame**2
        """
        self.shape = np.array(shape)
        self.radius = radius
        self.velocity = velocity
        self.gravity = gravity
        self.bounce = bounce
    
    def __str__(self):
        return "<Ball shape={s.shape} radius={s.radius} velocity={s.velocity}\
        gravity={s.gravity} bounce={s.bounce}>".format(s=self)

    def __call__(self, batch_size=1, sequence_length=20):
        """
        Return ndarray batch of images with shape (batch_size, sequence_length, channels, w, h) where channels=1
        """
        canvas = np.zeros((batch_size,sequence_length,1, *self.shape), dtype=np.float32)
        
        for i_b in range(batch_size):
            r = np.random.uniform(*self.radius)
            # Generate velocity vector
            v = np.random.uniform(-self.velocity, self.velocity, 2)

            # Choose position within constraints
            pos = np.zeros(2)
            for i in range(len(pos)):
                pos[i] = np.random.uniform(0, self.shape[i]-r*2) + r

            for i_s in range(sequence_length):
                # TODO: Integrate gravity, bounce
                # Generates coordinates of circle
                # Pass shape to avoid coloring outside the lines
                rr, cc = skimage.draw.circle(pos[0], pos[1], r, shape=self.shape)
                # Update position with velocity
                pos += v
                # Bounce ball from walls
                if self.bounce:
                    # For x,y coords
                    for j in range(len(pos)):
                        if pos[j]+r > self.shape[j]:
                            # Break distance vector
                            pos[j] = self.shape[j]*2 - r*2 - pos[j]
                            # Reverse direction
                            v[j] = -v[j]
                        if pos[j] < r:
                            pos[j] = r*2 - pos[j]
                            v[j] = -v[j]
                # Color canvas
                canvas[i_b,i_s,0,rr,cc] = 1
        
        return canvas


class MovingMNIST:
    def __init__(self, shape=(64,64), num_size=28, nums_per_image=2):
        self.shape = shape
        self.num_size = num_size
        self.nums_per_image = nums_per_image

    def __str__(self):
        return "<MovingMNIST shape={s.shape} num_size={s.num_size} nums_per_image={s.nums_per_image}>".format(s=self)

    def __call__(self, batch_size=1, sequence_length=20, dtype=np.float32):
        # Returns dataset as shape (b, t, c, h, w)
        dataset = moving_mnist.generate_moving_mnist(shape=self.shape,
                                                     seq_len=sequence_length,
                                                     seqs=batch_size,
                                                     num_sz=self.num_size,
                                                     nums_per_image=self.nums_per_image,
                                                     dtype=dtype)
        return dataset
