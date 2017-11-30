# Class-based transforms for easy re-use
import skimage.transform

class Slice():
    def __init__(self, x1, x2, y1, y2):
        self.p = (x1,x2,y1,y2)

    def __call__(self, arr):
        return arr[self.p[0] : self.p[1] , self.p[2] : self.p[3]]

class Resize():
    def __init__(self, target_height, target_width):
        self.target_width = target_width
        self.target_height = target_height

    def __call__(self, arr):
        # Anti-aliasing must be on when downsizing
        # Set mode to 'reflect' so it reflects pixel values at borders
        # Preserve range, so don't transform to [0,1]
        return skimage.transform.resize(arr, (self.target_height, self.target_width), preserve_range=True, mode='reflect')

class Clip():
    def __init__(self, clip_below, clip_val=0):
        self.clip_below = clip_below
        self.clip_val = clip_val

    def __call__(self, arr):
        "Mutates original array!"
        arr[arr<self.clip_below] = self.clip_val
        return arr

class Normalize():
    def __init__(self):
        pass
    def __call__(self, arr):
        return arr / 255.
