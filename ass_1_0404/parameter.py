class Parameter(object):
    def __init__(self, data):
        self.data = data
        self.grad = None
    
    def __str__(self) -> str:
        return f"data: {self.data}, grad: {self.grad}"