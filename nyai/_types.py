__all__ = [
    "NotGivenError",
    "NotGiven"
]

class NotGivenError(Exception):
    pass

class NotGiven:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NotGiven, cls).__new__(cls)
        return cls._instance

    def __getattribute__(self, name):
        return None

    def __bool__(self):
        return False

    def __eq__(self, other):
        return other is None 

    def __call__(self, *args, **kwargs):
        raise NotGivenError("This operation has not been setup")  