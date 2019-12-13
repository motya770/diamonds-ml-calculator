
class ItemsProvider(object):
    def __init__(self):
        print("init")

    def get(self) -> str:
        print(self)
        return "how did you do that"
