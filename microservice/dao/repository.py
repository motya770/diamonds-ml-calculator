from pymongo import MongoClient

class DiamondRepository(object):

    def __init__(self):
        print("init repository")
        mongoClient = MongoClient()
        self._mongoClient = mongoClient

    def findAll(self) -> object:
        db = self._mongoClient.test
        result = db.diamond.find({})
        print(result)
        return result

"""
re = DiamondRepository()
result = re.findAll()
for r in result:
    print(r)
"""