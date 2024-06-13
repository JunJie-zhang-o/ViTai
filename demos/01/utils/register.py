


from enum import IntEnum



class Register(IntEnum):

    @classmethod
    def getHighByteValue(cls, value):
        return (value >> 8) & 0xFF

    @classmethod
    def getLowByteValue(cls, value):
        return value & 0xFF