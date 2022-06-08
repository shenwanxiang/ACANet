
def pIC2Nm(pIC50):
    Nm = 10**(9-pIC50)
    return Nm


def Nm2pIC(Nm):
    #pIC50 = -np.log10(Nm*10**-9)
    pIC50 = 9 - np.log10(Nm)
    return pIC50