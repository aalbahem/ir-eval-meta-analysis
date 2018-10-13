from math import log



class AgreementTest:

    def __init__(self,T,R,M,S_M):
        """
        :param R: set of system runs:
        :param M: set of metrics
        :param s_m: a map from metrics to to their metric scores for the individual topics
        """
        self.agreemnts = {}
        self.T = T
        self.R = R
        self.M = M
        self.S_M = S_M
        self.total = 0

    def improves(self,t,m,i,j):
        verdict = self.S_M[i][t][m] > self.S_M[j][t][m]

        return verdict

    def hurts(self,t,m,i,j):
        verdict = self.S_M[i][t][m] < self.S_M[j][t][m]

        return verdict

    def tie(self,t,m,i,j):
        verdict = self.S_M[i][t][m] == self.S_M[j][t][m]

        return verdict

    def agreemnet_m(self,m):
        """
        Generates all improvment generates by m

        :param m:
        :return:
        """

        self.agreemnts[m] = 0

        for i in self.R:
            for j in self.R:
                if (i==j):
                    continue

                for t in self.T:
                    if not (t in self.S_M[i].keys() and t in self.S_M[j].keys()):
                        continue
                    verdict = all(self.improves(t,n,i,j) for n in self.M) or all(self.hurts(t,n,i,j) for n in self.M) or all(self.tie(t,n,i,j) for n in self.M)

                    if (verdict):
                        self.agreemnts[m]+=1.0
                    self.total +=1.0



    def concorden_test(self,m):
        """
        Calcuates metric uniaminty using for each m in M using runs S


        :return: A map of the metirc and their unimanity.
        """

        self.agreemnet_m(m)
        return self.agreemnts;


if __name__ =="__main__":

    ag = AgreementTest()

