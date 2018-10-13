from math import log



class MU:

    def __init__(self,T,R,M,S_M):
        """
        :param R: set of system runs:
        :param M: set of metrics
        :param s_m: a map from metrics to to their metric scores for the individual topics
        """
        self.d_mij = {}
        self.d_Mij = {}
        self.d_mMij = {}
        self.T = T
        self.R = R
        self.M = M
        self.S_M = S_M
        self.total = float(len(self.R) * (len(self.R)-1) * len(T))

    def improves(self,t,m,i,j):
        verdict = self.S_M[i][t][m] >= self.S_M[j][t][m]

        return verdict

    def pm_ij(self,m):
        return self.d_mij[m]/self.total

    def pM_ij(self,m):
        return self.d_Mij[m]/self.total

    def pmM_ij(self,m):
        return self.d_mMij[m] / self.total


    def pmi(self,m):
        pmM = self.pmM_ij(m)
        pmij = self.pm_ij(m)
        pMij = self.pM_ij(m)


        pmi_m = log((pmM)/float(pmij * pMij),2)
        return pmi_m

    def delta_m(self,m):
        """
        Generates all improvment generates by m

        :param m:
        :return:
        """

        self.d_mij[m] = 0

        for i in self.R:
            for j in self.R:
                if (j==i):
                    continue

                for t in self.T:
                    if not (t in self.S_M[i].keys() and t in self.S_M[j].keys() ):
                        continue

                    m_i = self.S_M[i][t][m]
                    m_j = self.S_M[j][t][m]

                    if (m_i > m_j):
                      self.d_mij[m] +=1
                    if (m_i ==m_j):
                      self.d_mij[m] += 0.5


    def delta_M(self,m,M):
        """
        Generates all improvment generates by m

        :param m:
        :return:
        """

        self.d_Mij[m] = 0

        for i in self.R:
            for j in self.R:
                if (i==j):
                    continue

                for t in self.T:
                    if not (t in self.S_M[i].keys() and t in self.S_M[j].keys()):
                        continue
                    verdict = all(self.improves(t,n,i,j) for n in M)

                    if (verdict):
                        self.d_Mij[m]+=1.0


    def delta_mM(self,m,M):
        """
        Generates all improvment observed by m and M simantounsly

        :param m:
        :return:
        """

        self.d_mMij[m] = 0

        for i in self.R:
            for j in self.R:
                if (i==j):
                    continue

                for t in self.T:
                    if not (t in self.S_M[i].keys() and t in self.S_M[j].keys()):
                        continue
                    verdict = all(self.improves(t,n,i,j) for n in [m]+M)

                    if (verdict):
                        self.d_mMij[m]+=1.0




    def mu(self):
        """
        Calcuates metric uniaminty using for each m in M using runs S


        :return: A map of the metirc and their unimanity.
        """

        pmis = {}
        for m in self.M:
            self.make_float()
            self.delta_m(m)
            M_star = set(self.M)
            M_star.remove(m)
            M_star = list(M_star)
            self.delta_M(m,M_star)
            self.delta_mM(m,M_star)


        for m in self.M:
            pmi = {}
            pmi["mu"] = self.pmi(m)
            pmi["m_ij"]=self.d_mij[m]
            pmi["set_ij"] = self.d_Mij[m]
            pmi["m_set_ij"] = self.d_mMij[m]
            pmis[m] = pmi

        return pmis

    def make_float(self):
        for s in self.S_M.keys():
            for t in self.T:
                for m in self.M:
                    if t in self.S_M[s].keys():
                        self.S_M[s][t][m] = float(self.S_M[s][t][m])


if __name__ =="__main__":

    mu = MU()

