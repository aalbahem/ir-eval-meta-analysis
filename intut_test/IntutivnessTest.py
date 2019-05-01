import itertools

from scipy import stats

class IntutivnessTestResult(object):

    def __init__(self,MS,left,right,total,disagree_count,agree_count,left_disagree_wins,right_disagree_wins, left_intutiveness_score,right_intutiveness_score):
        self.MS =  MS
        self.left = left
        self.right = right
        self.total = total
        self.disagree_count = disagree_count
        self.agree_count = agree_count
        self.left_disagree_wins = left_disagree_wins
        self.left_intutiveness_score = left_intutiveness_score
        self.right_intutiveness_score = right_intutiveness_score
        self.right_disagree_wins = right_disagree_wins

    def compute_sig_level(self):
        if self.disagree_count > 0:
            right_sig_level = 1
            left_sig_level = 1

            assert (self.left_disagree_wins + self.right_disagree_wins) <= self.disagree_count
            dis = self.left_disagree_wins + self.right_disagree_wins
            if self.left_disagree_wins > self.right_disagree_wins:
                left_sig_level = stats.binom_test(self.left_disagree_wins, dis)

            if self.left_disagree_wins < self.right_disagree_wins:
                right_sig_level = stats.binom_test(self.right_disagree_wins, dis)

            return (left_sig_level,right_sig_level)

        return (1,1)

    def to_csv(self):
        line = "{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}"\
            .format(
            "+".join(self.MS),
            self.left + "-" + self.right,
            self.left,
            self.right,
            self.total,
            self.disagree_count,
            self.left_disagree_wins,
            self.right_disagree_wins,
            self.left_intutiveness_score,
            self.right_intutiveness_score
        )
        return line



class IntutivnessTest(object):

    """
    Setup intutivness Test using needed information

    @:param left the left metric (Mc1)
    @:param right the right metric (Mc2)
    @param MS : list of simple metrics to use in calcuating the wins
    @:param runs the list of run names
    @:param topics  the list of topic ids
    @:param data the data object that contain the evalution scores. It is expected to be a nested maps as follows

            run:
               topic:
                 metric:score
    """
    def __init__(self,left, right,MS,runs,topics,data):
        self.left = left
        self.right = right
        self.MS = MS
        self.runs= runs
        self.topics = topics
        self.data = data

    def comp_no_zero(X):
        verdict = all(x > 0 for x in X) or all(x < 0 for x in X)
        return verdict

    def comp_with_zero(X):
        verdict = all(x >= 0 for x in X) or all(x < 0 for x in X)
        return verdict

    #
    def same_no_tie(self, y, X):
        verdict = all(y * x > 0 for x in X)
        return verdict

    def same_with_tie(self, y, X):
        verdict = all(y * x >= 0 for x in X)
        return verdict

    def compute(self):
            total = 0
            ties = 0
            disagreements, dis_correct_1, dis_correct_2 = 0, 0, 0

            agreements, agree_wrong_1, agree_wrong_2 = 0, 0, 0

            for r_1, r_2 in itertools.combinations(self.runs, 2):
                if r_1 == r_2:
                    continue

                for t in self.topics:
                    if t not in self.data[r_1].keys() or t not in self.data[r_2].keys():
                        continue

                    M1  = self.left
                    M2 = self.right
                    total += 1
                    left_r1_score  = self.data[r_1][t][self.left]
                    right_r2_score = self.data[r_1][t][self.right]

                    M1_r2 = self.data[r_2][t][M1]
                    M2_r2 = self.data[r_2][t][M2]

                    M1_d = float(left_r1_score) - float(M1_r2)
                    M2_d = float(right_r2_score) - float(M2_r2)

                    MG_Ds = []



                    for mg in self.MS:
                        Mg_r1 = self.data[r_1][t][mg]
                        Mg_r2 = self.data[r_2][t][mg]
                        mg_d = float(Mg_r1) - float(Mg_r2)

                        MG_Ds.append(mg_d)

                    if (M1_d * M2_d < 0):
                        disagreements += 1

                        if self.same_no_tie(M1_d, MG_Ds):
                            dis_correct_1 += 1

                        if self.same_no_tie(M2_d, MG_Ds):
                            dis_correct_2 += 1

                        if self.same_with_tie(M1_d, MG_Ds) and self.same_with_tie(M2_d, MG_Ds):
                            ties += 1

                    else:
                        agreements += 1

                        if not self.same_with_tie(M1_d, MG_Ds):
                            agree_wrong_1 += 1

                        if not self.same_with_tie(M2_d, MG_Ds):
                            agree_wrong_2 += 1



            left_intutiveness_score = dis_correct_1 / float(max(1,disagreements))
            right_intutiveness_score = dis_correct_2 / float(max(1,disagreements))
            return IntutivnessTestResult(
                MS=self.MS,left=self.left,
                right=self.right,
                total=total,
                disagree_count=disagreements,
                left_disagree_wins=dis_correct_1,
                right_disagree_wins=dis_correct_2,
                agree_count=agreements,
                left_intutiveness_score = left_intutiveness_score,
                right_intutiveness_score = right_intutiveness_score)

