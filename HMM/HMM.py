import numpy as np
class HMM:
    def __init__(self, n_state, n_output):
        self.n_state = n_state
        self.n_output = n_output
        self.A = np.zeros((n_state,n_state))  # shape (N,N)
        self.B = np.zeros((n_state,n_output))  # shape (N,M)
        self.PI = np.zeros(n_state)
        self.state2int = {'B':0, 'M':1, 'E':2, 'S':3}

    def fit(self, dataset_lines):
        for line in dataset_lines:
            words = line.strip().split()
            line_status = []
            for i, word in enumerate(words):
                if len(word) == 1:
                    status = 'S'
                    coding = ord(word)
                    self.B[self.state2int[status[0]]][coding] += 1  # count status -> word freq
                else:
                    status = 'B'+'M'*(len(word)-2)+'E'  # for every multi-word item, its status is BMM**MME
                    for c in range(len(word)):
                        coding = ord(word[c])
                        self.B[self.state2int[status[c]]][coding] += 1  # count status -> word freq
                if i==0:
                    self.PI[self.state2int[status[0]]] += 1  # count status -> init freq
                line_status.extend(status)
            for i in range(1,len(line_status)):
                self.A[self.state2int[line_status[i-1]]][self.state2int[line_status[i]]] += 1  # count state i->j freq
        total = np.sum(self.PI)
        for i in range(len(self.PI)):
            if self.PI[i] == 0:
                self.PI[i] = -3.14e+100
            else:
                self.PI[i] = np.log(self.PI[i]/total)
        for i in range(len(self.A)):
            total = np.sum(self.A[i])
            for j in range(len(self.A[i])):
                if self.A[i][j] == 0:
                    self.A[i][j] = -3.14e+100
                else:
                    self.A[i][j] = np.log(self.A[i][j]/total)
        for i in range(len(self.B)):
            total = np.sum(self.B[i])
            for j in range(len(self.B[i])):
                if self.B[i][j] == 0:
                    self.B[i][j] = -3.14e+100
                else:
                    self.B[i][j] = np.log(self.B[i][j]/total)

    def predict(self, article_lines):
        partitioned_article = []
        for line in article_lines:
            # for every time k, u_k has N values; so set up a matrix u of shape (N,M), u_ij -> at time j, when state is i, the p(z_1:j, x_1:j)
            # assume we are deciding the state of time k, a.k.a. z_k; in other words, we are finding the z_k-1 that makes some p max, for every z_k;
            # the formula:
            # u_k(z_k) = max p(x_k|z_k) p(z_k|z_{k-1}) \mu_{k-1} (z_{k-1})
            u_matrix = np.zeros((self.n_state, len(line)))
            for state in range(self.n_state):
                u_matrix[state][0] = self.PI[state] + self.B[state][ord(line[0])]
            former_states = []
            for t in range(1, len(line)):
                coding = ord(line[t])
                former = []
                for state in range(self.n_state):
                    ps = [self.A[old_state][state] + u_matrix[old_state][t-1] for old_state in range(self.n_state)]
                    u_matrix[state][t] = np.max(ps) + self.B[state][coding]
                    former.append(np.argmax(ps))
                former_states.append(former)
            optim_states = []
            optim_cur = np.argmax(u_matrix[:,-1])
            optim_states.append(optim_cur)
            for t in range(len(former_states)-1, -1, -1):
                optim_former = former_states[t][optim_cur]
                optim_states.append(optim_former)
                optim_cur = optim_former
            partitioned = ''
            length = len(line)
            for i in range(length):
                partitioned += line[i]
                if optim_states[length-i-1] == self.state2int['E'] or optim_states[length-i-1] == self.state2int['S']:
                    partitioned += '|'
            partitioned_article.append(partitioned[:-1])

        return partitioned_article

def load_article_dataset(filename):
    with open(filename, encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            line = line.strip()
            lines.append(line)
    return lines

if __name__ == '__main__':
    lines = load_article_dataset("../data/HMMTrainSet.txt")
    print("loaded article lines: ", len(lines))
    model = HMM(n_state=4, n_output=65536)
    model.fit(dataset_lines=lines)
    test_lines = load_article_dataset("../data/HMMTestSet.txt")
    partitioned = model.predict(test_lines)
    for line in partitioned:
        print(line)

