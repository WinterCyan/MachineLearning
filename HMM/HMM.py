import numpy as np
class HMM:
    def __init__(self, n_state, n_output):
        self.n_state = n_state
        self.n_output = n_output
        self.A = np.zeros(n_state,n_state)  # shape (N,N)
        self.B = np.zeros(n_state,n_output)  # shape (N,M)
        self.PI = np.zeros(n_state)
        self.state2int = {'B':0, 'M':1, 'E':2, 'S':3}

    def fit(self, dataset_lines):
        for line in dataset_lines:
            words = line.strip().split()
            status = []
            for i, word in enumerate(words):
                if len(word) == 1:
                    status = ['S']
                    coding = ord(word)
                    self.B[self.state2int[status[0]]][coding] += 1  # count status -> word freq
                else:
                    status = 'B'+'M'*(len(word)-2)+'E'  # for every multi-word item, its status is BMM**MME
                    for c in len(word):
                        coding = ord(word[c])
                        self.B[self.state2int[status[c]]][coding] += 1  # count status -> word freq
                if i==0:
                    self.PI[self.state2int[status[0]]] += 1  # count status -> init freq
            for i in (1,len(status)):
                self.A[self.state2int[status[i]]][self.state2int[status[i-1]]] += 1  # count state i->j freq
        total = np.sum(self.PI)
        for i in range(len(total)):
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

    def predict(self, article):
        pass
